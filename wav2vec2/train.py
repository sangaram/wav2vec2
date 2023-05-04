from wav2vec2.models import Wav2Vec2Model
from wav2vec2.data import LibriSpeechDatasetWrapper
from wav2vec2.losses import PretrainingLoss
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():
    init_process_group(backend="gloo")

class PreTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: _Loss,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"mps:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        output = self.model(batch)
        x = output['x']
        y = output['y']
        mask = output['mask']
        logits = output['codebook_logits']
        B, _, C = x.shape
        x = x[mask].view(B, -1, C)
        y = y[mask].view(B, -1, C)
        loss = self.loss_fn(x, y, logits)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            self._run_batch(batch.to(self.gpu_id))

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    dataset = LibriSpeechDatasetWrapper(
        root="..",
        max_sample_size=200
    )
    model = Wav2Vec2Model(
        fe_n_blocks=3,
        fe_dim=16,
        fe_kernel_sizes=[4, 4, 2],
        fe_strides=[1, 1, 1],
        p=0.063,
        with_mask=True,
        mask_span=4,
        drop_prob=0.05,
        rpe_kernel_size=4,
        rpe_groups=4,
        qt_n_groups=4,
        qt_n_entries=16,
        final_dim=16,
        temperature=1.3,
        tfe_dff=32,
        tfe_num_heads=2,
        tfe_num_layers=4,
        tfe_activation='relu',
        activation='gelu'
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = PretrainingLoss(alpha=0.1, temperature=1.3, n_distractors=5)
    return dataset, model, optimizer, loss_fn


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer, loss_fn = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = PreTrainer(model, train_data, optimizer, loss_fn, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size)

