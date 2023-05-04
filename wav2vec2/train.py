from wav2vec2.models import Wav2Vec2Model
from wav2vec2.data import LibriSpeechDatasetWrapper
from wav2vec2.losses import PretrainingLoss
import torch
from tqdm.auto import tqdm
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os



class LRScheduler:
    def __init__(self, optimizer:optim.Optimizer, epochs:int, max_lr:float):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.epochs = epochs
        self.epoch = 0
        self.warmup_steps = int(epochs*0.08)
    
    def step(self):
        if self.epoch <= self.warmup_steps:
            # Warm up steps
            for group in self.optimizer.param_groups:
                group["lr"] = self.max_lr * (self.warmup_steps ** -0.5) * (self.epoch ** 0.5)
        else:
            # Linear decay after warm up steps
            decay = 1 - (self.epoch - self.warmup_steps) / float(self.epochs - self.warmup_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = self.max_lr * decay 
        
        self.epoch += 1

def ddp_setup():
    init_process_group(backend="gloo")

class DistributedPreTrainer:
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
        loc = f"cpu:{self.gpu_id}"
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
    device = torch.device("mps")
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
    ).to(device)
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
    trainer = DistributedPreTrainer(model, train_data, optimizer, loss_fn, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size)
"""

class PreTrainer:
    def __init__(
        self,
        model:nn.Module,
        dataset:Dataset,
        batch_size:int,
        epochs:int,
        loss:PretrainingLoss,
        optimizer:optim.Optimizer,
        scheduler,
        log_period:int,
        save_period:int,
        save_path:str,
        device="cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset,
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size
        )
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_period = log_period
        self.save_period = save_period
        self.save_path = save_path

    def _train_batch(self, batch:torch.Tensor):
        result = self.model(batch)
        x = result['x']
        y = result['y']
        mask = result['mask']
        logits = result['codebook_logits']
        B, T, C = x.shape
        x = x[mask].view(B, -1, C)
        y = y[mask].view(B, -1, C)
        loss = self.loss(x, y, logits)
        return loss
    
    def _run_epoch(self):
        loss = .0
        n_batch = len(self.dataloader)
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            batch_loss = self._train_batch(batch)
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss
        self.scheduler.step()

        loss /= n_batch
        return loss

    def train(self):
        if os.path.exists(os.path.join(os.getcwd(), self.save_path)):
            state_dict = torch.load(os.path.join(os.getcwd(), self.save_path))
            self.model.load_state_dict(state_dict)
        
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            loss = self._run_epoch()
            if epoch % self.log_period == 0:
                print(f"Epoch: {epoch} | loss: {loss:.4f}")
            
            if epoch % self.save_period:
                print(f"Saving checkpoint to {self.save_path} ...")
                torch.save(self.model.state_dict(), self.save_path)
        
        self.model.train(False)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = Wav2Vec2Model(
        fe_n_blocks=7,
        fe_dim=512,
        fe_kernel_sizes=[10,3,3,3,3,2,2],
        fe_strides=[5,2,2,2,2,2,2],
        p=0.065,
        with_mask=True,
        mask_span=10,
        drop_prob=0.05,
        rpe_kernel_size=128,
        rpe_groups=16,
        qt_n_groups=2,
        qt_n_entries=320,
        final_dim=768,
        temperature=0.5,
        tfe_dff=3072,
        tfe_num_heads=8,
        tfe_num_layers=12,
        tfe_activation='relu',
        activation='gelu'
    )

    loss = PretrainingLoss(alpha=0.1, temperature=0.1, n_distractors=100)

    optimizer = optim.Adam(params=model.parameters())

    epochs = 200

    scheduler = LRScheduler(
        optimizer=optimizer,
        epochs=epochs,
        max_lr=1e-5
    )

    dataset = LibriSpeechDatasetWrapper(
        root="../..",
        max_sample_size=250_000
    )

    trainer = PreTrainer(
        model=model,
        dataset=dataset,
        batch_size=16,
        epochs=20,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        log_period=5,
        save_period=5,
        save_path="checkpoint.pt",
        device=device
    )
    
    trainer.train()
    

