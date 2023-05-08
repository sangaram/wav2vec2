from wav2vec2.models import Wav2Vec2Model
from wav2vec2.data import LibriSpeechDatasetWrapper
from wav2vec2.losses import PretrainingLoss
import torch
from tqdm.auto import tqdm
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
import json



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

def ddp_setup(backend):
    init_process_group(backend=backend)

class DistributedPreTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        max_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        loss_fn: _Loss,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.max_epochs = max_epochs
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"gpu:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        output = self.model(batch.to(self.gpu_id))
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
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        loss = .0
        for batch in self.train_data:
            loss += self._run_batch(batch.to(self.gpu_id))
        
        loss /= len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Loss: {loss}")

        self.scheduler.step()
        return loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self):
        if os.path.exists(os.path.join(os.getcwd(), self.snapshot_path)):
            print(f"Loading pre-existing snapshot at {os.path.join(os.getcwd(), self.snapshot_path)} ....")
            snapshot = torch.load(os.path.join(os.getcwd(), self.snapshot_path))
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        
        losses = []
        self.model.train()
        for epoch in range(self.epochs_run, self.max_epochs):
            loss = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                losses.append(loss)
                self._save_snapshot(epoch)
        
        self.model.train(False)
        return losses   


class PreTrainer:
    def __init__(
        self,
        model:nn.Module,
        dataset:Dataset,
        batch_size:int,
        epochs:int,
        loss_fn:PretrainingLoss,
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
            batch_size=batch_size,
            shuffle=False
        )
        self.epochs = epochs
        self.loss_fn = loss_fn
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
        loss = self.loss_fn(x, y, logits)
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
    parser = argparse.ArgumentParser(description="CLI arguments parser")
    parser.add_argument("--distributed", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--max_sample_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--save_every", type=int)

    args = parser.parse_args()
    
    try:
        dataset = LibriSpeechDatasetWrapper(
            root = args.data,
            max_sample_size=args.max_sample_size if args.max_sample_size else 250_000
        )
    except:
        print(f"LibriSpeech folder not found at {args.data}. Downloading LibriSpeech dataset into current directory ({os.getcwd()})")
        dataset = LibriSpeechDatasetWrapper(
            root = args.data,
            max_sample_size=args.max_sample_size if args.max_sample_size else 250_000,
            download=True
        )
    
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training the model on device: {device}")

    config_path = args.config
    if os.path.isfile(os.path.join(os.getcwd(), config_path)):
        config = json.load(open(os.path.join(os.getcwd(), config_path)))
    else:
        print(f"The given configuration file path does not exist or is not a proper file, using the default config at {os.path.join(os.getcwd(), './configs/base.json')}")
        config = json.load(open(os.path.join(os.getcwd(), './configs/base.json')))
    
    model = Wav2Vec2Model(
        fe_dim=config["fe_dim"],
        fe_kernel_sizes=config["fe_kernel_sizes"],
        fe_strides=config["fe_strides"],
        p=config["p"],
        with_mask=bool(config["with_mask"]),
        mask_span=config["mask_span"],
        drop_prob=config["drop_prob"],
        rpe_kernel_size=config["rpe_kernel_size"],
        rpe_groups=config["rpe_groups"],
        quantize=bool(config["quantize"]),
        qt_n_groups=config["qt_n_groups"],
        qt_n_entries=config["qt_n_entries"],
        final_dim=config["final_dim"],
        temperature=config["temperature"],
        tfe_dff=config["tfe_dff"],
        tfe_num_heads=config["tfe_num_heads"],
        tfe_num_layers=config["tfe_num_layers"],
        tfe_activation=config["tfe_activation"],
        activation=config["activation"]
    )
    
    loss_fn = PretrainingLoss(alpha=0.1, temperature=0.1, n_distractors=100)

    optimizer = optim.Adam(params=model.parameters())

    scheduler = LRScheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        max_lr=1e-5
    )

    if args.distributed == 1:
        backend = "gloo" if device == "cpu" else "nccl"
        ddp_setup(backend)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
        )

        trainer = DistributedPreTrainer(
            model=model,
            train_data=dataloader,
            max_epochs=args.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            save_every=args.save_every,
            snapshot_path=args.save_path,
        )
        trainer.train()
        destroy_process_group()
        
    else:
        trainer = PreTrainer(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            log_period=5,
            save_period=5,
            save_path=args.save_path,
            device=device
        )
        
        trainer.train()