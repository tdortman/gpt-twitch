import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torch import Tensor
from torch.amp import GradScaler, autocast
import torch.distributed as dist


class Trainer:
    def __init__(
        self,
        model: Module,
        train_data: DataLoader,
        optimizer: Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.scaler = GradScaler("cuda")

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source: Tensor, targets: Tensor):
        self.optimizer.zero_grad()
        with autocast("cuda"):
            _, loss = self.model(source, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def _run_epoch(self, epoch: int):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)

        # This is not ideal, but I literally do not have the resources to go through all the data
        # as part of every single epoch. This should be "good enough" for now.
        source, targets = next(iter(self.train_data))
        source = source.to(self.gpu_id)
        targets = targets.to(self.gpu_id)
        loss = self._run_batch(source, targets)

        if self.gpu_id == 0:
            print(f"Epoch {epoch} | Batchsize: {b_sz} | Loss: {loss:.4f}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run + 1, max_epochs + 1):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
