import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from models.loss import ShiftLoss
from models.physnet import PhysNet
from models.factorizephys import FactorizePhys
from src import config
from src.dataset import (
    RPPGDataset,
    build_dataloaders,
    discover_windows,
    split_train_val,
)
from src.utils import hr_metrics, fix_seed

ARCHITECTURES = {
    "physnet": PhysNet,
    "factorizephys": FactorizePhys,
}

def build_model(name):
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}. "
                         f"Available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name]()

class TrainModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.max_epoch_num = config.epochs
        self.model_dir = config.model_dir
        self.model_file_name = config.model_file_name
        self.batch_size = config.batch_size
        self.fps = config.fps_target
        self.max_shift = round(config.max_shift_sec * config.fps_target)
        self.model = build_model(config.architecture).to(self.device)
        print(f"model: {config.architecture} "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)")
        self.loss_model = ShiftLoss(self.max_shift)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = self.build_scheduler()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_path = os.path.join(self.model_dir, f"{self.model_file_name}_best.pth")

    def build_scheduler(self):
        warmup = self.config.warmup_epochs
        total = self.config.epochs
        floor = self.config.min_lr / self.config.lr

        def lr_lambda(epoch):
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = (epoch - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return floor + (1.0 - floor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def forward_batch(self, batch):
        frames = batch[0].to(torch.float32).to(self.device)
        bvp_label = batch[1].to(torch.float32).to(self.device)
        with torch.autocast("cuda", enabled=self.use_amp):
            rppg = self.model(frames)
        rppg = rppg.float()
        rppg = (rppg - rppg.mean(dim=1, keepdim=True)) / (rppg.std(dim=1, keepdim=True) + config.eps)
        bvp_label = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + config.eps)
        return rppg, bvp_label

    def train_step(self, batch):
        rppg, bvp_label = self.forward_batch(batch)
        loss = self.loss_model(rppg, bvp_label)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if self.config.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def train_epoch(self, epoch, data_loader):
        train_loss = []
        self.model.train()
        bar = tqdm(data_loader["train"], ncols=80)
        for batch in bar:
            bar.set_description(f"Train epoch {epoch}")
            loss = self.train_step(batch)
            train_loss.append(loss.item())
            bar.set_postfix(loss=loss.item())
        return float(np.mean(train_loss))

    def train(self, data_loader):
        best_corr = None
        best_epoch = 0
        epochs_without_improvement = 0
        epoch_history = []

        for epoch in range(self.max_epoch_num):
            lr = float(self.optimizer.param_groups[0]["lr"])
            print(f"\n==== Training epoch {epoch} (lr {lr:.2e}) ====")
            train_loss = self.train_epoch(epoch, data_loader)
            self.scheduler.step()

            valid_loss, metrics = self.valid(data_loader)
            corr = metrics["corr"]
            print(f"validation loss: {valid_loss:.4f} | "
                  f"HR MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | "
                  f"corr: {corr:.3f}")
            epoch_history.append({
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "hr_mae": metrics["mae"],
                "hr_rmse": metrics["rmse"],
                "corr": corr,
            })
            if best_corr is None or corr > best_corr:
                best_corr = corr
                best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(self.best_path)
                print(f"Update best model! Best epoch: {best_epoch} (corr {corr:.3f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}"
                      f"/{self.config.early_stop_patience} epoch(s)")
                if epochs_without_improvement >= self.config.early_stop_patience:
                    print("Early stopping: corr stalled.")
                    break

        print(f"\nbest epoch: {best_epoch}, best corr: {best_corr:.3f}")
        return {
            "best_epoch": best_epoch,
            "best_corr": best_corr,
            "epochs": epoch_history,
        }

    def valid(self, data_loader):
        print("\n ==== Validating ====")
        self.model.eval()
        valid_loss = []
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            bar = tqdm(data_loader["valid"], ncols=80)
            for batch in bar:
                bar.set_description("Validation")
                rppg, bvp_label = self.forward_batch(batch)
                loss = self.loss_model(rppg, bvp_label)
                valid_loss.append(loss.item())
                all_predictions.append(rppg.cpu().numpy())
                all_targets.append(bvp_label.cpu().numpy())
                bar.set_postfix(loss=loss.item())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        metrics = hr_metrics(predictions, targets, self.fps, max_shift=self.max_shift)
        return float(np.mean(valid_loss)), metrics

    def save_checkpoint(self, path):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print("Saved checkpoint:", path)

def main():
    fix_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {config.run_dir}")

    files = discover_windows(config.windows_dir)
    dataset = RPPGDataset(files)
    train_indices, val_indices = split_train_val(files)
    print(f"windows: {len(files)} | train: {len(train_indices)} "
          f"| val: {len(val_indices)}")

    train_loader, validation_loader = build_dataloaders(
        dataset, train_indices, val_indices, config.batch_size, config.num_workers
    )
    data_loader = {"train": train_loader, "valid": validation_loader}

    trainer = TrainModel(config)
    run_history = trainer.train(data_loader)

    history = {
        "architecture": config.architecture,
        "params": sum(p.numel() for p in trainer.model.parameters()),
        "windows": len(files),
        "train_windows": len(train_indices),
        "val_windows": len(val_indices),
        "hyperparams": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "min_lr": config.min_lr,
            "warmup_epochs": config.warmup_epochs,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "window": config.window,
            "max_shift_sec": config.max_shift_sec,
        },
        **run_history,
    }
    history_path = config.run_dir / config.history_file_name
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("Saved history:", history_path)

    print(f"\nDone. Best weights at {trainer.best_path}")


if __name__ == "__main__":
    main()
