import json
import math
from pathlib import Path
from time import perf_counter
import torch
import numpy as np
from tqdm import tqdm
from models.loss import PearsonLoss
from models.physnet import PhysNet
from src import config
from src.dataset import RPPGDataset, build_dataloaders, discover_windows, patient_id, split_train_val
from src.postprocess import process_recordings
from src.utils import fix_seed, hr_metrics

class TrainModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.max_epoch_num = config.epochs
        self.batch_size = config.batch_size
        self.max_shift = round(config.max_shift_sec * config.fps)
        self.model = PhysNet().to(self.device)
        self.loss_model = PearsonLoss(self.max_shift)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = self.build_scheduler()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) # FP16 for training/inference optimization
        self.best_path = config.run_dir / "physnet.pth"

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

    def forward_batch(self, batch, normalize=True):
        frames = batch[0].to(torch.float32).to(self.device)
        bvp_label = batch[1].to(torch.float32).to(self.device)
        with torch.autocast("cuda", enabled=self.use_amp):
            rppg = self.model(frames)
        rppg = rppg.float()
        if normalize:
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
            print(f"training epoch {epoch} (lr {lr:.2e})")
            train_loss = self.train_epoch(epoch, data_loader)
            self.scheduler.step()

            valid_loss, metrics = self.valid(data_loader)
            if metrics is None or not np.isfinite(metrics["corr"]):
                print("stop training: validation metrics are unavailable")
                return None
            corr = metrics["corr"]
            print(f"validation loss: {valid_loss:.4f} | HR MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} corr: {corr:.3f}")

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
                print(f"update best model! best epoch: {best_epoch} (corr {corr:.3f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} {self.config.early_stop_patience} epoch(s)")
                if epochs_without_improvement >= self.config.early_stop_patience:
                    print("early stopping: corr stalled.")
                    break

        print(f"best epoch: {best_epoch}, best corr: {best_corr:.3f}")
        
        return {
            "best_epoch": best_epoch,
            "best_corr": best_corr,
            "epochs": epoch_history,
        }

    def valid(self, data_loader):
        print("validating")
        self.model.eval()
        valid_loss = []
        all_predictions = []
        with torch.no_grad():
            bar = tqdm(data_loader["valid"], ncols=80)
            for batch in bar:
                bar.set_description("validation")
                rppg, bvp_label = self.forward_batch(batch, normalize=False)
                loss = self.loss_model(rppg, bvp_label)
                valid_loss.append(loss.item())
                all_predictions.append(rppg.cpu().numpy())
                bar.set_postfix(loss=loss.item())

        if not all_predictions:
            print("no validation windows")
            return None, None
        predictions = np.concatenate(all_predictions)
        subset = data_loader["valid"].dataset
        files = [subset.dataset.files[index] for index in subset.indices]
        processed = process_recordings(files, predictions)
        if processed is None:
            return None, None
        predictions, targets, fps = processed
        max_shift = np.rint(np.asarray(fps) * self.config.max_shift_sec).astype(int)
        metrics = hr_metrics(predictions, targets, fps, max_shift=max_shift)
        return float(np.mean(valid_loss)), metrics

    def save_checkpoint(self, path):
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print("Saved checkpoint:", path)

def main():
    started = perf_counter()
    fix_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {config.run_dir}")

    files = discover_windows(config.processed_path)
    if not files:
        print(f"no windows found in {config.processed_path}")
        return
    dataset = RPPGDataset(files)
    train_indices, val_indices = split_train_val(files)
    if not train_indices or not val_indices:
        print("training and validation must each contain at least one subject")
        return

    print(f"windows: {len(files)} | train: {len(train_indices)} | val: {len(val_indices)}")

    train_loader, validation_loader = build_dataloaders(dataset, train_indices, val_indices, config.batch_size, config.num_workers)
    data_loader = {"train": train_loader, "valid": validation_loader}

    trainer = TrainModel(config)
    run_history = trainer.train(data_loader)
    if run_history is None:
        return

    history = {
        "architecture": "physnet",
        "params": sum(p.numel() for p in trainer.model.parameters()),
        "loss": {
            "name": type(trainer.loss_model).__name__,
            "max_shift_frames": trainer.max_shift,
        },
        "optimizer": type(trainer.optimizer).__name__,
        "scheduler": "cosine_with_warmup",
        "selection_metric": "corr",
        **run_history,
    }

    history_path = config.run_dir / "history.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("saved history:", history_path)
    print(f"done. best weights at {trainer.best_path}")


if __name__ == "__main__":
    main()
