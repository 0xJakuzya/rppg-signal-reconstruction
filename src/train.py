import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.loss import ShiftLoss
from models.physnet import PhysNet
from src import config
from src.dataset import (
    RPPGDataset,
    build_dataloaders,
    discover_windows,
    split_train_val,
)
from src.utils import hr_metrics, fix_seed

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
        self.model = PhysNet().to(self.device)
        self.loss_model = ShiftLoss(self.max_shift)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_path = os.path.join(self.model_dir, f"{self.model_file_name}_best.pth")

    def load_weights(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        print("Resumed from", checkpoint_path)

    def forward_batch(self, batch):
        frames = batch[0].to(torch.float32).to(self.device)
        bvp_label = batch[1].to(torch.float32).to(self.device)
        with torch.autocast("cuda", enabled=self.use_amp):
            rppg = self.model(frames)
        rppg = rppg.float()
        rppg = (rppg - rppg.mean(dim=1, keepdim=True)) / (rppg.std(dim=1, keepdim=True) + config.eps)
        bvp_label = (bvp_label - bvp_label.mean(dim=1, keepdim=True)) / (bvp_label.std(dim=1, keepdim=True) + config.eps)
        return rppg, bvp_label

    def train(self, data_loader, phase_name="", stage_lr=None, use_onecycle=False):
        stage_lr = self.config.lr if stage_lr is None else stage_lr
        for group in self.optimizer.param_groups:
            group["lr"] = stage_lr
        if use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=stage_lr,
                epochs=self.max_epoch_num,
                steps_per_epoch=len(data_loader["train"]),
            )
        else:
            scheduler = None
        best_corr = None
        best_epoch = 0
        epochs_without_improvement = 0
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        epoch_history = []
        tag = f" [{phase_name}]" if phase_name else ""

        for epoch in range(self.max_epoch_num):
            print(f"\n==== Training epoch {epoch}{tag} ====")
            train_loss = []
            self.model.train()
            bar = tqdm(data_loader["train"], ncols=80)
            for batch in bar:
                bar.set_description(f"Train epoch {epoch}")
                rppg, bvp_label = self.forward_batch(batch)
                loss = self.loss_model(rppg, bvp_label)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if scheduler is not None:
                    scheduler.step()

                train_loss.append(loss.item())
                lrs.append(self.optimizer.param_groups[0]["lr"])
                bar.set_postfix(loss=loss.item())
            mean_training_losses.append(float(np.mean(train_loss)))

            if self.config.use_last_epoch:
                self.save_checkpoint(self.best_path)
            else:
                valid_loss, metrics = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                corr = metrics["corr"]
                print(f"validation loss: {valid_loss:.4f} | "
                      f"HR MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | "
                      f"corr: {corr:.3f}")
                epoch_history.append({
                    "phase": phase_name,
                    "epoch": epoch,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "train_loss": mean_training_losses[-1],
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
                        print(f"Early stopping{tag}: corr stalled, moving on.")
                        break

        if not self.config.use_last_epoch:
            print(f"best epoch{tag}: {best_epoch}, best corr: {best_corr:.3f}")
        if self.config.plot_losses_and_lr:
            suffix = f"_{phase_name}" if phase_name else ""
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, suffix)
        if phase_name:
            phase_path = os.path.join(self.model_dir, f"{self.model_file_name}_{phase_name}.pth")
            self.save_checkpoint(phase_path)
        return {
            "phase": phase_name,
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

    def plot_losses_and_lrs(self, train_losses, valid_losses, lrs, suffix=""):
        epochs = range(len(train_losses))
        figure, (loss_axis, lr_axis) = plt.subplots(1, 2, figsize=(12, 4))

        loss_axis.plot(epochs, train_losses, label="train")
        if valid_losses:
            loss_axis.plot(epochs, valid_losses, label="valid")
        loss_axis.set_xlabel("epoch")
        loss_axis.set_ylabel("loss")
        loss_axis.set_title("Loss")
        loss_axis.legend()

        lr_axis.plot(range(len(lrs)), lrs)
        lr_axis.set_xlabel("step")
        lr_axis.set_ylabel("lr")
        lr_axis.set_title("Learning rate")

        figure.tight_layout()
        out_path = os.path.join(self.model_dir, f"{self.model_file_name}{suffix}_curves.png")
        figure.savefig(out_path)
        plt.close(figure)
        print("Saved training curves:", out_path)

def main(resume=None):
    fix_seed(config.seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)

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
    if resume:
        trainer.load_weights(resume)
        phases = [
            ("phase1", config.mid_lr, False),
            ("phase2", config.finetune_lr, False),
        ]
    else:
        phases = [
            ("phase0", config.lr, True),
            ("phase1", config.mid_lr, False),
            ("phase2", config.finetune_lr, False),
        ]

    history = {
        "windows": len(files),
        "train_windows": len(train_indices),
        "val_windows": len(val_indices),
        "phases": [],
    }
    for phase_name, phase_lr, use_onecycle in phases:
        print(f"\n#### {phase_name}: lr={phase_lr:g}"
              f"{' + OneCycle' if use_onecycle else ''} ####")
        phase_history = trainer.train(data_loader, phase_name=phase_name,
                                      stage_lr=phase_lr, use_onecycle=use_onecycle)
        history["phases"].append(phase_history)
        history_path = os.path.join(config.model_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print("Saved history:", history_path)

    print(f"\nDone. Best weights at {trainer.best_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None,
                        help="checkpoint .pth to continue from (fine-tune on grown data)")
    args = parser.parse_args()
    main(resume=args.resume)
