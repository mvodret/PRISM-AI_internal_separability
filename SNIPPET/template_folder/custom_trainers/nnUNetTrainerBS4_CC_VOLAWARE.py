from pathlib import Path
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, smooth=1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        if torch.is_tensor(target) and target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)

        probs = torch.softmax(logits, dim=1)
        p = probs[:, 1]
        t = target[:, 0].float()

        spatial = tuple(range(1, p.ndim))
        tp = (p * t).sum(dim=spatial)
        fp = (p * (1 - t)).sum(dim=spatial)
        fn = ((1 - p) * t).sum(dim=spatial)

        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tv.mean()


class _EarlyStopException(RuntimeError):
    pass


class nnUNetTrainerBS4_CC_VOLAWARE(nnUNetTrainer):
    oversample_foreground_percent = 0.6
    skip_end_of_training_prediction = False
    best_eps = 1e-6
    num_epochs_per_validation = 1
    early_stopping_patience = 25
    early_stopping_min_delta = 1e-3
    early_stopping_warmup = 25

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._best_global = -1.0
        self._es_best = -1.0
        self._es_bad_epochs = 0
        self._it_count = 0
        self._it_times = []

    def initialize(self):
        super().initialize()
        self.batch_size = 8
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        self.print_to_log_file("[INIT] nnUNetTrainerBS4_CC_VOLAWARE")
        self.print_to_log_file(f"[INIT] batch_size={self.batch_size}")
        self.print_to_log_file(f"[PATHS] output_folder={self.output_folder}")
        self.print_to_log_file(f"[ENV] nnUNet_results={os.environ.get('nnUNet_results','<unset>')}")
        try:
            b0 = next(iter(self.dataloader_train))["data"].shape[0]
            self.print_to_log_file(f"[DEBUG] train_loader_batch0={b0}")
        except Exception as e:
            self.print_to_log_file(f"[DEBUG] cannot read train loader batch: {e}")
        out_dir = Path(self.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        test_p = out_dir / "__write_test.txt"
        test_p.write_text("ok")
        try:
            test_p.unlink()
        except Exception:
            pass
        self.print_to_log_file(f"[WriteTest] OK in {out_dir}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._it_count = 0
        self._it_times = []

    def _build_loss(self):
        ce = nn.CrossEntropyLoss()
        tv = TverskyLoss(alpha=0.5, beta=0.5)

        class DeepSupervisionCETwitter(nn.Module):
            def __init__(self, ce, tv, weights):
                super().__init__()
                self.ce = ce
                self.tv = tv
                self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

            def forward(self, logits, target):
                outs = list(logits) if isinstance(logits, (list, tuple)) else [logits]
                if isinstance(target, (list, tuple)):
                    tgts = list(target)
                else:
                    tgts = [target] * len(outs)
                if len(tgts) != len(outs):
                    tgts = [tgts[0]] * len(outs)
                w = self.weights[:len(outs)].to(outs[0].device)
                w = w / (w.sum() + 1e-8)
                loss = 0.0
                for oi, ti, wi in zip(outs, tgts, w):
                    if ti.ndim == oi.ndim - 1:
                        ti = ti.unsqueeze(1)
                    if ti.shape[2:] != oi.shape[2:]:
                        ti = F.interpolate(ti.float(), size=oi.shape[2:], mode="nearest")
                    y = ti[:, 0].long()
                    loss_i = self.ce(oi, y) + self.tv(oi, ti)
                    loss = loss + wi * loss_i
                return loss

        weights = [1.0 / (2 ** i) for i in range(10)]
        return DeepSupervisionCETwitter(ce, tv, weights)

    def train_step(self, batch):
        self._it_count += 1
        use_cuda = (self.device.type == "cuda")
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            t0 = time.perf_counter()

        out = super().train_step(batch)

        if use_cuda:
            ender.record()
            torch.cuda.synchronize()
            dt = starter.elapsed_time(ender) / 1000.0
        else:
            dt = time.perf_counter() - t0

        self._it_times.append(dt)
        N = 20
        if len(self._it_times) >= N:
            avg = sum(self._it_times) / len(self._it_times)
            it_per_epoch = int(getattr(self, "num_iterations_per_epoch", 0)) or None
            if it_per_epoch is not None:
                remaining = max(it_per_epoch - self._it_count, 0)
                eta_min = (remaining * avg) / 60.0
                self.print_to_log_file(
                    f"[IterTimeAvg{N}] {avg:.3f} s/iter | iter {self._it_count}/{it_per_epoch} | ETA epoch ~{eta_min:.1f} min (BS={self.batch_size})"
                )
            else:
                self.print_to_log_file(
                    f"[IterTimeAvg{N}] {avg:.3f} s/iter | iter {self._it_count} (BS={self.batch_size})"
                )
            self._it_times = []
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        try:
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.print_to_log_file(f"[LR] {lr:.6e}")
        except Exception:
            pass

    def _to_cpu_float_tensor(self, x):
        if torch.is_tensor(x):
            return x.detach().float().cpu()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().cpu()
        return torch.tensor(x).float().cpu()

    def _extract_tp_fp_fn(self, outputs):
        tp = fp = fn = None
        for o in outputs:
            if not isinstance(o, dict):
                continue
            for k_tp, k_fp, k_fn in [("tp_hard", "fp_hard", "fn_hard"), ("tp", "fp", "fn")]:
                if k_tp in o and k_fp in o and k_fn in o:
                    _tp = self._to_cpu_float_tensor(o[k_tp])
                    _fp = self._to_cpu_float_tensor(o[k_fp])
                    _fn = self._to_cpu_float_tensor(o[k_fn])
                    tp = _tp if tp is None else tp + _tp
                    fp = _fp if fp is None else fp + _fp
                    fn = _fn if fn is None else fn + _fn
                    break
        return tp, fp, fn

    def _global_fg_dice(self, val_outputs):
        tp, fp, fn = self._extract_tp_fp_fn(val_outputs)
        if tp is None:
            return None, None, None, None
        tp_s = float(tp.sum().item())
        fp_s = float(fp.sum().item())
        fn_s = float(fn.sum().item())
        g = (2.0 * tp_s) / (2.0 * tp_s + fp_s + fn_s + 1e-8)
        return g, tp_s, fp_s, fn_s

    def _save_best_global_checkpoint(self):
        out_dir = Path(self.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_dst = out_dir / "checkpoint_best_global.pth"
        final_dst = out_dir / "checkpoint_final.pth"
        self.save_checkpoint(str(best_dst))
        shutil.copy2(best_dst, final_dst)
        self.print_to_log_file(
            f"[BestGlobal] SAVED -> {best_dst.name} ({best_dst.stat().st_size} bytes)"
        )
        self.print_to_log_file(
            f"[BestGlobal] ALIAS  -> {final_dst.name} ({final_dst.stat().st_size} bytes)"
        )

    def _load_best_global_checkpoint(self):
        ckpt = Path(self.output_folder) / "checkpoint_best_global.pth"
        if not ckpt.exists():
            self.print_to_log_file(f"[BestGlobal] checkpoint not found: {ckpt}")
            return False
        self.print_to_log_file(f"[BestGlobal] loading {ckpt.name} for final prediction")
        self.load_checkpoint(str(ckpt))
        return True

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        tp, fp, fn = self._extract_tp_fp_fn(val_outputs)
        if tp is not None:
            pred_fg = float((tp + fp).sum().item())
            gt_fg = float((tp + fn).sum().item())
            self.print_to_log_file(f"[Metric] pred_fg_vox={pred_fg:.1f} | gt_fg_vox={gt_fg:.1f}")
        g, tp_s, fp_s, fn_s = self._global_fg_dice(val_outputs)
        if g is None:
            self.print_to_log_file("[Metric] global_fg_dice unavailable (tp/fp/fn not found)")
            return
        self.print_to_log_file(f"[Metric] global_fg_dice={g:.6f} | TP={tp_s:.1f} FP={fp_s:.1f} FN={fn_s:.1f}")
        if g > self._best_global + self.best_eps:
            prev = self._best_global
            self._best_global = g
            cfg = getattr(self, "configuration_name", getattr(self, "configuration", "<unknown>"))
            self.print_to_log_file(
                f"[BestGlobal] {g:.6f} (prev {prev:.6f}) | cfg={cfg} -> updating checkpoint_best_global.pth"
            )
            self._save_best_global_checkpoint()
        epoch = int(getattr(self, "current_epoch", 0))
        warmup = int(getattr(self, "early_stopping_warmup", 0))
        if epoch < warmup:
            self.print_to_log_file(f"[EarlyStop] warmup epoch {epoch}/{warmup} (no early stop)")
            return
        min_delta = float(getattr(self, "early_stopping_min_delta", 0.0))
        patience = int(getattr(self, "early_stopping_patience", 50))
        if g > self._es_best + min_delta:
            self._es_best = g
            self._es_bad_epochs = 0
            self.print_to_log_file(f"[EarlyStop] global_fg_dice improved to {g:.6f} (reset patience)")
        else:
            self._es_bad_epochs += 1
            self.print_to_log_file(
                f"[EarlyStop] no improvement (best {self._es_best:.6f}) bad_epochs={self._es_bad_epochs}/{patience}"
            )
            if self._es_bad_epochs >= patience:
                self.print_to_log_file(f"[EarlyStop] STOP triggered on global_fg_dice at epoch={epoch}")
                raise _EarlyStopException("Early stopping triggered")

    def run_training(self):
        try:
            out = super().run_training()
            loaded = self._load_best_global_checkpoint()
            if loaded:
                self.perform_actual_validation()
            else:
                self.print_to_log_file("[FinalPred] best global checkpoint unavailable, skipping final prediction")
            return out
        except _EarlyStopException as e:
            try:
                self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))
            except Exception:
                pass
            self.print_to_log_file(f"[EarlyStop] Training stopped: {e}")
            loaded = self._load_best_global_checkpoint()
            if loaded:
                try:
                    self.perform_actual_validation()
                except Exception as ee:
                    self.print_to_log_file(f"[FinalPred] perform_actual_validation failed: {repr(ee)}")
            else:
                self.print_to_log_file("[FinalPred] best global checkpoint unavailable, skipping final prediction")
            return

    def perform_actual_validation(self, *args, **kwargs):
        if getattr(self, "skip_end_of_training_prediction", False):
            self.print_to_log_file("[Skip] perform_actual_validation (end-of-training prediction/export)")
            return
        return super().perform_actual_validation(*args, **kwargs)
