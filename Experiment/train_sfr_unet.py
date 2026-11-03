"""Exp1/Exp2 逐格 SFRUNet 共同训练核心：物理损失、固定选模指标及完整断点。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from Experiment.SFRUNet import SFRUNet
from Experiment.cell_sfr_dataset import CellSFRDataset, validate_cell_splits
from Experiment.cell_sfr_objective import CellSFRLoss
from raw_data_processing.metadata_schema import sha256_file


@dataclass
class TrainConfig:
    """配置使用独立字段；不继承标量 ExperimentSetting.json 的隐式默认值。"""

    data_exp: str = ""
    cache_dir: str = ".cache"
    cell_exp: str = ""  # 空时使用 data_exp；文件名为 <cell_exp><split>_cells.pt。
    cell_cache_dir: str = ""  # 空时使用 cache_dir。
    output_root: str = "Results"
    run_name: str = ""
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 4
    epochs: int = 100
    patience: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = .1
    min_lr_ratio: float = .01
    grad_clip: float = 1.0
    base_ch: int = 32
    encoder_depths: tuple = (2, 2)
    decoder_depths: tuple = (2, 2)
    bottleneck_depth: int = 2
    pos_num_bands: int = 4
    pos_max_freq: float = 4.0
    p_drop: float = 0.0
    se_reduction: int = 4
    total_weight: float = 1.0
    negative_weight: float = 1.0
    amp: str = "none"
    tensorboard: bool = True
    resume: str = ""
    stop_after_epochs: int = 0  # 仅暂停本次执行，不改变预先设定的学习率日程。

    def validate(self):
        for name in ("data_exp", "cell_exp", "run_name"):
            value = getattr(self, name)
            if value and (Path(value).name != value or any(c in value for c in "/\\:") or value in {".", ".."}):
                raise ValueError(f"{name} 必须是单层名称/前缀。")
        if not self.data_exp:
            raise ValueError("必须指定 data_exp。")
        for name in ("cpu_threads", "epochs", "batch_size", "base_ch", "bottleneck_depth", "pos_num_bands", "se_reduction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数。")
        for name in ("patience", "stop_after_epochs", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须为非负整数。")
        if self.seed >= 2**32:
            raise ValueError("seed 必须小于 2**32。")
        for name in ("lr", "grad_clip", "pos_max_freq"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数。")
        for name in ("weight_decay", "total_weight", "negative_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数。")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.p_drop < 1:
            raise ValueError("warmup_ratio/p_drop 范围 [0,1)，min_lr_ratio 范围 [0,1]。")
        if self.pos_max_freq < 1 or self.amp not in {"none", "bfloat16"}:
            raise ValueError("pos_max_freq>=1；amp 只支持 none/bfloat16。")
        for name in ("encoder_depths", "decoder_depths"):
            value = getattr(self, name)
            if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value):
                raise ValueError(f"{name} 需要两个正整数。")
        if not isinstance(self.tensorboard, bool):
            raise ValueError("tensorboard 必须为布尔值。")
        return self


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_save(path, value, *, json_format=False):
    """在任务输出目录同盘写临时文件，再原子替换本次任务自己的输出。"""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".writing_", dir=path.parent)
    os.close(fd)
    try:
        if json_format:
            Path(temporary).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        else:
            # 显式关闭句柄，Windows 上失败时也不会阻塞临时文件清理。
            with open(temporary, "wb") as handle:
                torch.save(value, handle)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class DataBundle:
    """同一 study 只加载一次 train/val；正式训练额外加载 test/out 并验证隔离。"""

    def __init__(self, config, *, evaluation=True):
        config.validate()
        root = Path(config.cache_dir)
        metadata_path = root / f"{config.data_exp}metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        experiment = self.metadata.get("experiment")
        strategies = {"exp1": "random_snapshot_within_group", "exp2": "leave_one_galaxy_out"}
        if experiment not in strategies or self.metadata.get("split_strategy") != strategies[experiment]:
            raise ValueError("当前入口仅支持标准 Exp1/Exp2 cache 契约。")
        if self.metadata.get("training_sampling", "uniform") != "uniform":
            raise ValueError("Exp1/2 首版沿用 cache 内过采样，只支持 uniform，不叠加运行/时间重采样。")
        splits = ["train", "val"] + (["test"] + (["out"] if experiment == "exp2" else []) if evaluation else [])
        self.datasets = {}
        identities = {}
        for split in splits:
            scalar = root / f"{config.data_exp}{split}.pt"
            cells = Path(config.cell_cache_dir or config.cache_dir) / f"{config.cell_exp or config.data_exp}{split}_cells.pt"
            ds = CellSFRDataset(scalar, cells, split=split)
            self.datasets[split] = ds
            identities[split] = {"scalar_sha256": ds.source_cache_sha256, "cell_sha256": ds.cell_cache_sha256}
            actual = set(ds.base.group_names)
            expected = set(self.metadata.get("held_out_groups" if split == "out" else "train_groups", []))
            if not expected or actual != expected:
                raise ValueError(f"{split} 实际运行 {actual} 与 metadata {expected} 不一致。")
            if any("torus" in group.lower() for group in actual):
                raise ValueError("当前 Exp1/2 入口不处理 torus。")
            if split != "train" and len(set(ds.sample_uid)) != len(ds):
                raise ValueError(f"{split} 不应含重复样本。")
            count = self.metadata.get("sample_counts", {}).get(split)
            if count is not None and count != len(ds):
                raise ValueError(f"{split} 样本数与 metadata 不符。")
        held = set(self.metadata.get("held_out_groups", []))
        training_groups = set(self.metadata["train_groups"])
        if held & training_groups or (experiment == "exp2" and len(held) != 1) or (experiment == "exp1" and held):
            raise ValueError("训练/留出运行声明不合法。")
        validate_cell_splits(**self.datasets)
        train = self.datasets["train"]
        if min(train.spatial_shape) < 4:
            raise ValueError("SFRUNet 要求 H/W 至少为 4。")
        norm_path = root / f"{config.data_exp}norm.npz"
        with np.load(norm_path, allow_pickle=False) as norms:
            mean, std = norms["mean"], norms["std"]
            if (mean.size != len(train.input_channels) or mean.shape != std.shape
                    or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any()):
                raise ValueError("输入归一化 mean/std 不合法。")
            if "input_channels" in norms and norms["input_channels"].tolist() != list(train.input_channels):
                raise ValueError("norm 通道与 cache 不一致。")
            self.normalization = {"mean": mean.tolist(), "std": std.tolist(),
                                  "log_transform_mask": train.base.log_transform_mask,
                                  "geometry_inputs": train.base.geometry_input_contract,
                                  "source": "original_training_cache"}
        for ds in self.datasets.values():
            if (ds.base.log_transform_mask != train.base.log_transform_mask
                    or ds.base.geometry_input_contract != train.base.geometry_input_contract):
                raise ValueError("split 间输入变换契约不一致。")
        self.scales = train.fit_scales()
        # HPO 与正式训练共享 train/val 契约；test/out 身份另行保存，不进入参数选择。
        self.contract = {"schema": "macnet_sfr_unet_data_v1", "metadata_sha256": sha256_file(metadata_path),
                         "norm_sha256": sha256_file(norm_path), "experiment": experiment,
                         "train_val": {s: identities[s] for s in ("train", "val")},
                         "input_channels": list(train.input_channels), "spatial_shape": list(train.spatial_shape),
                         "normalization": self.normalization, "scales": self.scales}
        # 文本读取统一换行，Windows CRLF 与 Linux LF 不应造成伪代码版本差异。
        self.contract["implementation_sha256"] = {name: hashlib.sha256(
            (Path(__file__).parent / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in
            ("SFRUNet.py", "cell_sfr_objective.py", "cell_sfr_dataset.py", "train_sfr_unet.py")}
        self.evaluation_identities = {s: v for s, v in identities.items() if s not in {"train", "val"}}


def batch_indices(n, size, *, shuffle=False):
    """整批索引，避免逐样本 Python collate；尾批为一帧时合入前一批，兼容瓶颈 BN。"""
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    chunks = list(indices.split(size))
    if shuffle and len(chunks) > 1 and len(chunks[-1]) == 1:
        tail = chunks.pop()
        chunks[-1] = torch.cat([chunks[-1], tail])
    return chunks


def move_batch(dataset, indices, device):
    return tuple(t[indices].to(device) for t in (dataset.base.x, dataset.base.coord, dataset.baseline, dataset.target))


def _statistics(prediction, target, scales):
    """每帧统计；总量误差是先对格子求和，再计算误差，不能平均格子相对误差。"""
    e = prediction.double() - target.double()
    total = e.sum((1, 2, 3))
    mse = e.square().mean((1, 2, 3))
    return torch.stack([mse / scales["cell_scale"]**2, total.square() / scales["total_scale"]**2,
                        torch.relu(-prediction).square().mean((1, 2, 3)) / scales["cell_scale"]**2,
                        e.abs().mean((1, 2, 3)), mse, total.abs(), total.square(), total,
                        (prediction < 0).double().mean((1, 2, 3))], dim=1)


def summarize_statistics(rows):
    values = rows.mean(0).tolist()
    return dict(n=len(rows), cell_loss=values[0], total_loss=values[1], negative_penalty=values[2],
                joint_score=sum(values[:3]), cell_mae=values[3], cell_rmse=math.sqrt(values[4]),
                total_mae=values[5], total_rmse=math.sqrt(values[6]), total_bias=values[7], negative_fraction=values[8])


@torch.no_grad()
def evaluate(model, dataset, objective, scales, *, device, batch_size, amp="none", save_predictions=False):
    """固定系数联合指标，先各运行内部平均再等权平均；不随训练 loss 权重变化。"""
    model.eval()
    stats, baseline_stats, maps = [], [], []
    for indices in batch_indices(len(dataset), batch_size):
        x, coord, b, y = move_batch(dataset, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp == "bfloat16"):
            z = model(x, coord[:, :1], coord[:, 1:2])
        _, detail = objective(z, b, y)
        p = detail["sfr_map_physical"]
        stats.append(_statistics(p, y, scales).cpu())
        baseline_stats.append(_statistics(b, y, scales).cpu())
        if save_predictions:
            maps.append(p.cpu())
    stats, baseline_stats = torch.cat(stats), torch.cat(baseline_stats)
    groups = dataset.base.groups.flatten()
    def aggregate(rows):
        per_group = {name: summarize_statistics(rows[groups == i]) for i, name in enumerate(dataset.base.group_names)}
        return {"all_samples": summarize_statistics(rows), "per_run": per_group,
                "macro_joint_score": sum(v["joint_score"] for v in per_group.values()) / len(per_group)}
    metrics = {"model": aggregate(stats), "baseline": aggregate(baseline_stats), "units": "MACER_mass_code/Gyr"}
    predictions = None
    if save_predictions:
        p = torch.cat(maps)
        predictions = {"schema": "macnet_sfr_unet_predictions_v1", "sample_uid": dataset.sample_uid,
                       "parent_hr_uid": dataset.parent_hr_uid, "time": dataset.base.time,
                       "groups": [dataset.base.group_names[i] for i in groups.tolist()],
                       "prediction_map": p, "target_map": dataset.target, "baseline_map": dataset.baseline,
                       "closure_map": p - dataset.baseline, "prediction_total": p.sum((1, 2, 3)),
                       "target_total": dataset.target.sum((1, 2, 3)), "baseline_total": dataset.baseline.sum((1, 2, 3)),
                       "units": "MACER_mass_code/Gyr", "to_msun_per_yr": .025}
    return metrics, predictions


def invariant_config(config):
    """允许换输出位置/执行设备与暂停长度，其余恢复条件必须相同。"""
    excluded = {"cache_dir", "cell_cache_dir", "output_root", "run_name", "device", "cpu_threads",
                "tensorboard", "resume", "stop_after_epochs"}
    return json.loads(json.dumps({k: v for k, v in asdict(config).items() if k not in excluded}))


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config, *, bundle=None, hpo=False, epoch_callback=None):
    """正式训练和 HPO 共用；callback 只收到 epoch 与验证指标，不接触测试集。"""
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if config.amp != "none" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 AMP 需要支持该格式的 CUDA 设备。")
    bundle = bundle or DataBundle(config, evaluation=not hpo)
    if not hpo and "test" not in bundle.datasets:
        raise ValueError("正式训练必须验证并评估 test，Exp2 还需要 out。")
    ds = bundle.datasets["train"]
    if len(ds) < 2:
        raise ValueError("训练至少需要两个 snapshot。")
    config_model = {k: getattr(config, k) for k in ("base_ch", "encoder_depths", "decoder_depths", "bottleneck_depth",
                    "pos_num_bands", "pos_max_freq", "p_drop", "se_reduction")}
    model = SFRUNet(len(ds.input_channels), **config_model).to(device)
    objective = CellSFRLoss.from_scale_contract(bundle.scales, total_weight=config.total_weight,
                                               negative_weight=config.negative_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = math.ceil(config.epochs * config.warmup_ratio)
    def schedule(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(config.epochs - warmup - 1, 1)
        return config.min_lr_ratio + (1-config.min_lr_ratio) * (1+math.cos(math.pi*min(progress, 1))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best, best_epoch, stale, history = 0, math.inf, 0, 0, []
    if config.resume:
        path = Path(config.resume)
        if path.name != "last.pt":
            raise ValueError("断点恢复需指定本入口的 last.pt。")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("schema") != "macnet_sfr_unet_checkpoint_v1" or checkpoint["hpo"] != hpo
                or checkpoint["invariant_config"] != invariant_config(config)
                or fingerprint(checkpoint["data_contract"]) != fingerprint(bundle.contract)
                or checkpoint["evaluation_identities"] != bundle.evaluation_identities):
            raise ValueError("恢复配置或数据契约与 checkpoint 不一致。")
        output = path.resolve().parent
        if not (output / "best.pt").is_file():
            raise ValueError("断点目录缺失 best.pt，不能恢复最优模型选择。")
        model.load_state_dict(checkpoint["model"]); objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, best_epoch, stale, history = (checkpoint[k] for k in ("epoch", "best_score", "best_epoch", "stale", "history"))
        _restore_rng(checkpoint["rng"])
    else:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        name = config.run_name or f"{config.data_exp}SFRUNet_seed{config.seed}"
        output = Path(config.output_root) / f"{name}_{stamp}_{uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=False)
    manifest = {split: {"sample_uid": data.sample_uid, "parent_hr_uid": data.parent_hr_uid}
                for split, data in bundle.datasets.items()}
    atomic_save(output / "config.json", asdict(config), json_format=True)
    atomic_save(output / "contract.json", {"data": bundle.contract, "model": model.model_contract(),
                "loss": objective.loss_contract(), "selection": "run_macro(cell+total+negative), fixed coefficients=1",
                "evaluation_identities": bundle.evaluation_identities, "cache_metadata": bundle.metadata}, json_format=True)
    atomic_save(output / "split_manifest.json", manifest, json_format=True)
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=start+1 if start else None)
    print(f"SFRUNet: objective=signed_closure; loss_space=physical_sfr; conditioning=none; device={device}")
    print(f"Input channels: {ds.input_channels}; shape={ds.spatial_shape}; scales={bundle.scales['closure_scale']:.6g},"
          f"{bundle.scales['cell_scale']:.6g},{bundle.scales['total_scale']:.6g}; CPU threads={torch.get_num_threads()}")
    print(f"Output: {output}", flush=True)
    end = min(config.epochs, start + config.stop_after_epochs) if config.stop_after_epochs else config.epochs
    stopped = config.patience > 0 and stale >= config.patience
    try:
        for epoch in range(start, end):
            if stopped:
                break
            begin = time.perf_counter()
            model.train()
            sums = torch.zeros(4, device=device, dtype=torch.float64)
            lr = optimizer.param_groups[0]["lr"]
            for indices in batch_indices(len(ds), config.batch_size, shuffle=True):
                x, coord, b, y = move_batch(ds, indices, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp == "bfloat16"):
                    z = model(x, coord[:, :1], coord[:, 1:2])
                loss, detail = objective(z, b, y)
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                except RuntimeError as exc:
                    if "non-finite" in str(exc):
                        raise FloatingPointError("梯度非有限，停止当前训练。") from exc
                    raise
                optimizer.step()
                sums += torch.stack([loss.detach(), detail["cell_loss"].detach(), detail["total_sfr_loss"].detach(),
                                     detail["negative_penalty"].detach()]) * len(indices)
            validation, _ = evaluate(model, bundle.datasets["val"], objective, bundle.scales, device=device,
                                     batch_size=config.batch_size, amp=config.amp)
            score = validation["model"]["macro_joint_score"]
            if not math.isfinite(score):
                raise FloatingPointError("验证指标非有限。")
            improved = score < best
            if improved:
                best, best_epoch, stale = score, epoch+1, 0
            else:
                stale += 1
            scheduler.step()
            row = {"epoch": epoch+1, "train_loss": (sums / len(ds)).cpu().tolist(), "val": validation,
                   "score": score, "lr": lr, "seconds": time.perf_counter()-begin}
            history.append(row)
            checkpoint = {"schema": "macnet_sfr_unet_checkpoint_v1", "hpo": hpo,
                          "config": asdict(config), "invariant_config": invariant_config(config),
                          "data_contract": bundle.contract, "evaluation_identities": bundle.evaluation_identities,
                          "model_contract": model.model_contract(), "loss_contract": objective.loss_contract(),
                          "model": model.state_dict(), "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": _rng_state(), "epoch": epoch+1,
                          "best_score": best, "best_epoch": best_epoch, "stale": stale, "history": history}
            if improved:
                atomic_save(output / "best.pt", checkpoint)
            atomic_save(output / "last.pt", checkpoint)
            atomic_save(output / "history.json", history, json_format=True)
            print(f"Epoch {epoch+1:03d} train={row['train_loss'][0]:.6g} val_macro={score:.6g} "
                  f"lr={lr:.3g} time={row['seconds']:.2f}s", flush=True)
            if writer:
                for key, value in zip(("total", "cell", "global", "negative"), row["train_loss"]):
                    writer.add_scalar(f"train/{key}", value, epoch+1)
                writer.add_scalar("val/macro_joint", score, epoch+1)
                for group, metrics in validation["model"]["per_run"].items():
                    for key in ("cell_rmse", "total_rmse", "total_mae", "negative_fraction"):
                        writer.add_scalar(f"val/{group}/{key}", metrics[key], epoch+1)
                writer.add_scalar("performance/epoch_seconds", row["seconds"], epoch+1)
            if epoch_callback:
                epoch_callback(epoch+1, score)
            stopped = config.patience > 0 and stale >= config.patience
        completed = stopped or len(history) >= config.epochs
        report = {"output_dir": str(output), "status": "complete" if completed else "paused_at_epoch_boundary",
                  "best_score": best, "best_epoch": best_epoch, "epochs_completed": len(history), "hpo": hpo}
        if completed and not hpo:
            model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=False)["model"])
            report["evaluation"] = {}
            for split in ("test", "out"):
                if split in bundle.datasets:
                    metrics, predictions = evaluate(model, bundle.datasets[split], objective, bundle.scales,
                        device=device, batch_size=config.batch_size, amp=config.amp, save_predictions=True)
                    report["evaluation"][split] = metrics
                    atomic_save(output / f"{split}_predictions.pt", predictions)
        atomic_save(output / "summary.json", report, json_format=True)
        return report
    finally:
        if writer:
            writer.close()


def add_training_arguments(parser):
    """三个入口共用 CLI 类型与默认值，配置 JSON 使用下划线字段名。"""
    parser.add_argument("--config", help="独立 TrainConfig JSON，不读取旧 ExperimentSetting.json")
    defaults = TrainConfig()
    for field in fields(TrainConfig):
        name, value = field.name, getattr(defaults, field.name)
        if name in {"encoder_depths", "decoder_depths"}:
            parser.add_argument("--"+name.replace("_", "-"), type=int, nargs=2, default=value)
        elif isinstance(value, bool):
            parser.add_argument("--"+name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument("--"+name.replace("_", "-"), type=type(value), default=value)
    return parser


def parse_with_config(parser, argv=None):
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        values = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        unknown = set(values) - {f.name for f in fields(TrainConfig)}
        if unknown:
            parser.error(f"配置含未知字段：{sorted(unknown)}")
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    config.validate()
    return args, config


def main(argv=None):
    _, config = parse_with_config(add_training_arguments(argparse.ArgumentParser(description=__doc__)), argv)
    train(config)


if __name__ == "__main__":
    main()
