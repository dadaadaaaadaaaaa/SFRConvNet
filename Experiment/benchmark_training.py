"""使用已有 CachePT 对固定小训练交替计时，记录速度与数值一致性（不运行 HPO）。"""

import argparse
import copy
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from Experiment import train
from Experiment.models import SFRConvNet


class ScalarRecorder:
    """记录验证输出；测稳态循环时排除文件写入噪声。"""
    def __init__(self):
        self.values = {}

    def add_scalar(self, key, value, *args):
        self.values[key] = float(value)


def load_original(path):
    spec = importlib.util.spec_from_file_location("train_original_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def subset(dataset, count):
    """取已打乱 cache 的固定前缀，并覆盖尾批；所有逐样本字段保持对齐。"""
    result = copy.copy(dataset)
    for key, value in vars(dataset).items():
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim and len(value) == len(dataset):
            setattr(result, key, value[:count])
    return result


def make_loader(dataset, batch_size, mode, workers, seed):
    # 两条数据路径消费完全相同的显式 batch 索引，排除随机顺序的干扰。
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if mode in {"batched", "device"}:
        from Experiment.training_performance import TensorBatchLoader
        return TensorBatchLoader(dataset, batch_size, shuffle=False,
                                 device="cuda" if mode == "device" else "cpu",
                                 batches=batches)
    return DataLoader(dataset, batch_sampler=batches, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


def run_case(name, args, original, datasets):
    train.set_seed(31415)
    threads = int(name.split(":")[1]) if ":" in name else args.threads
    torch.set_num_threads(threads)
    base_name = name.split(":")[0]
    mode = base_name if base_name in {"batched", "device"} else args.loader
    if base_name == "baseline":
        mode = "standard"
    train_fn = original.train_one_epoch if base_name == "baseline" else train.train_one_epoch
    eval_fn = original.eval_epoch if base_name in {"baseline", "accum"} else train.eval_epoch
    start = time.perf_counter()
    train_loader = make_loader(datasets[0], args.batch_size, mode, args.workers, 711)
    val_loader = make_loader(datasets[1], args.batch_size, mode, args.workers, 712)
    model = SFRConvNet(c_in=12, base_ch=32, stage_depths=(2, 2, 2, 2),
                      pos_num_bands=32, pos_max_freq=101, p_drop=0.0,
                      prediction_mode="signed_closure").cuda()
    if base_name == "pe":
        from Experiment.training_performance import configure_position_cache
        configure_position_cache(model, datasets, augmentor=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=True)
    criterion = train.make_criterion("l1", 1.0, weight_mode="none")
    writer = ScalarRecorder()
    options = dict(objective_mode="signed_closure", closure_transform="signed_asinh",
                   closure_scale=datasets[0].closure_scale, negative_sfr_weight=1.0)
    sync()
    setup = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    records = []
    for epoch in range(args.epochs):
        sync()
        t0 = time.perf_counter()
        loss = train_fn(model, train_loader, criterion, optimizer, torch.device("cuda"),
                        scaler=scaler, grad_clip=0.3, **options)
        sync()
        t1 = time.perf_counter()
        metric, groups = eval_fn(model, val_loader, criterion, torch.device("cuda"),
                                 writer, epoch, **options)
        sync()
        t2 = time.perf_counter()
        records.append(dict(train_s=t1-t0, val_s=t2-t1, loss=loss, metric=metric, groups=groups))
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    result = dict(case=name, threads=threads, loader=mode, setup_s=setup, epochs=records,
                  peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20)
    del model, optimizer, scaler, train_loader, val_loader
    return result, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="Exp1_Rh8x8_Closure_auto_")
    parser.add_argument("--original", default="Results/performance_audit_20260908/train_before.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", default=["baseline", "accum"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=1539)
    parser.add_argument("--val-samples", type=int, default=769)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--loader", default="standard")
    args = parser.parse_args()
    original = load_original(args.original)
    torch.set_num_threads(args.threads)
    datasets = [subset(train.GalPTDataset(ROOT / ".cache" / f"{args.prefix}{split}.pt",
                                        objective_mode="signed_closure"), count)
                for split, count in (("train", args.train_samples), ("val", args.val_samples))]
    output = dict(environment=dict(platform=platform.platform(), torch=torch.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                   cpu_count=os.cpu_count()), config=vars(args), runs=[])
    reference_state = None
    for repeat in range(args.repeats):
        for name in (args.cases if repeat % 2 == 0 else list(reversed(args.cases))):
            result, state = run_case(name, args, original, datasets)
            if reference_state is None:
                reference_state = state
            result["state_max_abs_vs_first"] = max(
                (value.to(torch.float64) - reference_state[key].to(torch.float64)).abs().max().item()
                for key, value in state.items())
            result["repeat"] = repeat
            output["runs"].append(result)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
            timed = result["epochs"][1:]
            print(name, "train", statistics.median(x["train_s"] for x in timed),
                  "val", statistics.median(x["val_s"] for x in timed),
                  "state_error", result["state_max_abs_vs_first"], flush=True)


if __name__ == "__main__":
    main()
