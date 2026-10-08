"""以真实 CachePT 的固定子集测量完整 train.main，分离启动开销和训练循环。

只运行小训练，不创建 Optuna study，不覆盖正式结果；缓存文件保持只读。
"""

import argparse
import contextlib
import json
import time
from pathlib import Path
from unittest import mock

try:
    from Experiment.benchmark_training import ROOT, load_original, subset, sync
except ModuleNotFoundError:
    from benchmark_training import ROOT, load_original, subset, sync
from Experiment import train
from Experiment.training_performance import DatasetCache
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=1539)
    parser.add_argument("--val-samples", type=int, default=769)
    parser.add_argument("--cases", nargs="+", default=["reference", "updated", "no_graph", "reuse", "batched"])
    cli = parser.parse_args()
    torch.set_num_threads(8)
    original = load_original(ROOT / "Results/performance_audit_20260908/train_before.py")
    store = DatasetCache()
    reference_state = None
    results = []
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 先完整加载和校验每个 PT，再取固定前缀；启动的 IO 仍代表实际 cache。
    dataset_type = train.GalPTDataset
    def small_dataset(path, **kwargs):
        return subset(dataset_type(path, **kwargs), cli.train_samples if str(path).endswith("train.pt") else cli.val_samples)

    for repeat in range(cli.repeats):
        for name in (cli.cases if repeat % 2 == 0 else list(reversed(cli.cases))):
            module = original if name == "reference" else train
            args = train.parse_args([])
            args.data_exp = "Exp1_Rh8x8_Closure_auto_"
            args.cache_dir = str(ROOT / ".cache")
            args.model_type = "convnet"
            args.base_ch = 32
            args.p_drop = 0.0
            args.batch_size = 128
            args.epochs = cli.epochs
            args.lr = 5e-4
            args.weight_decay = 1e-5
            args.loss = "l1"
            args.objective_mode = "signed_closure"
            args.no_save = args.skip_test = True
            args.num_workers = 0
            args.log_dir = str(ROOT / "Results/performance_audit_20260908/tensorboard")
            args.exp_name = f"Startup_{name}_{repeat}"
            args.skip_model_graph = name not in {"reference", "updated"}
            args.batch_backend = "tensor_cpu" if name == "batched" else "standard"
            train_fn, val_fn = module.train_one_epoch, module.eval_epoch
            timing, holder = {"train_s": 0.0, "val_s": 0.0}, {}
            def timed_train(model, *args, **kwargs):
                sync()
                start = time.perf_counter()
                result = train_fn(model, *args, **kwargs)
                sync()
                timing["train_s"] += time.perf_counter() - start
                holder["model"] = model
                return result
            def timed_val(*args, **kwargs):
                sync()
                start = time.perf_counter()
                result = val_fn(*args, **kwargs)
                sync()
                timing["val_s"] += time.perf_counter() - start
                return result
            kwargs = {"dataset_cache": store} if name in {"reuse", "batched"} else {}
            with output.with_suffix(f".{repeat}_{name}.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), \
                     mock.patch.object(module, "GalPTDataset", small_dataset), \
                     mock.patch.object(module, "train_one_epoch", timed_train), \
                     mock.patch.object(module, "eval_epoch", timed_val):
                    sync()
                    start = time.perf_counter()
                    metric = module.main(args, **kwargs)
                    sync()
                    elapsed = time.perf_counter() - start
            state = {k: v.detach().cpu().clone() for k, v in holder["model"].state_dict().items()}
            if reference_state is None:
                reference_state = state
            error = max((v.double()-reference_state[k].double()).abs().max().item() for k, v in state.items())
            row = dict(case=name, repeat=repeat, elapsed_s=elapsed,
                       startup_and_logging_s=elapsed-sum(timing.values()), **timing,
                       metric=metric, state_max_abs_vs_first=error)
            results.append(row)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(row, flush=True)
            del holder, state


if __name__ == "__main__":
    main()
