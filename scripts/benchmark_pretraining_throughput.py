"""Bounded CUDA throughput benchmark; never writes official experiment evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mepi_v1.config import load_pretraining_config, resolve_path
from src.mepi_v1.constants import CANDIDATE_BACKBONES
from src.mepi_v1.data import MagNetDataset
from src.mepi_v1.models import PretrainingModel
from src.mepi_v1.pretrain import _fit_preprocessing, _read_indices, _seed_everything


class GPUUtilizationSampler:
    def __init__(self) -> None:
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self._stop.wait(0.25):
                try:
                    value = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                            "--id=0",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    ).strip().splitlines()[0]
                    self.values.append(float(value))
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return float(np.mean(self.values)) if self.values else None


def prepare_dataset(cache_mode: str) -> tuple[MagNetDataset, int, dict[str, Any]]:
    config = load_pretraining_config(ROOT / "configs/pretrain_tcn.yaml")
    h5_path = resolve_path(config, config["dataset"]["path"])
    train_indices = _read_indices(
        resolve_path(config, config["dataset"]["split_dir"]) / "magnet_train.csv"
    )
    preprocessing, target = _fit_preprocessing(h5_path, train_indices)
    with h5py.File(h5_path, "r") as handle:
        material_names = sorted(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in set(handle["material"][train_indices])
        )
    dataset = MagNetDataset(
        h5_path,
        train_indices,
        material_to_index={name: index for index, name in enumerate(material_names)},
        waveform_mean=preprocessing["waveform"]["mean"],
        waveform_scale=preprocessing["waveform"]["scale"],
        operating_mean=preprocessing["operating"]["mean"],
        operating_scale=preprocessing["operating"]["scale"],
        target_transform=target,
        cache_mode=cache_mode,
    )
    return dataset, len(material_names), config


def loader_for(
    dataset: MagNetDataset, *, batch_size: int, workers: int, persistent_workers: bool
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "generator": torch.Generator().manual_seed(42),
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers:
        kwargs.update(persistent_workers=persistent_workers, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def synchronize() -> None:
    torch.cuda.synchronize()


def benchmark(
    dataset: MagNetDataset,
    material_count: int,
    config: dict[str, Any],
    *,
    backbone: str,
    batch_size: int,
    workers: int,
    steps: int,
    warmup: int,
    persistent_workers: bool = True,
) -> dict[str, Any]:
    device = torch.device("cuda")
    _seed_everything(42, config.get("runtime"))
    loader = loader_for(
        dataset,
        batch_size=batch_size,
        workers=workers,
        persistent_workers=persistent_workers,
    )
    iterator = iter(loader)
    model = PretrainingModel(
        backbone,
        material_count=material_count,
        latent_dim=int(config["model"]["latent_dim"]),
        backbone_layers=int(config["model"]["backbone_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    timings = {name: [] for name in ("data", "transfer", "forward", "backward", "optimizer")}
    torch.cuda.reset_peak_memory_stats()
    sampler = GPUUtilizationSampler()
    measured_samples = 0
    result: dict[str, Any]
    try:
        model.train()
        for step in range(warmup + steps):
            started = time.perf_counter()
            batch = next(iterator)
            data_done = time.perf_counter()
            waveform = batch["waveform"].to(device, non_blocking=True)
            operating = batch["tabular"].to(device, non_blocking=True)
            material_index = batch["material_index"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            synchronize()
            transfer_done = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                output = model(waveform, operating, material_index)
                loss = F.mse_loss(output, target) + 0.3 * F.l1_loss(output, target)
            synchronize()
            forward_done = time.perf_counter()
            scaler.scale(loss).backward()
            synchronize()
            backward_done = time.perf_counter()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            synchronize()
            optimizer_done = time.perf_counter()
            if step == warmup:
                sampler.start()
            if step >= warmup:
                timings["data"].append(data_done - started)
                timings["transfer"].append(transfer_done - data_done)
                timings["forward"].append(forward_done - transfer_done)
                timings["backward"].append(backward_done - forward_done)
                timings["optimizer"].append(optimizer_done - backward_done)
                measured_samples += int(output.shape[0])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
        total = sum(sum(values) for values in timings.values())
        measured_steps = len(timings["data"])
        result = {
            "status": "PASS",
            "backbone": backbone,
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": persistent_workers if workers else False,
            "prefetch_factor": 2 if workers else None,
            "cache_mode": dataset.cache_mode,
            "steps": measured_steps,
            "steps_per_second": measured_steps / total,
            "samples_per_second": measured_samples / total,
            "mean_data_loading_ms": 1000 * float(np.mean(timings["data"])),
            "mean_host_to_device_ms": 1000 * float(np.mean(timings["transfer"])),
            "mean_forward_ms": 1000 * float(np.mean(timings["forward"])),
            "mean_backward_ms": 1000 * float(np.mean(timings["backward"])),
            "mean_optimizer_step_ms": 1000 * float(np.mean(timings["optimizer"])),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            "finite_loss": True,
            "material_indices_in_last_batch": int(torch.unique(material_index).numel()),
        }
    except torch.OutOfMemoryError as error:
        result = {
            "status": "OOM",
            "backbone": backbone,
            "batch_size": batch_size,
            "num_workers": workers,
            "cache_mode": dataset.cache_mode,
            "error": str(error),
        }
    finally:
        result["mean_gpu_utilization_percent"] = sampler.stop()
        del iterator, loader, model, optimizer, scaler
        torch.cuda.empty_cache()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def write_results(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("profile", "batch-sweep", "loader-sweep", "all-backbones", "comprehensive"), required=True
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, nargs="+", default=[4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--cache-mode", choices=("hdf5", "ram"), default="hdf5")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    dataset, material_count, config = prepare_dataset(args.cache_mode)
    rows: list[dict[str, Any]] = []
    if args.suite == "profile":
        rows.append(
            benchmark(
                dataset, material_count, config, backbone="TCN", batch_size=args.batch_size,
                workers=args.workers[0], steps=args.steps, warmup=args.warmup,
            )
        )
    elif args.suite == "batch-sweep":
        for batch_size in args.batch_sizes:
            row = benchmark(
                dataset, material_count, config, backbone="TCN", batch_size=batch_size,
                workers=args.workers[0], steps=args.steps, warmup=args.warmup,
            )
            rows.append(row)
            if row["status"] == "OOM":
                break
    elif args.suite == "loader-sweep":
        for workers in args.workers:
            rows.append(
                benchmark(
                    dataset, material_count, config, backbone="TCN", batch_size=args.batch_size,
                    workers=workers, steps=args.steps, warmup=args.warmup,
                )
            )
    elif args.suite == "all-backbones":
        for backbone in CANDIDATE_BACKBONES:
            row = benchmark(
                dataset, material_count, config, backbone=backbone, batch_size=args.batch_size,
                workers=args.workers[0], steps=args.steps, warmup=args.warmup,
            )
            rows.append(row)
            if row["status"] == "OOM":
                break
    else:
        batch_rows = []
        passing_batch_sizes = []
        for batch_size in args.batch_sizes:
            row = benchmark(
                dataset, material_count, config, backbone="TCN", batch_size=batch_size,
                workers=args.workers[0], steps=args.steps, warmup=args.warmup,
            )
            batch_rows.append(row)
            if row["status"] == "OOM":
                break
            passing_batch_sizes.append(batch_size)
        memory_rows = []
        common_batch_size = None
        for batch_size in passing_batch_sizes:
            batch_passed = True
            for backbone in CANDIDATE_BACKBONES:
                row = benchmark(
                    dataset, material_count, config, backbone=backbone,
                    batch_size=batch_size, workers=args.workers[0], steps=1, warmup=1,
                )
                memory_rows.append(row)
                if row["status"] != "PASS":
                    batch_passed = False
                    break
            if not batch_passed:
                break
            common_batch_size = batch_size
        if common_batch_size is None:
            raise RuntimeError("No candidate batch size passed all eight backbones")
        loader_rows = []
        for workers in (2, 4, 8):
            loader_rows.append(
                benchmark(
                    dataset, material_count, config, backbone="TCN",
                    batch_size=common_batch_size, workers=workers,
                    steps=max(20, args.steps), warmup=args.warmup,
                )
            )
        ram_dataset = MagNetDataset(
            dataset.h5_path,
            dataset.indices,
            material_to_index=dataset.material_to_index,
            waveform_mean=dataset.waveform_mean,
            waveform_scale=dataset.waveform_scale,
            operating_mean=dataset.operating_mean,
            operating_scale=dataset.operating_scale,
            target_transform=dataset.target_transform,
            cache_mode="ram",
        )
        for workers in (2, 4, 8):
            loader_rows.append(
                benchmark(
                    ram_dataset, material_count, config, backbone="TCN",
                    batch_size=common_batch_size, workers=workers,
                    steps=max(20, args.steps), warmup=args.warmup,
                )
            )
        rows = batch_rows + memory_rows + loader_rows
    payload = {
        "benchmark_only_not_official_evidence": True,
        "suite": args.suite,
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
    }
    if args.suite == "comprehensive":
        payload.update(
            {
                "batch_sweep": batch_rows,
                "all_backbone_memory_smoke": memory_rows,
                "loader_sweep": loader_rows,
                "largest_common_passing_batch_size": common_batch_size,
            }
        )
    write_results(args.output, payload)


if __name__ == "__main__":
    main()
