from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import threading
import time
import tracemalloc
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeRecord:
    wall_seconds: float
    process_cpu_seconds: float
    python_peak_megabytes: float
    rss_baseline_megabytes: float | None
    rss_peak_megabytes: float | None
    rss_delta_peak_megabytes: float | None
    cuda_peak_megabytes: float | None
    platform: str
    python: str
    processor: str
    pid: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.wall_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "python_peak_megabytes": self.python_peak_megabytes,
            "rss_baseline_megabytes": self.rss_baseline_megabytes,
            "rss_peak_megabytes": self.rss_peak_megabytes,
            "rss_delta_peak_megabytes": self.rss_delta_peak_megabytes,
            "cuda_peak_megabytes": self.cuda_peak_megabytes,
            "platform": self.platform,
            "python": self.python,
            "processor": self.processor,
            "pid": self.pid,
        }


def measure_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, RuntimeRecord]:
    cuda_available = False
    torch_module = None
    try:
        import torch  # type: ignore

        torch_module = torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        cuda_available = False
    rss_samples: list[int] = []
    stop_sampling = threading.Event()
    sampler: threading.Thread | None = None
    rss_baseline: int | None = None
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        rss_baseline = int(process.memory_info().rss)
        rss_samples.append(rss_baseline)

        def sample_rss() -> None:
            while not stop_sampling.wait(0.01):
                try:
                    rss_samples.append(int(process.memory_info().rss))
                except (psutil.Error, OSError):
                    return

        sampler = threading.Thread(target=sample_rss, name="nsi-rss-sampler", daemon=True)
        sampler.start()
    except (ImportError, OSError):
        rss_baseline = None
    tracemalloc.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
        if cuda_available and torch_module is not None:
            torch_module.cuda.synchronize()
    finally:
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stop_sampling.set()
        if sampler is not None:
            sampler.join(timeout=1.0)
            try:
                rss_samples.append(int(process.memory_info().rss))
            except (psutil.Error, OSError):
                pass
    cuda_peak = None
    if cuda_available and torch_module is not None:
        cuda_peak = float(torch_module.cuda.max_memory_allocated() / (1024**2))
    rss_peak = max(rss_samples) if rss_samples else None
    record = RuntimeRecord(
        wall_seconds=float(wall_seconds),
        process_cpu_seconds=float(cpu_seconds),
        python_peak_megabytes=float(peak_bytes / (1024**2)),
        rss_baseline_megabytes=(
            float(rss_baseline / (1024**2)) if rss_baseline is not None else None
        ),
        rss_peak_megabytes=(float(rss_peak / (1024**2)) if rss_peak is not None else None),
        rss_delta_peak_megabytes=(
            float(max(rss_peak - rss_baseline, 0) / (1024**2))
            if rss_peak is not None and rss_baseline is not None
            else None
        ),
        cuda_peak_megabytes=cuda_peak,
        platform=platform.platform(),
        python=platform.python_version(),
        processor=platform.processor(),
        pid=os.getpid(),
    )
    return result, record
