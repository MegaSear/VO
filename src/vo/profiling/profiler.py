# =========================================================
# ======================= Profiler ========================
# =========================================================
import gc
import os
import time
import threading

import numpy as np
import torch
import psutil
from tqdm.auto import tqdm


class EstimatorProfiler():
    @staticmethod
    def run_time_benchmark(estimator, img1, img2, K, warmup, runs):
        use_cuda = torch.cuda.is_available()
        # Warmup-прогон
        for _ in range(warmup):
            estimator.estimate(img1, img2, K)
            if use_cuda:
                torch.cuda.synchronize()

        measurement_samples = []
        for _ in range(runs):
            start = time.perf_counter()
            estimator.estimate(img1, img2, K)
            if use_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            measurement_samples.append((end - start) * 1000.0)

        measurement_samples = np.array(measurement_samples)
        print("Benchmark finished!")
        return {
            "mean_time_ms": float(np.mean(measurement_samples)),
            "q95_time_ms": float(np.quantile(measurement_samples, 0.95)),
            "median_time_ms": float(np.median(measurement_samples)),
        }

    @staticmethod
    def _sample_peak_rss_mb(func, args, interval=0.005):
        """Поточный сэмплер RSS: ловит пиковое потребление ВО ВРЕМЯ вызова
        (в отличие от before/after-дельты, которая видит только то, что
        осталось резидентным после вызова — а короткоживущие аллокации
        внутри SIFT/BFMatcher к этому моменту уже могли быть освобождены).
        Baseline вычитается, чтобы не учитывать память, уже занятую другими
        объектами в процессе (например, весами соседнего эстиматора)."""
        process = psutil.Process(os.getpid())
        baseline_mb = process.memory_info().rss / 1024 ** 2
        samples = []
        stop_event = threading.Event()

        def monitor():
            while not stop_event.is_set():
                samples.append(process.memory_info().rss / 1024 ** 2 - baseline_mb)
                stop_event.wait(interval)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            result = func(*args)
        finally:
            stop_event.set()
            thread.join()

        peak_mb = max(0.0, max(samples, default=0.0))
        return peak_mb, result

    @classmethod
    def run_memory_usage_benchmark(cls, estimator, img1, img2, K, runs):
        print("Run memory usage benchmark...")
        use_cuda = torch.cuda.is_available()

        cpu_samples, gpu_samples = [], []
        for _ in range(runs):
            gc.collect()

            if use_cuda:
                torch.cuda.synchronize()
                gpu_before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()

            cpu_peak_mb, _ = cls._sample_peak_rss_mb(
                estimator.estimate, (img1, img2, K)
            )
            cpu_samples.append(cpu_peak_mb)

            if use_cuda:
                torch.cuda.synchronize()
                gpu_peak = torch.cuda.max_memory_allocated()
                # Дельта относительно baseline ДО reset — иначе резидентные
                # веса другого эстиматора (например, SuperPoint+LightGlue,
                # уже созданного в этом же процессе) попадут в цифру для
                # чисто-CPU методов вроде SIFT.
                gpu_samples.append(max(0.0, (gpu_peak - gpu_before) / 1024 ** 2))
            else:
                gpu_samples.append(0.0)

        cpu_samples, gpu_samples = np.array(cpu_samples), np.array(gpu_samples)
        print("Benchmark finished!")
        return {
            "mean_cpu_mem_mb": float(np.mean(cpu_samples)),
            "median_cpu_mem_mb": float(np.median(cpu_samples)),
            "q95_cpu_mem_mb": float(np.quantile(cpu_samples, 0.95)),
            "mean_gpu_mem_mb": float(np.mean(gpu_samples)),
            "median_gpu_mem_mb": float(np.median(gpu_samples)),
            "q95_gpu_mem_mb": float(np.quantile(gpu_samples, 0.95)),
        }

    @classmethod
    def profile_on_sample(cls, estimator, dataset, K, pairs_indices, warmup, runs):
        print("Profile on sample...")
        stats = []
        for idx1, idx2 in tqdm(pairs_indices, desc="Profiling"):
            (img1, _, _), (img2, _, _), K = dataset[idx1], dataset[idx2], K
            runtime_stat = cls.run_time_benchmark(estimator, img1, img2, K, warmup, runs)
            mem_usage_stat = cls.run_memory_usage_benchmark(estimator, img1, img2, K, runs)
            stats.append({"idx1": idx1, "idx2": idx2, **runtime_stat, **mem_usage_stat})
        print("Profiling finished!")
        return stats