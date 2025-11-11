import os
import time
import json
import torch
import subprocess
from glob import glob


def start_run_log(task="gsm8k", tag=None, log_dir="results"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{task}_{tag or 'run'}_{timestamp}.json"
    path = os.path.join(log_dir, filename)
    print(f"[LOG] Starting run: {filename}")
    return {
        "t0": time.time(),
        "path": path,
        "task": task,
        "tag": tag,
        "timestamp": timestamp,
    }


def get_gpu_info():
    """Return a dict with GPU specs (name, memory, driver, CUDA)."""
    info = {"gpu_available": torch.cuda.is_available()}
    if not info["gpu_available"]:
        return info

    try:
        device_idx = 0
        props = torch.cuda.get_device_properties(device_idx)
        info.update({
            "gpu_name": props.name,
            "gpu_total_vram_gb": round(props.total_memory / (1024**3), 2),
            "cuda_device": device_idx,
            "cuda_version": torch.version.cuda,
            "driver_version": None,
        })

        # get driver version via nvidia-smi (fallback to torch if missing)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode().strip().split("\n")[device_idx]
            info["driver_version"] = out
        except Exception:
            pass

        return info
    except Exception as e:
        info["error"] = str(e)
        return info


def extract_accuracy_from_results(results_dir):
    """Find the newest results_*.json file under the model directory and return accuracy."""
    try:
        pattern = os.path.join(results_dir, "*/results_*.json")
        result_files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        if not result_files:
            print(f"[WARN] No result files found under {results_dir}")
            return None
        latest_file = result_files[0]
        with open(latest_file, "r") as f:
            data = json.load(f)

        gsm8k_metrics = data.get("results", {}).get("gsm8k", {})
        for key, value in gsm8k_metrics.items():
            if "flexible-extract" in key and key.startswith("exact_match"):
                print(
                    f"[LOG] Parsed accuracy from {os.path.basename(latest_file)}: {value}"
                )
                return value

        print(f"[WARN] flexible-extract accuracy not found in {latest_file}")
        return None
    except Exception as e:
        print(f"[WARN] Could not extract accuracy: {e}")
        return None


def end_run_log(run_info, results_dir=None, **metrics):
    """End timer, compute throughput, and optionally merge accuracy from results_dir."""
    t1 = time.time()
    duration = t1 - run_info["t0"]

    # extract token-related metrics safely
    tokens_generated = metrics.get("tokens_generated")
    total_time_s = metrics.get("total_time_s", duration)
    tokens_per_s = metrics.get("tokens_per_s")

    # if not provided, compute throughput fallback
    if tokens_generated is not None and (tokens_per_s is None or tokens_per_s == 0):
        tokens_per_s = tokens_generated / total_time_s if total_time_s > 0 else None

    # extract accuracy from results JSON
    accuracy = None
    if results_dir and os.path.exists(results_dir):
        accuracy = extract_accuracy_from_results(results_dir)

    data = {
        "task": run_info["task"],
        "tag": run_info["tag"],
        "timestamp": run_info["timestamp"],
        "total_time_s": total_time_s,
        "tokens_generated": tokens_generated,
        "tokens_per_s": tokens_per_s,
        "accuracy_flexible": accuracy,
    }

    # include any other custom fields that might exist in metrics
    extra = {k: v for k, v in metrics.items() if k not in data}
    data.update(extra)

    # merge runtime metrics from the latest runtime_metrics_*.json file
    runtime_metrics_files = sorted(glob(os.path.join(os.path.dirname(run_info["path"]), "runtime_metrics_*.json")))
    if runtime_metrics_files:
        runtime_metrics_path = runtime_metrics_files[-1]
        with open(runtime_metrics_path) as f:
            runtime_metrics = json.load(f)
        data.update(runtime_metrics)

    # add GPU info
    data["gpu_info"] = get_gpu_info()

    # save JSON
    with open(run_info["path"], "w") as f:
        json.dump(data, f, indent=2)

    print(f"[LOG] Saved run summary to {run_info['path']}")
    return data
