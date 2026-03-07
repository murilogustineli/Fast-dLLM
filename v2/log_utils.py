import os
import sys
import json
import socket
import torch
import subprocess


def get_git_info():
    """Return git commit, branch, and dirty state."""
    info = {}
    try:
        # Get current commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit"] = commit

        # Get current branch
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = branch if branch else "detached"

        # Check if working directory is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = len(status) > 0

    except Exception as e:
        info["error"] = str(e)

    return info


def get_environment_info():
    """Return python/torch versions, hostname, slurm job id."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "hostname": socket.gethostname(),
    }

    # Get SLURM job ID if running in SLURM
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        info["slurm_job_id"] = slurm_job_id

    return info


def get_gpu_info():
    """Return a dict with GPU specs (name, memory, driver, CUDA)."""
    info = {"gpu_available": torch.cuda.is_available()}
    if not info["gpu_available"]:
        return info

    try:
        device_idx = 0
        props = torch.cuda.get_device_properties(device_idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_vram_gb": round(props.total_memory / (1024**3), 2),
                "cuda_device": device_idx,
                "cuda_version": torch.version.cuda,
                "driver_version": None,
            }
        )

        try:
            out = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .split("\n")[device_idx]
            )
            info["driver_version"] = out
        except Exception:
            pass

        return info
    except Exception as e:
        info["error"] = str(e)
        return info


def find_results_file(results_dir):
    """Find lm_eval results file (handles subdirectory and timestamped names)."""
    # First, check for results.json directly
    direct_path = os.path.join(results_dir, "results.json")
    if os.path.exists(direct_path):
        return direct_path

    # lm_eval writes to <model_name_sanitized>/results_<timestamp>.json
    # Collect all matching files and return the latest (by filename sort,
    # since lm_eval uses ISO timestamps like results_2026-02-16T12:34:56.json)
    candidates = []
    for item in os.listdir(results_dir):
        subdir = os.path.join(results_dir, item)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.startswith("results_") and filename.endswith(".json"):
                    candidates.append(os.path.join(subdir, filename))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0]

    return None


def extract_accuracy(results_dir, task="gsm8k"):
    """Extract accuracy from lm_eval results.json file."""
    results_path = find_results_file(results_dir)
    if results_path is None:
        return None

    try:
        with open(results_path) as f:
            data = json.load(f)

        task_results = data.get("results", {}).get(task, {})

        # Try different accuracy keys (task-dependent)
        accuracy_keys = [
            "exact_match,flexible-extract",
            "exact_match,get-answer",
            "exact_match,strict-match",
            "exact_match",
            "acc,none",
            "acc_norm,none",
        ]

        for key in accuracy_keys:
            if key in task_results:
                return task_results[key]

        # Fallback: return first numeric metric found
        for key, value in task_results.items():
            if isinstance(value, (int, float)) and not key.endswith("_stderr"):
                return value

        return None
    except Exception as e:
        print(f"[WARN] Could not extract accuracy: {e}")
        return None


def extract_groups(results_dir):
    """Extract the 'groups' object from the lm_eval results.json file."""
    results_path = find_results_file(results_dir)
    if results_path is None:
        return None

    try:
        with open(results_path) as f:
            data = json.load(f)
        # .get() returns None if the key doesn't exist, which is handled below
        return data.get("groups")
    except Exception as e:
        print(f"[WARN] Could not extract groups from results file: {e}")
        return None


def write_summary(output_dir, task, config, throughput_metrics, timestamp,
                  model_args=None, eval_params=None):
    """Write a single summary.json combining accuracy, throughput, config, and metadata."""
    summary = {
        "task": task,
        "config": config,
        "timestamp": timestamp,
        "accuracy": extract_accuracy(output_dir, task),
        "throughput": throughput_metrics,
        "gpu_info": get_gpu_info(),
        "git": get_git_info(),
        "environment": get_environment_info(),
    }

    # Extract and add group results if they exist
    group_results = extract_groups(output_dir)
    if group_results:
        summary["group_results"] = group_results

    # Add optional model args and eval params
    if model_args is not None:
        summary["model_args"] = model_args
    if eval_params is not None:
        summary["eval_params"] = eval_params

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[LOG] Summary saved to {summary_path}")
    return summary
