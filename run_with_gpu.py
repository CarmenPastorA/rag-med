import os
import sys
import subprocess
import time
import signal
import argparse

def get_least_used_gpu(exclude_gpus=None):
    """
    Returns the GPU index with the lowest memory usage, excluding any GPUs specified.
    """
    exclude_gpus = exclude_gpus or set()

    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used,index",
            "--format=csv,noheader,nounits"
        ]).decode()

        usage = []
        for line in output.strip().split("\n"):
            mem_used, idx = line.split(",")
            idx = idx.strip()
            if idx not in exclude_gpus:
                usage.append((int(mem_used.strip()), idx))

        if not usage:
            raise RuntimeError("No suitable GPUs available after exclusions.")

        sorted_gpus = sorted(usage, key=lambda x: x[0])
        return sorted_gpus[0][1]
    except Exception as e:
        print(f"[Error] Failed to detect available GPU: {e}")
        return "0"

def show_gpu_status(label):
    """
    Prints current GPU usage with a label.
    """
    print(f"\n=== [GPU STATUS] {label} ===")
    subprocess.run(["nvidia-smi"])

def main():
    parser = argparse.ArgumentParser(description="Run a script on the least-used GPU or a specified one.")
    parser.add_argument("script", help="Path to the script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for the target script")
    parser.add_argument("--prefer-gpu", type=str, help="Force use of a specific GPU index")
    parser.add_argument("--exclude-gpus", type=str, help="Comma-separated list of GPU indices to exclude")
    args = parser.parse_args()

    # Determine which GPU to use
    if args.prefer_gpu:
        gpu_id = args.prefer_gpu
        print(f"\n[Info] Forcing use of GPU {gpu_id}")
    else:
        exclude_set = set(args.exclude_gpus.split(",")) if args.exclude_gpus else set()
        gpu_id = get_least_used_gpu(exclude_gpus=exclude_set)
        print(f"\n[Info] Automatically selected GPU {gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"\nLaunching on GPU {gpu_id}")

    show_gpu_status("BEFORE RUN")

    start_time = time.time()

    try:
        proc = subprocess.Popen(["python", args.script] + args.script_args)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Info] Interrupted. Sending SIGINT to the child process...")
        proc.send_signal(signal.SIGINT)
        proc.wait()
    except Exception as e:
        print(f"[Error] Exception during subprocess: {e}")
    finally:
        duration = time.time() - start_time
        print(f"\nScript finished in {duration:.2f} seconds")
        show_gpu_status("AFTER RUN")

if __name__ == "__main__":
    main()

