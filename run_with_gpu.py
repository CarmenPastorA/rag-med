import os
import sys
import subprocess
import time
import signal

def get_least_used_gpu():
    """
    Checks GPU memory usage using `nvidia-smi` and returns the index of the GPU
    with the lowest memory usage. This helps avoid conflicts when running on shared systems.
    """
    try:
        output = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=memory.used,index", 
            "--format=csv,noheader,nounits"
        ]).decode()
        
        # Parse each line: (used_memory, gpu_index)
        gpu_lines = output.strip().split("\n")
        usage = [(int(line.split(",")[0].strip()), line.split(",")[1].strip()) for line in gpu_lines]
        
        # Sort GPUs by memory usage (ascending)
        sorted_gpus = sorted(usage, key=lambda x: x[0])
        
        return sorted_gpus[0][1]  # Return GPU index with least memory used
    except Exception as e:
        print(f"[Error] Failed to detect available GPU: {e}")
        return "0"  # Fallback to GPU 0 if detection fails

def show_gpu_status(label):
    """
    Prints current GPU usage with a label to indicate the timing (e.g., before or after running).
    """
    print(f"\n=== [GPU STATUS] {label} ===")
    subprocess.run(["nvidia-smi"])

def main():
    # Ensure at least one argument (the target script) is passed
    if len(sys.argv) < 2:
        print("Usage: python run_with_gpu.py script.py [args...]")
        sys.exit(1)

    # Detect and set the least used GPU
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"\nLaunching on GPU {gpu_id}")

    # Show GPU status before starting the script
    show_gpu_status("BEFORE RUN")

    # Extract script and arguments from command-line input
    script = sys.argv[1]
    args = sys.argv[2:]

    # Start timer to measure total execution time
    start_time = time.time()

    try:
        # Run the target script as a subprocess
        proc = subprocess.Popen(["python", script] + args)
        proc.wait()  # Wait for the script to finish
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n[Info] Interrupted. Sending SIGINT to the child process...")
        proc.send_signal(signal.SIGINT)
        proc.wait()
    except Exception as e:
        print(f"[Error] Exception during subprocess: {e}")
    finally:
        # Compute and display execution time
        duration = time.time() - start_time
        print(f"\nScript finished in {duration:.2f} seconds")

        # Show GPU status after script completes
        show_gpu_status("AFTER RUN")

# Entry point
if __name__ == "__main__":
    main()
