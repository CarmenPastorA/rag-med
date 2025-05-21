import os
import sys
import subprocess
import re

def get_least_used_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,index", "--format=csv,noheader,nounits"]).decode()
        gpu_lines = output.strip().split("\n")
        usage = [(int(line.split(",")[0].strip()), line.split(",")[1].strip()) for line in gpu_lines]
        # Ordena por menos memoria usada
        sorted_gpus = sorted(usage, key=lambda x: x[0])
        return sorted_gpus[0][1]  # índice de la GPU menos usada
    except Exception as e:
        print(f"Error detectando GPU libre: {e}")
        return "0"  # fallback: usar GPU 0

def main():
    if len(sys.argv) < 2:
        print("Uso: python run_with_gpu.py script.py [args...]")
        sys.exit(1)

    # GPU más libre
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Ejecutando en GPU {gpu_id}")

    script = sys.argv[1]
    args = sys.argv[2:]

    subprocess.run(["python", script] + args)

if __name__ == "__main__":
    main()
