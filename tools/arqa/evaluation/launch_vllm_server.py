import os
import subprocess

# Asegura que las GPUs se ordenen por BUS ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "mistralai/Mistral-7B-Instruct-v0.2",
    "--tokenizer", "mistralai/Mistral-7B-Instruct-v0.2",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.8",
    "--max-model-len", "4096",
    "--port", "8000"
]

print("[🚀] Lanzando servidor vLLM...")
subprocess.run(cmd)
