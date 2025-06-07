"""Utility script to start the vLLM server with sensible defaults."""

import os
import subprocess

# Ensure that GPUs are listed in a stable order based on PCI BUS ID. This helps
# when multiple GPUs are available.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Command to launch the OpenAI-compatible API server from vLLM.
cmd = [
    "python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "--tokenizer",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "--dtype",
    "float16",
    "--gpu-memory-utilization",
    "0.8",
    "--max-model-len",
    "4096",
    "--port",
    "8000",
]

print("[🚀] Starting vLLM server...")
subprocess.run(cmd)
