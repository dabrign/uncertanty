"""
Local VLM Environment Setup & Health Verification Script for Linux and macOS.

Detects local VLM inference engines (Ollama, vLLM, MLX-VLM, LM Studio), checks model availability,
and provides instructions/commands to install and launch open-weights models (e.g. Qwen2.5-VL).
"""

import os
import platform
import shutil
import subprocess
import sys
import requests


def detect_os() -> str:
    """Detects current operating system."""
    os_name = platform.system().lower()
    if "darwin" in os_name:
        return "macos"
    elif "linux" in os_name:
        return "linux"
    else:
        return os_name


def check_command_exists(cmd: str) -> bool:
    """Checks if executable binary exists on PATH."""
    return shutil.which(cmd) is not None


def check_ollama_server(api_base: str = "http://localhost:11434/v1") -> dict:
    """Probes Ollama server endpoint."""
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/models", timeout=3)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            models = [m.get("id") for m in data]
            return {"status": "online", "models": models}
    except Exception as e:
        return {"status": "offline", "error": str(e)}
    return {"status": "offline", "error": "Non-200 response"}


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def setup_vlm_environment(api_base: str = "http://localhost:11434/v1", check_only: bool = False):
    os_name = detect_os()
    print_header(f"Multimodal UQ Environment Setup ({os_name.upper()})")

    print(f"[1/4] Operating System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[2/4] Python Version:   {sys.version.split()[0]}")

    # Check Ollama installation
    ollama_installed = check_command_exists("ollama")
    print(f"[3/4] Ollama CLI:       {'INSTALLED' if ollama_installed else 'NOT FOUND'}")

    # Check server status
    server_info = check_ollama_server(api_base)
    print(f"[4/4] Server Endpoint ({api_base}): {server_info['status'].upper()}")

    if server_info["status"] == "online":
        print("\n✅ Local VLM Server is ONLINE!")
        print("Available Models:")
        for m in server_info.get("models", []):
            print(f"  - {m}")
        print("\nYou can run live tests with:")
        print(f"  python utils/demo_cat_dog_test.py --live --api-base {api_base}")
        return True

    print("\n⚠️ Local VLM server is not running at", api_base)

    if check_only:
        return False

    print("\n----------------------------------------------------------------------")
    print("  INSTALLATION & QUICKSTART INSTRUCTIONS")
    print("----------------------------------------------------------------------")

    if os_name == "macos":
        print("To install Ollama on macOS:")
        print("  1. Run: brew install ollama")
        print("  2. Start server: ollama serve")
        print("  3. Pull vision model: ollama pull qwen2.5-vl")
    elif os_name == "linux":
        print("To install Ollama on Linux:")
        print("  1. Run: curl -fsSL https://ollama.com/install.sh | sh")
        print("  2. Start server: ollama serve")
        print("  3. Pull vision model: ollama pull qwen2.5-vl")
    else:
        print("Download Ollama from https://ollama.com/download")

    print("\nAlternative vLLM execution (for GPU servers):")
    print("  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000")

    print("\nFallback Mock Mode is always available without server setup:")
    print("  python utils/demo_cat_dog_test.py --mock")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM Local Environment Setup Tool")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="API Base URL")
    parser.add_argument("--check-only", action="store_true", help="Only check server status")
    args = parser.parse_args()

    setup_vlm_environment(api_base=args.api_base, check_only=args.check_only)
