"""
Local VLM Runner helper: Connects to local servers (Ollama, MLX, vLLM).
"""

from typing import Any

import requests


def check_local_vlm_health(api_base: str = "http://localhost:11434/v1") -> dict[str, Any]:
    """
    Checks connection health and lists available models at an OpenAI-compatible endpoint.
    """
    base = api_base.rstrip("/")
    try:
        resp = requests.get(f"{base}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            model_ids = [m.get("id") for m in models]
            return {
                "status": "online",
                "api_base": api_base,
                "available_models": model_ids,
            }
        else:
            return {
                "status": "error",
                "http_status": resp.status_code,
                "api_base": api_base,
            }
    except Exception as e:
        return {
            "status": "offline",
            "api_base": api_base,
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check local VLM server health")
    parser.add_argument(
        "--api-base", default="http://localhost:11434/v1", help="OpenAI API Base URL"
    )
    args = parser.parse_args()

    health = check_local_vlm_health(args.api_base)
    print("Local VLM Server Status:")
    for k, v in health.items():
        print(f"  {k}: {v}")
