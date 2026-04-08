import requests
import json
import time
import argparse

def validate_environment(url: str):
    """Run a battery of checks against the OpenEnv server."""
    print(f"--- Validating OpenEnv at {url} ---")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{url}/health")
        resp.raise_for_status()
        print(f"[SUCCESS] Health check: {resp.json()}")
    except Exception as e:
        print(f"[FAILED] Health check: {e}")
        return

    # 2. Reset (steady_state)
    try:
        # OpenEnv v10 uses POST /reset with task_id in body
        print(f"\n--- Testing Reset (steady_state) ---")
        resp = requests.post(f"{url}/reset", json={"task_id": "steady_state"})
        resp.raise_for_status()
        print("[SUCCESS] Reset (steady_state) endpoint works")
        data = resp.json()
        obs = data.get("observation", {})
        if "text_summary" in obs:
            print(f"[SUCCESS] Received observation summary: {obs['text_summary'][:50]}...")
    except Exception as e:
        print(f"[FAILED] Reset (steady_state): {e}")

    # 3. State
    try:
        # OpenEnv v10 uses GET /state
        print(f"\n--- Testing State ---")
        resp = requests.get(f"{url}/state")
        resp.raise_for_status()
        print(f"[SUCCESS] State endpoint works: {resp.json().get('episode_id', 'no-id')}")
    except Exception as e:
        print(f"[FAILED] State endpoint: {e}")

    # 4. Step
    try:
        # OpenEnv v10 uses POST /step
        print(f"\n--- Testing Step ---")
        payload = {
            "action": {"decisions": []}
        }
        resp = requests.post(f"{url}/step", json=payload)
        resp.raise_for_status()
        print("[SUCCESS] Step endpoint works")
    except Exception as e:
        print(f"[FAILED] Step endpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860", help="Base URL of the OpenEnv server")
    args = parser.parse_args()
    
    validate_environment(args.url)
