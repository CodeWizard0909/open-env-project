#!/usr/bin/env python3
# Copyright (c) 2026 Hospital Bed Env Team
# Baseline Inference Script for Hackathon Submission

import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from hospital_bed_env.models import HospitalAction, PatientDecision
from hospital_bed_env.client import HospitalEnv


# Configuration matches hackathon spec exactly
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "hospital-bed-env")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "hospital_bed_env"

# Maximum steps just for safety (controlled by task max_steps)
MAX_STEPS = 100
TEMPERATURE = 0.5
MAX_TOKENS = 500

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Hospital Operations Manager.
    Your job is to read the WARD STATUS and WAITING PATIENTS queue and decide exactly what action to take for EACH patient in the queue.
    
    If beds are full, DO NOT admit. Defer instead, or discharge someone IF they are recovered.
    If a patient is critical/urgent, prioritize admitting them.
    
    You MUST output valid JSON representing the actions to take. 
    Format your response EXACTLY as the following JSON structure:
    {
      "decisions": [
        {
          "patient_id": "PXXXX",
          "action": "admit_icu",
          "discharge_patient_id": null
        },
        ... one decision for EACH patient in ED queue
      ]
    }
    
    Possible actions: admit_icu, admit_general, admit_surgical, defer, discharge
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, text_summary: str, last_reward: float, history: List[str]) -> str:
    start_idx = max(0, len(history) - 3)
    recent_history = [history[i] for i in range(start_idx, len(history))]
    history_block = "\n".join(recent_history) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last Reward: {last_reward:.2f}
        Recent History:
        {history_block}
        
        Current State:
        {text_summary}
        
        What are your decisions for ALL waiting patients? Respond with JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, text_summary: str, last_reward: float, history: List[str]) -> HospitalAction:
    user_prompt = build_user_prompt(step, text_summary, last_reward, history)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"}
        )
        text = completion.choices[0].message.content or "{}"
        data = json.loads(text)
        
        # Parse output into our typing format
        decisions = []
        for d in data.get("decisions", []):
            decisions.append(PatientDecision(
                patient_id=d.get("patient_id", ""),
                action=d.get("action", "defer"),
                discharge_patient_id=d.get("discharge_patient_id")
            ))
            
        return HospitalAction(decisions=decisions)
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed or parse error: {exc}", flush=True)
        # Fallback greedy action (defer everyone)
        return HospitalAction(decisions=[])


async def run_task(client: OpenAI, task_name: str, port: int) -> float:
    # Use generic localhost for now, but in standard OpenEnv setup it might be deployed
    base_url = f"http://localhost:{port}"
    env = HospitalEnv(base_url=base_url)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_name)
        obs = result.observation
        text_summary = obs.text_summary
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, text_summary, last_reward, history)

            # Simple stringification of action for logging
            action_str = f"decisions={len(action.decisions)}"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            text_summary = obs.text_summary
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Record history
            h_str = f"Step {step}: Took {len(action.decisions)} decisions -> reward {reward:+.2f}"
            history.append(h_str)

            if done:
                break

        # Hackathon spec says normalized score from 0.0 to 1.0 is in metadata
        total_norm_score = obs.metadata.get("normalized_score", 0.5)
        success = total_norm_score > 0.5  # Simple success threshold

    except Exception as e:
        print(f"[DEBUG] Exception during task {task_name}: {e}\n\n", flush=True)
        import traceback
        traceback.print_exc()
        total_norm_score = 0.05
    finally:
        # Close the environment connection
        try:
            await env.close()
        except Exception:
            pass

    # Safety clamp: validator requires score strictly in (0, 1)
    total_norm_score = max(0.05, min(0.95, total_norm_score))
    log_end(success=success, steps=steps_taken, score=total_norm_score, rewards=rewards)

    return total_norm_score


async def main() -> None:
    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # We assume the FastAPI server is ALREADY running on port 7860 before running inference
    # Start it manually, or the Hackathon validator handles it.
    # OpenEnv validates local container running. We use port 7860.

    # Run all three tasks sequentially
    tasks = ["steady_state", "surge_event", "mass_casualty"]

    for task in tasks:
        await run_task(client, task, port=7860)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
