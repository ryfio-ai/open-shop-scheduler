import os
import json
import argparse
import sys
import time
from typing import List, Optional
from openai import OpenAI
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

# Mandatory environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fallback model list to ensure reliability
DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2-9b-it",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

def log_start(task: str, env: str, model: str) -> None:
    line = f"[START] task={task} env={env} model={model}"
    print(line, flush=True)
    return line

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    line = f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"
    print(line, flush=True)
    return line

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    line = f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"
    print(line, flush=True)
    return line

ELITE_PROMPT = (
    "You are an ELITE manufacturing scheduler. Your goal is a 1.000 score by following this STRATEGIC HIERARCHY:\n\n"
    "1. RUSH JOBS FIRST: Identify all 'rush' jobs in 'jobs_pending'. Assign them immediately to any idle machines.\n"
    "2. FAMILY MATCH: For remaining idle machines, prioritize jobs that match the machine's 'current_family' to avoid the 2-unit setup penalty.\n"
    "3. NO IDLE MACHINES: You must use ALL available machines (M1, M2, M3). If a machine is idle and a job is pending, assign it. Do not be lazy!\n"
    "4. LEAST PENALTY SWITCH: If no family match exists, choose a job that minimizes future penalties or pick the highest priority.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "- Only assign jobs where current_time >= arrival_time.\n"
    "- A 2-unit setup penalty is incurred if the job family != machine current_family.\n\n"
    "Respond with a JSON object:\n"
    '{"assignments": [{"machine_id": "M1", "job_id": "J1"}, {"machine_id": "M2", "job_id": "J2"}], "reasoning": "Reasoning here."}'
)

def run_inference_generator(task_id: str):
    if not HF_TOKEN:
        yield "Error: HF_TOKEN environment variable is not set."
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    # Determine which model to use
    selected_model = os.getenv("MODEL_NAME")
    models_to_try = [selected_model] if selected_model else DEFAULT_MODELS

    log_history = []
    current_model = "unknown"

    # We only log [START] once we successfully get a response or start the loop
    # For simplicity in the dashboard, we use the first available model in the list
    current_model = models_to_try[0]
    start_log = log_start(task=task_id, env="shop_scheduler_env", model=current_model)
    log_history.append(start_log)
    yield "\n".join(log_history)

    done = False
    step_count = 0
    rewards = []

    try:
        while not done and step_count < 15:
            step_count += 1
            prompt = f"Current state: {obs.model_dump_json()}"
            
            # Fallback Retry Logic
            action_data = {}
            error_msg = None
            success_call = False
            
            for model_attempt in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_attempt,
                        messages=[
                            {"role": "system", "content": ELITE_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        timeout=30
                    )
                    action_data = json.loads(response.choices[0].message.content)
                    current_model = model_attempt
                    success_call = True
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "402" in error_msg or "429" in error_msg or "model_not_found" in error_msg:
                        continue # Try next model
                    else:
                        break # Fatal error

            if not success_call:
                done = True
                # Use last error_msg
            else:
                try:
                    action = Action(**action_data)
                    obs, reward_obj, done, info = env.step(action)
                    reward = reward_obj.value
                    error_msg = None
                except Exception as e:
                    reward = 0.0
                    done = True
                    error_msg = str(e)

            rewards.append(reward if 'reward' in locals() else 0.0)
            step_log = log_step(step=step_count, action=json.dumps(action_data), reward=rewards[-1], done=done, error=error_msg)
            log_history.append(step_log)
            yield "\n".join(log_history)

        final_state = env.state()
        score = final_state.normalized_score
        end_log = log_end(success=(score > 0.1), steps=step_count, score=score, rewards=rewards)
        log_history.append(end_log)
        yield "\n".join(log_history)

    finally:
        pass

def run_inference(task_id: str):
    # Standard CLI version with fallback
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    selected_model = os.getenv("MODEL_NAME")
    models_to_try = [selected_model] if selected_model else DEFAULT_MODELS
    
    # Just show the first model in [START] for simplicity
    log_start(task=task_id, env="shop_scheduler_env", model=models_to_try[0])

    done = False
    step_count = 0
    rewards = []

    try:
        while not done:
            step_count += 1
            prompt = obs.model_dump_json()
            success_call = False
            action_dict = {}
            error = None

            for model_attempt in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_attempt,
                        messages=[{"role": "system", "content": ELITE_PROMPT}, {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    action_dict = json.loads(response.choices[0].message.content)
                    success_call = True
                    break
                except Exception as e:
                    if "402" in str(e) or "429" in str(e): continue
                    error = str(e); break

            if not success_call: break

            action = Action(**action_dict)
            obs, reward_obj, done, info = env.step(action)
            reward = reward_obj.value
            rewards.append(reward)
            log_step(step=step_count, action=json.dumps(action_dict).replace(" ", ""), reward=reward, done=done, error=info.get("last_action_error"))

            if step_count >= 20: break
    finally:
        final_state = env.state()
        log_end(success=final_state.normalized_score >= 0.1, steps=step_count, score=final_state.normalized_score, rewards=rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="easy_single_machine")
    args = parser.parse_args()
    run_inference(args.task_id)
