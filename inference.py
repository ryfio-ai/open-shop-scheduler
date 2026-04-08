import os
import json
import argparse
import sys
from typing import List, Optional
from openai import OpenAI
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

# --- API Configuration ---
# Priority: GROQ_API_KEY -> HF_TOKEN
# The code picks the first available provider automatically.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_client_and_models():
    """Returns (OpenAI client, list of model IDs) based on available API keys."""
    # Priority 1: Groq (free tier, very fast, reliable)
    if GROQ_API_KEY:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        models = ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]
        return client, models

    # Priority 2: HF Router (may have quota issues)
    if HF_TOKEN:
        base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        client = OpenAI(base_url=base, api_key=HF_TOKEN)
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        return client, [model_name]

    return None, []

# --- Logging helpers (OpenEnv compliant) ---
def log_start(task: str, env: str, model: str) -> str:
    line = f"[START] task={task} env={env} model={model}"
    print(line, flush=True)
    return line

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> str:
    error_val = error if error else "null"
    done_val = str(done).lower()
    line = f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"
    print(line, flush=True)
    return line

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> str:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    line = f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"
    print(line, flush=True)
    return line

# --- Elite Scheduling Prompt ---
ELITE_PROMPT = (
    "You are an ELITE manufacturing scheduler. Your goal is a 1.000 score by following this STRATEGIC HIERARCHY:\n\n"
    "1. RUSH JOBS FIRST: Identify all 'rush' jobs in 'jobs_pending'. Assign them immediately to any idle machines.\n"
    "2. FAMILY MATCH: For remaining idle machines, prioritize jobs that match the machine's 'current_family' to avoid the 2-unit setup penalty.\n"
    "3. NO IDLE MACHINES: You must use ALL available machines. If a machine is idle and a job is pending, assign it. Do not be lazy!\n"
    "4. LEAST PENALTY SWITCH: If no family match exists, choose a job that minimizes future penalties (due dates & priority).\n\n"
    "CRITICAL SCHEMA RULES:\n"
    "- Respond ONLY with a valid JSON object.\n"
    "- 'assignments' must contain ONLY valid {machine_id, job_id} objects. NO COMMENTS OR STRINGS inside the list.\n"
    "- If you have nothing to assign for a machine, do not list it or use job_id: null.\n\n"
    "JSON Format example:\n"
    '{"assignments": [{"machine_id": "M1", "job_id": "J1"}], "reasoning": "Reasoning here."}'
)

def clean_assignments(assignments: list) -> list:
    """Prunes non-dict garbage from the model output to prevent Pydantic crashes."""
    return [a for a in assignments if isinstance(a, dict) and "machine_id" in a]

# --- Gradio streaming generator ---
def run_inference_generator(task_id: str):
    client, models_to_try = get_client_and_models()
    if client is None:
        yield "Error: No API key set. Please set GROQ_API_KEY or HF_TOKEN."
        return

    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    log_history = []
    current_model = models_to_try[0]
    start_log = log_start(task=task_id, env="shop_scheduler_env", model=current_model)
    log_history.append(start_log)
    yield "\n".join(log_history)

    done = False
    step_count = 0
    rewards = []

    try:
        while not done and step_count < 25:
            step_count += 1
            prompt = f"Current state: {obs.model_dump_json()}"

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
                    # Defense: Clean the assignments before validation
                    if "assignments" in action_data:
                        action_data["assignments"] = clean_assignments(action_data["assignments"])
                    
                    current_model = model_attempt
                    success_call = True
                    break
                except Exception as e:
                    error_msg = str(e)
                    if any(code in error_msg for code in ["402", "429", "not_supported", "not supported"]):
                        continue
                    else:
                        break

            if not success_call:
                reward = 0.0
                done = True
            else:
                try:
                    action = Action(**action_data)
                    obs, reward_obj, done, info = env.step(action)
                    reward = reward_obj.value
                    error_msg = info.get("last_action_error")
                except Exception as e:
                    reward = 0.0
                    done = True
                    error_msg = f"Validation Error: {str(e)}"

            rewards.append(reward)
            step_log = log_step(step=step_count, action=json.dumps(action_data), reward=reward, done=done, error=error_msg)
            log_history.append(step_log)
            yield "\n".join(log_history)

        final_state = env.state()
        score = final_state.normalized_score
        end_log = log_end(success=(score > 0.1), steps=step_count, score=score, rewards=rewards)
        log_history.append(end_log)
        yield "\n".join(log_history)
    finally:
        pass

# --- CLI entry point ---
def run_inference(task_id: str):
    client, models_to_try = get_client_and_models()
    if client is None:
        print("Error: No API key set. Set GROQ_API_KEY or HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

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
                    if "402" in str(e) or "429" in str(e) or "not_supported" in str(e):
                        continue
                    error = str(e)
                    break

            if not success_call:
                break

            action = Action(**action_dict)
            obs, reward_obj, done, info = env.step(action)
            reward = reward_obj.value
            rewards.append(reward)
            log_step(step=step_count, action=json.dumps(action_dict).replace(" ", ""), reward=reward, done=done, error=info.get("last_action_error"))

            if step_count >= 20:
                break
    finally:
        final_state = env.state()
        log_end(success=final_state.normalized_score >= 0.1, steps=step_count, score=final_state.normalized_score, rewards=rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="easy_single_machine")
    args = parser.parse_args()
    run_inference(args.task_id)
