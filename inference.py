import os
import json
import argparse
import sys
from typing import List, Optional
from openai import OpenAI
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

# Mandatory environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") # No default as per requirements

def log_start(task: str, env: str, model: str) -> None:
    line = f"[START] task={task} env={env} model={model}"
    print(line, flush=True)
    return line

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # EXACT: [STEP] + 2 spaces
    line = f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"
    print(line, flush=True)
    return line

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # EXACT: [END] + 3 spaces
    line = f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"
    print(line, flush=True)
    return line

# --- ELITE AGENT PROMPT ---
ELITE_PROMPT = (
    "You are an ELITE manufacturing scheduler. Your goal is to achieve a 1.000 score by minimizing tardiness.\n\n"
    "CORE RULES:\n"
    "1. PENALTY COST: Switching families (e.g., A -> B) costs 2 full units. This is a MASSIVE penalty. Avoid it at all costs.\n"
    "2. FAMILY BATCHING: If a machine just finished a Family A job, you MUST assign another Family A job to it if any are pending. This is your top priority.\n"
    "3. RUSH JOBS: Jobs marked 'rush' are critical. Process them immediately, but still try to use a machine already in that family if possible.\n"
    "4. IDLE MACHINES: A machine is 'idle' if its current job is null. Use idle machines immediately to keep the factory moving.\n"
    "5. DYNAMIC ARRIVALS: Watch the 'arrival_time'. You cannot assign a job before its arrival time.\n\n"
    "THINKING PROCESS:\n"
    "- Step 1: Look at each machine. Is it idle?\n"
    "- Step 2: For each idle machine, what was its last family? Search 'jobs_pending' for jobs with THAT SAME family.\n"
    "- Step 3: If no same-family jobs exist, pick the highest priority job from another family.\n\n"
    "Respond with a JSON object following the Action schema:\n"
    '{"assignments": [{"machine_id": "M1", "job_id": "J1"}], "reasoning": "Batching Family A to avoid 2-unit penalty."}'
)

def run_inference_generator(task_id: str):
    """Generator version for Gradio streaming that yields the full history."""
    if not HF_TOKEN:
        yield "Error: HF_TOKEN environment variable is not set."
        return

    log_history = []

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    start_log = log_start(task=task_id, env="shop_scheduler_env", model=MODEL_NAME)
    log_history.append(start_log)
    yield "\n".join(log_history)

    done = False
    step_count = 0
    rewards = []

    try:
        while not done and step_count < 15:
            step_count += 1
            prompt = f"Current state: {obs.model_dump_json()}"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": ELITE_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                action_data = json.loads(response.choices[0].message.content)
                action = Action(**action_data)
                
                obs, reward_obj, done, info = env.step(action)
                reward = reward_obj.value
                error_msg = None
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
                action_data = {}

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

def run_inference(task_id: str):
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    log_start(task=task_id, env="shop_scheduler_env", model=MODEL_NAME)

    done = False
    step_count = 0
    rewards = []
    
    try:
        while not done:
            step_count += 1
            
            messages = [
                {"role": "system", "content": ELITE_PROMPT},
                {"role": "user", "content": obs.model_dump_json()}
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_content = response.choices[0].message.content
            action_dict = json.loads(action_content)
            action = Action(**action_dict)
            
            obs, reward_obj, done, info = env.step(action)
            reward = reward_obj.value
            rewards.append(reward)
            
            action_str = json.dumps(action_dict).replace(" ", "")
            error = info.get("last_action_error")
            
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)

            if step_count >= 20: # Safety break
                break

    except Exception as e:
        # Emit END even on exception
        pass
    finally:
        final_state = env.state()
        success = final_state.normalized_score >= 0.1
        log_end(
            success=success,
            steps=step_count,
            score=final_state.normalized_score,
            rewards=rewards
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="easy_single_machine")
    args = parser.parse_args()
    run_inference(args.task_id)
