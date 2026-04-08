import os
import json
import argparse
import sys
from typing import List, Optional, Dict, Any
from openai import OpenAI
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

# --- API Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_client_and_models():
    """Returns (OpenAI client, list of model IDs) based on available API keys."""
    if GROQ_API_KEY:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-3b-preview"]
        return client, models

    if HF_TOKEN:
        base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        client = OpenAI(base_url=base, api_key=HF_TOKEN)
        model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        return client, [model_name]

    return None, []

# --- Strategic Hierarchies ---
STRATEGIES = {
    "Single Machine": """
1. FAMILY MATCH (TOP PRIORITY): Batching is CRITICAL here. Prioritize jobs matching the machine's current family to avoid setup penalties.
2. RUSH JOB: Assign rush jobs only after checking for family matches.
3. UTILIZATION MANDATE: NEVER leave the machine idle if jobs are pending. If no family match exists and no rush jobs are available, you MUST take the setup penalty and assign the job with the earliest due date immediately. Sitting idle is always more expensive than a setup penalty if work is waiting.
""",
    "Multi-Machine": """
1. RUSH JOBS FIRST: Assign urgent tasks across all available machines.
2. FAMILY MATCH: Match jobs to machines with the same current family.
3. LOAD BALANCE: Maximize utilization by using all idle machines. Never leave a machine idle if a job is pending.
4. LEAST PENALTY SWITCH: Use EDD and priority for non-matching jobs.
""",
    "Dynamic Arrivals": """
1. RUSH JOBS FIRST: Process urgent arrivals immediately.
2. FAMILY MATCH: Group jobs by family to minimize changeover costs.
3. KEEP FLEXIBLE: If machines are idle and a high-priority arrival is expected soon, you may wait briefly, but otherwise MAXIMIZE UTILIZATION.
4. MIN FUTURE PENALTY: Prioritize high-priority and short-deadline jobs.
"""
}

def detect_task_strategy(task_id: str, mode: str = "Auto") -> str:
    if mode != "Auto":
        return mode
    if "single_machine" in task_id:
        return "Single Machine"
    if "hard" in task_id or "dynamic" in task_id:
        return "Dynamic Arrivals"
    return "Multi-Machine"

def get_system_prompt(strategy_name: str) -> str:
    hierarchy = STRATEGIES.get(strategy_name, STRATEGIES["Multi-Machine"])
    return f"""You are an ADAPTIVE manufacturing scheduler. YOU MUST follow this {strategy_name} hierarchy:

{hierarchy}

CRITICAL SCHEMA RULES:
- Respond ONLY with a valid JSON object.
- 'assignments' must contain ONLY valid {{machine_id, job_id}} for IDLE machines.
- PROVIDE a 'score_breakdown' object showing weights (0-100) for your primary decision:
  {{ "family_match_bonus": X, "rush_priority_bonus": Y, "idle_penalty_avoidance": Z }}

Example Format:
{{
  "assignments": [{{"machine_id": "M1", "job_id": "J1"}}],
  "reasoning": "Matching family A to machine M1 to avoid setup cost.",
  "score_breakdown": {{ "family_match": 80, "utilization": 20 }}
}}"""

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

def clean_assignments(assignments: list, obs) -> list:
    valid_assignments = []
    seen_machines, seen_jobs = set(), set()
    pending_job_ids = {job.job_id for job in obs.jobs_pending}
    idle_machine_ids = {m.machine_id for m in obs.machines if m.status == "idle"}

    for a in assignments:
        if not isinstance(a, dict): continue
        m_id, j_id = a.get("machine_id"), a.get("job_id")
        if m_id not in idle_machine_ids or m_id in seen_machines or j_id in seen_jobs or j_id not in pending_job_ids:
            continue
        valid_assignments.append({"machine_id": m_id, "job_id": j_id})
        seen_machines.add(m_id)
        seen_jobs.add(j_id)
    return valid_assignments

# --- Gradio streaming generator ---
def run_inference_generator(task_id: str, strategy_mode: str = "Auto", model_override: Optional[str] = None):
    client, models_to_try = get_client_and_models()
    if model_override:
        models_to_try = [model_override] + models_to_try
    
    if not client:
        yield {"logs": "Error: No API key set.", "completed": 0, "total_reward": 0.0, "status": "Error", "score": 0.0}
        return

    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()
    
    strategy = detect_task_strategy(task_id, strategy_mode)
    system_prompt = get_system_prompt(strategy)
    
    log_history = [log_start(task=task_id, env="shop_scheduler_env", model=models_to_try[0])]
    yield {
        "logs": f"--- Strategy: {strategy} ---\n" + "\n".join(log_history),
        "completed": len(obs.completed_jobs),
        "total_reward": 0.0,
        "status": "Starting",
        "score": 0.0
    }

    done, step_count, rewards = False, 0, []

    while not done and step_count < 25:
        prompt = obs.model_dump_json()
        success_call, action_data = False, {}

        for model_attempt in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=model_attempt,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    timeout=30
                )
                content = resp.choices[0].message.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                data = json.loads(content)
                if "assignments" in data:
                    data["assignments"] = clean_assignments(data["assignments"], obs)
                action_data = data
                success_call = True
                break
            except Exception as e:
                print(f"DEBUG: {model_attempt} failed: {e}")
                continue

        if not success_call: break

        step_count += 1
        try:
            action = Action(**action_data)
            obs, reward_obj, done, info = env.step(action)
            rewards.append(reward_obj.value)
            
            scoring = action_data.get("score_breakdown", {})
            score_str = f" [Score: {scoring}]" if scoring else ""
            
            step_log = log_step(step=step_count, action=json.dumps(action_data), reward=reward_obj.value, done=done, error=info.get("last_action_error"))
            log_history.append(step_log + score_str)
        except Exception as e:
            done = True
            log_history.append(f"Step {step_count} failed: {e}")
        
        current_state = env.state()
        yield {
            "logs": "\n".join(log_history),
            "completed": len(obs.completed_jobs),
            "total_reward": round(sum(rewards), 2),
            "status": "Finished" if done else "Processing",
            "score": current_state.normalized_score
        }

    state = env.state()
    log_history.append(log_end(success=state.normalized_score >= 0.1, steps=step_count, score=state.normalized_score, rewards=rewards))
    yield {
        "logs": "\n".join(log_history),
        "completed": len(obs.completed_jobs),
        "total_reward": round(sum(rewards), 2),
        "status": "Completed",
        "score": state.normalized_score
    }

# --- CLI entry point ---
def run_inference(task_id: str, strategy_mode: str = "Auto"):
    for update in run_inference_generator(task_id, strategy_mode):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="easy_single_machine")
    parser.add_argument("--strategy", type=str, default="Auto", choices=["Auto", "Single Machine", "Multi-Machine", "Dynamic Arrivals"])
    args = parser.parse_args()
    run_inference(args.task_id, args.strategy)
