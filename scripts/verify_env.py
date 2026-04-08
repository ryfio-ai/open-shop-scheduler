import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action, MachineAssignment
from inference import log_start, log_step, log_end

def heuristic_agent(obs):
    if not obs.jobs_pending:
        return Action(assignments=[], reasoning="No pending jobs.")
    
    # Simple Earliest Due Date (EDD) heuristic
    pending = sorted(obs.jobs_pending, key=lambda x: x.due_time)
    idle_machines = [m for m in obs.machines if m.status == "idle"]
    
    assignments = []
    for m in idle_machines:
        if pending:
            job = pending.pop(0)
            assignments.append(MachineAssignment(machine_id=m.machine_id, job_id=job.job_id))
            
    return Action(assignments=assignments, reasoning="EDD heuristic.")

def verify_task(task_id):
    print(f"--- Verifying Task: {task_id} ---")
    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()
    
    log_start(task=task_id, env="shop_scheduler_env", model="HeuristicEDD")
    
    done = False
    step = 0
    rewards = []
    while not done:
        step += 1
        action = heuristic_agent(obs)
        obs, reward_obj, done, info = env.step(action)
        reward = reward_obj.value
        rewards.append(reward)
        
        action_str = action.model_dump_json().replace(" ", "")
        log_step(step=step, action=action_str, reward=reward, done=done, error=info.get("last_action_error"))
        
    final_state = env.state()
    log_end(success=final_state.normalized_score >= 0.1, steps=step, score=final_state.normalized_score, rewards=rewards)
    print("----------------------------\n")

if __name__ == "__main__":
    tasks = ["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"]
    for task in tasks:
        try:
            verify_task(task)
        except Exception as e:
            print(f"Error verifying {task}: {e}")
