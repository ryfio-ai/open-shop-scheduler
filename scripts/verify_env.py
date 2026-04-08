import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action, MachineAssignment

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
    
    print(f"[START] task={task_id}")
    
    done = False
    step = 0
    while not done:
        step += 1
        action = heuristic_agent(obs)
        obs, reward, done, info = env.step(action)
        
        # Verify STEP format
        action_str = action.model_dump_json().replace(" ", "")
        error_str = info.get("last_action_error") or "null"
        print(f"[STEP] step={step} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error={error_str}")
        
    final_state = env.state()
    print(f"[END] score={final_state.normalized_score:.4f}")
    print("----------------------------\n")

if __name__ == "__main__":
    tasks = ["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"]
    for task in tasks:
        try:
            verify_task(task)
        except Exception as e:
            print(f"Error verifying {task}: {e}")
