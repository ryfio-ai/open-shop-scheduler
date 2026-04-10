import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action, MachineAssignment

def run_smoke_test(task_id: str):
    print(f"--- Starting smoke test for {task_id} ---")
    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()
    
    print(f"Initial State: Time={obs.current_time}, Pending Jobs={len(obs.jobs_pending)}")
    
    done = False
    step = 0
    while not done and step < 50:
        step += 1
        # Heuristic: Pick the job with the earliest due date for ANY idle machine
        idle_machines = [m for m in obs.machines if m.status == "idle"]
        assignments = []
        
        if idle_machines and obs.jobs_pending:
            sorted_jobs = sorted(obs.jobs_pending, key=lambda x: x.due_time)
            for m in idle_machines:
                if sorted_jobs:
                    job = sorted_jobs.pop(0)
                    assignments.append(MachineAssignment(machine_id=m.machine_id, job_id=job.job_id))
            
            action = Action(
                assignments=assignments,
                reasoning=f"Heuristic: Assigned {len(assignments)} jobs based on EDD."
            )
        else:
            action = Action(assignments=[])
            
        obs, reward, done, info = env.step(action)
        if step % 5 == 0 or done:
            print(f"Step {step}: Time={obs.current_time}, Score={info['score']:.2f}, Completed={len(obs.completed_jobs)}/{len(env.state().jobs)}")
        
    final_state = env.state()
    print(f"Final Results for {task_id}: Score={final_state.normalized_score:.4f}, Done={final_state.done}")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    tasks = ["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"]
    for tid in tasks:
        try:
            run_smoke_test(tid)
        except Exception as e:
            print(f"Error testing {tid}: {e}")
