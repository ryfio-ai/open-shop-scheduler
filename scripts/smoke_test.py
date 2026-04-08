import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action, MachineAssignment

def run_smoke_test():
    print("Starting smoke test for ShopSchedulerEnv...")
    env = ShopSchedulerEnv(task_id="easy_single_machine")
    obs = env.reset()
    
    print(f"Initial State: Time={obs.current_time}, Pending Jobs={len(obs.jobs_pending)}")
    
    done = False
    step = 0
    while not done and step < 10:
        step += 1
        # Heuristic: Pick the job with the earliest due date
        if obs.jobs_pending:
            # Sort by due_time
            sorted_jobs = sorted(obs.jobs_pending, key=lambda x: x.due_time)
            target_job = sorted_jobs[0]
            
            # Create action
            action = Action(
                assignments=[
                    MachineAssignment(machine_id="M1", job_id=target_job.job_id)
                ],
                reasoning=f"Heuristic: Selecting job {target_job.job_id} with due date {target_job.due_time}"
            )
        else:
            # No pending jobs, just step (idle)
            action = Action(assignments=[])
            
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: Time={obs.current_time}, Reward={reward.value}, Score={info['score']:.2f}, Pending={len(obs.jobs_pending)}, Completed={len(obs.completed_jobs)}")
        
    print("Smoke test completed.")
    final_state = env.state()
    print(f"Final Score: {final_state.normalized_score:.4f}")
    
    if final_state.normalized_score > 0.5:
        print("Test Result: SUCCESS (Score looks reasonable)")
    else:
        print("Test Result: WARNING (Score is low, check logic)")

if __name__ == "__main__":
    run_smoke_test()
