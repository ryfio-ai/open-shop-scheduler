import os
import sys

# Add the root directory to the python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action
from envs.shop_scheduler_env.graders import grade_episode

class BaseGrader:
    def __init__(self, task_id):
        self.task_id = task_id

    def grade(self, env=None, *args, **kwargs) -> float:
        """
        Standard OpenEnv grading interface.
        If env is provided, we grade the current state.
        If not, we run a quick greedy episode.
        """
        try:
            if env is not None:
                # If env is a dict (some validators do this), convert to a state object
                from envs.shop_scheduler_env.models import EnvState
                if isinstance(env, dict):
                    state = EnvState.model_validate(env)
                else:
                    # Assume it's the Env instance or State object
                    state = getattr(env, "state", lambda: env)()
                
                raw_score = grade_episode(state)
            else:
                # Fallback: Run quick greedy episode
                env = ShopSchedulerEnv(task_id=self.task_id)
                obs = env.reset()
                done = False
                steps = 0
                while not done and steps < 30:
                    assignments = []
                    idle_machines = [m.machine_id for m in obs.machines if m.status == "idle"]
                    pending_jobs = [j.job_id for j in obs.jobs_pending]
                    for m_id, j_id in zip(idle_machines, pending_jobs):
                        assignments.append({"machine_id": m_id, "job_id": j_id})
                    action = Action(assignments=assignments)
                    obs, _, done, _ = env.step(action)
                    steps += 1
                raw_score = grade_episode(env.state())

            # Strict clamping [0.05, 0.95] as required by dashboard
            return max(0.05, min(0.95, float(raw_score)))
        except Exception as e:
            print(f"Grader Error: {e}")
            return 0.05

class EasyGrader(BaseGrader):
    def __init__(self):
        super().__init__("easy_single_machine")

class MediumGrader(BaseGrader):
    def __init__(self):
        super().__init__("medium_parallel_changeover")

class HardGrader(BaseGrader):
    def __init__(self):
        super().__init__("hard_dynamic_arrivals")
