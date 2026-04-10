from envs.shop_scheduler_env.graders import grade_episode
from envs.shop_scheduler_env.models import EnvState

def grade(state):
    # Handle dict input if provided
    if isinstance(state, dict):
        state = EnvState.model_validate(state)
    
    try:
        raw_score = grade_episode(state)
        # Ensure score is strictly between 0 and 1 (Scaler requirement)
        return max(0.05, min(0.95, float(raw_score)))
    except:
        return 0.1
