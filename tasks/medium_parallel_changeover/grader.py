from envs.shop_scheduler_env.graders import grade_episode

# Relay function for OpenEnv validation
def grade(state):
    raw_score = grade_episode(state)
    return max(0.01, min(0.99, float(raw_score)))
