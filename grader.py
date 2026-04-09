from envs.shop_scheduler_env.graders import grade_episode

# Relay function for OpenEnv validation
def grade(state):
    return grade_episode(state)
