import yaml
import sys
try:
    from openenv.schemas.env import EnvironmentManifest
    with open('openenv.yaml') as f:
        data = yaml.safe_load(f)
        print('* Raw Tasks:', len(data.get('tasks', [])))
        env = EnvironmentManifest(**data)
        print('* Validated Tasks:', len(env.tasks))
except Exception as e:
    print('Error:', e)
