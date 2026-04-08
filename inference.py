import os
import json
import argparse
from typing import List
from openai import OpenAI
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

def run_inference_generator(task_id: str):
    # Load configuration from environment variables
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    hf_token = os.getenv("HF_TOKEN", "")

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token if hf_token else "dummy-key"
    )

    env = ShopSchedulerEnv(task_id=task_id)
    obs = env.reset()

    start_log = f"[START] task={task_id} env=shop_scheduler_env model={model_name}"
    print(start_log)
    yield start_log

    done = False
    step_count = 0
    rewards = []
    
    system_prompt = (
        "You are an expert manufacturing scheduler. Your goal is to assign jobs to machines to minimize tardiness. "
        "Domain knowledge: Changing job families on a machine incurs a setup time penalty. Observe 'family' attributes. "
        "You will receive an observation in JSON format. Respond with a JSON object following the Action schema: "
        '{"assignments": [{"machine_id": "M1", "job_id": "J1"}], "reasoning": "..."}'
    )

    while not done:
        step_count += 1
        
        # Prepare the conversation for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs.model_dump_json()}
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_content = response.choices[0].message.content
            action_dict = json.loads(action_content)
            action = Action(**action_dict)
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            rewards.append(reward.value)
            
            # Print step log
            action_str = json.dumps(action_dict).replace(" ", "")
            error_str = info.get("last_action_error") or "null"
            step_log = f"[STEP] step={step_count} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error={error_str}"
            print(step_log)
            yield step_log

        except Exception as e:
            err_log = f"[STEP] step={step_count} action=null reward=0.00 done=true error={str(e)}"
            print(err_log)
            yield err_log
            break

    # Calculate final results
    final_state = env.state()
    success = all(j.status == "completed" for j in final_state.jobs)
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    
    # Print end log
    end_log = f"[END] success={str(success).lower()} steps={step_count} score={final_state.normalized_score:.4f} rewards={rewards_str}"
    print(end_log)
    yield end_log

def run_inference(task_id: str):
    for _ in run_inference_generator(task_id):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="easy_single_machine")
    args = parser.parse_args()
    run_inference(args.task_id)
