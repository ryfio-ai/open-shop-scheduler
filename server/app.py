import gradio as gr
import os
import sys
import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add the root directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import run_inference_generator, get_client_and_models
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action
from envs.shop_scheduler_env.graders import grade_episode

# Initialize Core API Env
api_env = ShopSchedulerEnv(task_id="easy_single_machine")
app = FastAPI(title="Ryfio-AI: Industrial Scheduler API")

# --- OpenEnv API Endpoints ---
@app.post("/reset")
async def reset():
    obs = api_env.reset()
    return JSONResponse(content=obs.model_dump())

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = Action(**data)
    obs, reward_obj, done, info = api_env.step(action)
    return JSONResponse(content={"observation": obs.model_dump(), "reward": reward_obj.value, "done": done, "info": info})

# --- Hackathon-Specific Grading Endpoints ---
@app.get("/grade/{task_id}")
async def get_grade(task_id: str):
    """Returns the current episode score for validation."""
    score = grade_episode(api_env.state)
    # Clamp to hackathon expectations if needed, but grade_episode is already 0-1
    return JSONResponse(content={
        "score": score,
        "reward": score,  # Some validators look for reward instead of score
        "status": "completed" if api_env.state.done else "in_progress",
        "reasoning": f"Current tardiness-based score for {task_id}"
    })

@app.post("/grade/{task_id}")
async def post_grade(task_id: str):
    """POST version for validators that use POST."""
    return await get_grade(task_id)

# --- Gradio Premium UI ---
def create_ui():
    _, models = get_client_and_models()
    
    css = """
    .decision-card { border-left: 5px solid #ff9800; padding: 10px; margin: 10px 0; background: #2b2b2b; }
    .metric-box { font-size: 20px; font-weight: bold; color: #4facfe; }
    """

    with gr.Blocks(title="Ryfio-AI | Adaptive Industrial Scheduler", theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"), css=css) as demo:
        gr.Markdown("# 🏭 Ryfio-AI | Adaptive Industrial Scheduler")
        gr.Markdown("An advanced manufacturing command center featuring task-aware strategy shifting and real-time decision scoring.")

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🎛️ Control Panel")
                task_id = gr.Dropdown(
                    choices=["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"],
                    value="easy_single_machine", label="Active Task"
                )
                strategy_mode = gr.Dropdown(
                    choices=["Auto", "Single Machine", "Multi-Machine", "Dynamic Arrivals"],
                    value="Auto", label="Scheduling Strategy"
                )
                model_name = gr.Dropdown(
                    choices=models if models else ["llama-3.1-8b-instant"],
                    value=models[0] if models else "llama-3.1-8b-instant", label="AI Inference Model"
                )
                run_btn = gr.Button("🚀 Start Production Loop", variant="primary")
            
            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("### 📊 Metrics Feed")
                with gr.Row():
                    m1 = gr.Textbox(label="Jobs Completed", value="0", interactive=False)
                    m2 = gr.Textbox(label="Total Reward", value="0.00", interactive=False)
                    m4 = gr.Textbox(label="Official Score", value="0.000", interactive=False)
                    m3 = gr.Textbox(label="Status", value="Ready", interactive=False)
                
                gr.Markdown("### 🧠 Decision & Reasoning Log")
                output_area = gr.Markdown(
                    value="*Production idle. Select a task and click run to begin.*",
                    container=True
                )

        def runner(tid, smode, mname):
            for update in run_inference_generator(tid, smode, mname):
                # Map the structured update to the multiple UI outputs
                yield (
                    update["logs"],
                    str(update["completed"]),
                    f"{update['total_reward']:.2f}",
                    update["status"],
                    f"{update['score']:.3f}"
                )

        run_btn.click(
            fn=runner,
            inputs=[task_id, strategy_mode, model_name],
            outputs=[output_area, m1, m2, m3, m4]
        )

    return demo

demo = create_ui()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
