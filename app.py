import gradio as gr
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from inference import run_inference_generator
from envs.shop_scheduler_env.env import ShopSchedulerEnv
from envs.shop_scheduler_env.models import Action

# 1. Initialize the Core Environment for the API
# By default, use the easy task for the automated pings
api_env = ShopSchedulerEnv(task_id="easy_single_machine")

# 2. Create FastAPI App
app = FastAPI(title="OpenShop Scheduler API")

# --- OpenEnv Standard API Endpoints ---

@app.post("/reset")
async def reset(request: Request):
    obs = api_env.reset()
    return JSONResponse(content=obs.model_dump())

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = Action(**data)
    obs, reward_obj, done, info = api_env.step(action)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": reward_obj.value,
        "done": done,
        "info": info
    })

@app.get("/state")
async def state():
    state_data = api_env.state()
    return JSONResponse(content=state_data.model_dump())

# --- Gradio Dashboard UI ---

def create_ui():
    with gr.Blocks(title="OpenShop Scheduler Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏭 OpenShop Scheduler Dashboard")
        gr.Markdown("Watch the AI agent solve manufacturing scheduling tasks in real-time.")
        
        with gr.Row():
            with gr.Column(scale=1):
                task_id = gr.Dropdown(
                    choices=["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"],
                    value="easy_single_machine",
                    label="Select Task"
                )
                run_btn = gr.Button("🚀 Run Agent", variant="primary")
            
            with gr.Column(scale=2):
                output = gr.TextArea(
                    label="Inference Logs (Standard Format)",
                    interactive=False,
                    lines=15
                )

        run_btn.click(
            fn=run_inference_generator,
            inputs=[task_id],
            outputs=[output]
        )
    return demo

# 3. Mount Gradio to FastAPI
# We mount it at "/" so the human-friendly UI is the first thing people see,
# but FastAPI will handle the /reset, /step, /state routes first.
demo = create_ui()
app = gr.mount_gradio_app(app, demo, path="/")

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
