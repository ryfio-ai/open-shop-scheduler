import gradio as gr
import os
from inference import run_inference_generator

def launch_task(task_id):
    logs = ""
    for log_line in run_inference_generator(task_id):
        logs += log_line + "\n"
        yield logs

with gr.Blocks(title="OpenShop Scheduler Demo") as demo:
    gr.Markdown("# 🏭 OpenShop Scheduler Dashboard")
    gr.Markdown("Watch the AI agent solve manufacturing scheduling tasks in real-time.")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["easy_single_machine", "medium_parallel_changeover", "hard_dynamic_arrivals"],
            value="easy_single_machine",
            label="Select Task"
        )
        run_btn = gr.Button("🚀 Run Agent", variant="primary")
    
    with gr.Row():
        log_output = gr.Textbox(
            label="Inference Logs (Standard Format)",
            lines=20,
            max_lines=30,
            interactive=False
        )
    
    run_btn.click(
        fn=launch_task,
        inputs=task_dropdown,
        outputs=log_output
    )
    
    gr.Markdown("### 🛠️ Environment Variables Configured")
    gr.Markdown(f"- **API_BASE_URL**: `{os.getenv('API_BASE_URL', 'Default')}`")
    gr.Markdown(f"- **MODEL_NAME**: `{os.getenv('MODEL_NAME', 'Default')}`")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
