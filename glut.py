import gradio as gr
import base64
import requests
import time
import os
import json
from io import BytesIO
from PIL import Image
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()

# 创建输出目录
os.makedirs("outputs", exist_ok=True)

BASE_URL = "https://api.modelverse.cn/v1"

MODEL_CHOICES = [
    "openai/sora-2/image-to-video-pro",
    "openai/sora-2/image-to-video", 
    "openai/sora-2/text-to-video-pro", 
    "openai/sora-2/text-to-video",
    "Wan-AI/Wan2.2-I2V",
    "Wan-AI/Wan2.2-T2V",
    "Wan-AI/Wan2.5-I2V",
    "Wan-AI/Wan2.5-T2V"
]


def image_to_base64(image):
    """
    将PIL图像转换为base64字符串
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def submit_task(api_key, first_frame_image, prompt, size, duration, model):
    """
    提交图像到视频生成任务
    """
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model
    }
    
    # 根据模型类型构建不同的输入参数
    if "image-to-video" in model:
        if first_frame_image is None:
            raise Exception("Image-to-video models require an input image")
        first_frame_url = image_to_base64(first_frame_image)
        payload["input"] = {
            "first_frame_url": first_frame_url
        }
        if prompt:
            payload["input"]["prompt"] = prompt
            
        # 添加参数部分
        payload["parameters"] = {
            "size": size,
            "duration": duration
        }
    elif "text-to-video" in model:
        if not prompt:
            raise Exception("Text-to-video models require a prompt")
        payload["input"] = {
            "prompt": prompt
        }
        
        # 添加参数部分
        payload["parameters"] = {
            "size": size,
            "duration": duration
        }
    elif model == "Wan-AI/Wan2.2-I2V":
        # 处理 Wan-AI/Wan2.2-I2V 模型
        if first_frame_image is None:
            raise Exception("Wan-AI/Wan2.2-I2V model requires an input image")
        first_frame_url = image_to_base64(first_frame_image)
        if not prompt:
            raise Exception("Wan-AI/Wan2.2-I2V model requires a prompt")
            
        payload["input"] = {
            "first_frame_url": first_frame_url,
            "prompt": prompt
        }
        
        # 设置分辨率参数 (仅支持 720P 和 480P)
        resolution = "720P" if "1280" in size or "720" in size else "480P"
        payload["parameters"] = {
            "resolution": resolution
        }
    elif model == "Wan-AI/Wan2.2-T2V":
        # 处理 Wan-AI/Wan2.2-T2V 模型
        if not prompt:
            raise Exception("Wan-AI/Wan2.2-T2V model requires a prompt")
            
        payload["input"] = {
            "prompt": prompt
        }
        
        # 设置参数
        resolution = "720P" if "1280" in size or "720" in size else "480P"
        payload["parameters"] = {
            "size": size,
            "resolution": resolution
        }
    elif model == "Wan-AI/Wan2.5-I2V":
        # 处理 Wan-AI/Wan2.5-I2V 模型
        if first_frame_image is None:
            raise Exception("Wan-AI/Wan2.5-I2V model requires an input image")
        first_frame_url = image_to_base64(first_frame_image)
        if not prompt:
            raise Exception("Wan-AI/Wan2.5-I2V model requires a prompt")
            
        payload["input"] = {
            "first_frame_url": first_frame_url,
            "prompt": prompt
        }
        
        # 设置分辨率参数 (支持 480p, 720p, 1080p)
        resolution_map = {
            "720x1280": "720p",
            "1280x720": "720p",
            "832x480": "480p",
            "480x832": "480p",
            "1920x1080": "1080p",
            "1080x1920": "1080p"
        }
        resolution = resolution_map.get(size, "720p")
        payload["parameters"] = {
            "resolution": resolution,
            "duration": duration
        }
    elif model == "Wan-AI/Wan2.5-T2V":
        # 处理 Wan-AI/Wan2.5-T2V 模型
        if not prompt:
            raise Exception("Wan-AI/Wan2.5-T2V model requires a prompt")
            
        payload["input"] = {
            "prompt": prompt
        }
        
        # 设置参数
        payload["parameters"] = {
            "size": size,
            "duration": duration
        }
    
    response = requests.post(f"{BASE_URL}/tasks/submit", 
                            headers=headers, 
                            data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["output"]["task_id"]
        return task_id
    else:
        raise Exception(f"❌ 任务提交失败: {response.text}")
    

def check_task_status(api_key, task_id):
    """
    查询任务状态
    """
    headers = {
        "Authorization": api_key
    }
    
    response = requests.get(f"{BASE_URL}/tasks/status?task_id={task_id}", 
                           headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"❌ 查询任务状态失败: {response.text}")


def download_video(url, filename):
    """
    下载视频到本地
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"❌ 视频下载失败: {response.status_code}")


def generate_video(api_key, first_frame_image, prompt, size, duration, model):
    """
    主函数：上传图片，提交任务，轮询状态并下载结果
    """
    try:
        # 提交任务
        task_id = submit_task(api_key, first_frame_image, prompt, size, duration, model)
        yield f"任务已提交，任务ID: {task_id}", None
        
        # 轮询任务状态
        while True:
            time.sleep(5)  # 每5秒查询一次
            
            status_result = check_task_status(api_key, task_id)
            task_status = status_result["output"]["task_status"]
            
            if task_status == "Success":
                video_url = status_result["output"]["urls"][0]
                filename = f"outputs/{task_id}.mp4"
                
                # 下载视频
                downloaded_file = download_video(video_url, filename)
                yield f"✅ 视频生成完毕", downloaded_file
                break
            elif task_status == "Failure":
                error_msg = status_result["output"].get("error_message", "未知错误")
                yield f"❌ 任务失败: {error_msg}", None
                break
            else:
                yield f"任务状态: {task_status}...", None
                
    except Exception as e:
        yield f"❌ 发生错误: {str(e)}", None


def update_visibility(model):
    # 根据选择的模型更新界面元素的可见性
    if "image-to-video" in model or model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.5-I2V"]:
        return gr.update(visible=True), gr.update(label="提示词")
    else:
        return gr.update(visible=False), gr.update(label="提示词")
    

def update_resolution_choices(model):
    """根据模型类型更新可用的分辨率选项"""
    if "pro" in model:
        # pro版本支持所有分辨率
        return gr.update(
            choices=["720x1280", "1280x720", "1024x1792", "1792x1024"],
            value="720x1280",
            visible=True
        )
    elif model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
        # Wan-AI 2.2 模型只支持特定分辨率
        return gr.update(
            choices=["720x1280", "1280x720", "832x480", "480x832"],
            value="720x1280",
            visible=True
        )
    elif model in ["Wan-AI/Wan2.5-I2V", "Wan-AI/Wan2.5-T2V"]:
        # Wan-AI 2.5 模型支持新分辨率选项
        return gr.update(
            choices=["720x1280", "1280x720", "832x480", "480x832", "1920x1080", "1080x1920"],
            value="720x1280",
            visible=True
        )
    else:
        # 非pro版本只支持720x1280和1280x720
        return gr.update(
            choices=["720x1280", "1280x720"],
            value="720x1280",
            visible=True
        )
        

def update_duration_slider(model):
    """根据模型类型更新时长滑块"""
    if model in ["Wan-AI/Wan2.5-I2V", "Wan-AI/Wan2.5-T2V"]:
        return gr.update(
            minimum=5,
            maximum=10,
            step=5,
            value=5,
            visible=True
        )
    elif model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
        return gr.update(
            minimum=5,
            maximum=5,
            step=5,
            value=5,
            visible=True
        )
    else:
        return gr.update(
            minimum=4,
            maximum=12,
            step=4,
            value=4,
            visible=True
        )


with gr.Blocks(title="优云智算 视频生成在线体验", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">优云智算 视频生成在线体验</h2>
            </div>
            <div style="text-align: center;">
                使用说明：体验前请先前往 <b><a href="https://www.compshare.cn/?ytag=GPU_YY-SZY_Gradio">优云智算</a></b> 平台注册实名，新用户立得10元赠金。 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 本工具仅提供API调用界面，用户需对生成内容承担全部责任。请确保遵守当地法律法规，不生成任何违法违规内容。
            </div>
            """)
    
    with gr.Row():
        with gr.Column():
            api_key_input = gr.Textbox(
                label="API KEY", 
                info="(请先去 [优云智算](https://console.compshare.cn/light-gpu/api-keys?ytag=GPU_YY-SZY_Gradio) 创建API KEY)",
                placeholder="请输入您的API KEY...",
                type="password"
            )
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="openai/sora-2/image-to-video",
                label="选择模型"
            )
            first_frame = gr.Image(type="pil", label="首帧图片", visible=True, height=500)
            prompt = gr.Textbox(label="提示词", placeholder="请输入提示词指导视频生成...")
            size = gr.Dropdown(
                choices=["720x1280", "1280x720", "1024x1792", "1792x1024"],
                value="720x1280",
                label="视频尺寸"
            )
            duration = gr.Slider(
                minimum=4,
                maximum=12,
                step=1,
                value=4,
                label="视频时长 (秒)"
            )
            submit_btn = gr.Button("🎬 开始生成", variant="primary")
        
        with gr.Column():
            status_output = gr.Textbox(label="任务状态", interactive=False)
            video_output = gr.Video(label="视频生成", height=800)
            gr.Markdown("更多使用方法详见[API调用文档](https://www.compshare.cn/docs/modelverse/models/video_api/OpenAI-Sora2-I2V)")
    
    model_choice.change(
        fn=lambda model: [
            update_visibility(model)[0],
            update_visibility(model)[1],
            update_resolution_choices(model),
            update_duration_slider(model)
        ],
        inputs=model_choice,
        outputs=[first_frame, prompt, size, duration]
    )

    gr.on(
        triggers=[submit_btn.click, prompt.submit],
        fn=generate_video,
        inputs=[api_key_input, first_frame, prompt, size, duration, model_choice],
        outputs=[status_output, video_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )