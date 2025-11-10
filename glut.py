import os
import gradio as gr
import base64
import requests
import time
import json
import random
from io import BytesIO
import argparse
from openai import OpenAI
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()

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

TTS_MODEL_CHOICES = [
    "IndexTeam/IndexTTS-2"
]

TTS_VOICE_CHOICES = [
    "jack_cheng",
    "sales_voice",
    "crystla_liu",
    "stephen_chow",
    "xiaoyueyue",
    "mkas",
    "entertain",
    "novel",
    "movie"
]


def image_to_base64(image):
    """
    将PIL图像转换为base64字符串
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def submit_task(api_key, first_frame_image, last_frame_image, prompt, size, duration, model, negative_prompt=None, seed=None, enable_prompt_expansion=False, audio_url=None):
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
        if "pro" in model:
            # 对于pro版本，使用resolution参数
            resolution_map = {
                "720x1280": "720P",
                "1280x720": "720P",
                "1024x1792": "1080P",
                "1792x1024": "1080P"
            }
            resolution = resolution_map.get(size, "720P")
            payload["parameters"] = {
                "resolution": resolution,
                "duration": duration
            }
        else:
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
        # 添加尾帧图片（如果提供）
        if last_frame_image is not None:
            last_frame_url = image_to_base64(last_frame_image)
            payload["input"]["last_frame_url"] = last_frame_url
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        
        # 设置分辨率参数 (仅支持 720P 和 480P)
        resolution_map = {
            "720x1280": "720P",
            "1280x720": "720P",
            "832x480": "480P",
            "480x832": "480P"
        }
        resolution = resolution_map.get(size, "720P")
        payload["parameters"] = {
            "resolution": resolution
        }
        if seed is not None:
            # 如果种子小于0，在允许范围内生成随机种子
            if seed < 0:
                if model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
                    # 2.2版本种子范围: [0, 2147483647]
                    seed = random.randint(0, 2147483647)
                else:
                    # 2.5版本支持-1表示随机，但也可以生成其他随机数
                    seed = random.randint(0, 2147483647)
            payload["parameters"]["seed"] = seed
    elif model == "Wan-AI/Wan2.2-T2V":
        # 处理 Wan-AI/Wan2.2-T2V 模型
        if not prompt:
            raise Exception("Wan-AI/Wan2.2-T2V model requires a prompt")
            
        payload["input"] = {
            "prompt": prompt
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        
        # 设置参数
        payload["parameters"] = {
            "size": size,
            "resolution": "720P" if "1280" in size or "720" in size else "480P"
        }
        if seed is not None:
            # 如果种子小于0，在允许范围内生成随机种子
            if seed < 0:
                if model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
                    # 2.2版本种子范围: [0, 2147483647]
                    seed = random.randint(0, 2147483647)
                else:
                    # 2.5版本支持-1表示随机，但也可以生成其他随机数
                    seed = random.randint(0, 2147483647)
            payload["parameters"]["seed"] = seed
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
        # 添加尾帧图片（如果提供）
        if last_frame_image is not None:
            last_frame_url = image_to_base64(last_frame_image)
            payload["input"]["last_frame_url"] = last_frame_url
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        # 添加音频 URL（如果提供）
        if audio_url:
            payload["input"]["audio_url"] = audio_url
        
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
        if enable_prompt_expansion:
            payload["parameters"]["prompt_extend"] = enable_prompt_expansion
        if seed is not None:
            # 如果种子小于0，在允许范围内生成随机种子
            if seed < 0:
                if model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
                    # 2.2版本种子范围: [0, 2147483647]
                    seed = random.randint(0, 2147483647)
                else:
                    # 2.5版本支持-1表示随机，但也可以生成其他随机数
                    seed = random.randint(0, 2147483647)
            payload["parameters"]["seed"] = seed
    elif model == "Wan-AI/Wan2.5-T2V":
        # 处理 Wan-AI/Wan2.5-T2V 模型
        if not prompt:
            raise Exception("Wan-AI/Wan2.5-T2V model requires a prompt")
            
        payload["input"] = {
            "prompt": prompt
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        # 添加音频 URL（如果提供）
        if audio_url:
            payload["input"]["audio_url"] = audio_url
        
        # 设置参数
        payload["parameters"] = {
            "size": size,
            "duration": duration
        }
        if enable_prompt_expansion:
            payload["parameters"]["prompt_extend"] = enable_prompt_expansion
        if seed is not None:
            # 如果种子小于0，在允许范围内生成随机种子
            if seed < 0:
                if model in ["Wan-AI/Wan2.2-I2V", "Wan-AI/Wan2.2-T2V"]:
                    # 2.2版本种子范围: [0, 2147483647]
                    seed = random.randint(0, 2147483647)
                else:
                    # 2.5版本支持-1表示随机，但也可以生成其他随机数
                    seed = random.randint(0, 2147483647)
            payload["parameters"]["seed"] = seed
    
    response = requests.post(f"{BASE_URL}/tasks/submit",
                            headers=headers,
                            data=json.dumps(payload),
                            timeout=30)
    
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


def generate_speech(api_key, model, text, voice):
    """
    生成语音 - 使用 OpenAI SDK 调用 Modelverse TTS API
    """
    try:
        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.modelverse.cn/v1/",
        )
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 生成语音文件路径
        speech_file_path = Path(__file__).parent / f"outputs/{timestamp}_{voice}.wav"
        
        # 调用 TTS API
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
        ) as response:
            response.stream_to_file(speech_file_path)
        
        return str(speech_file_path)
        
    except Exception as e:
        raise Exception(f"❌ 语音生成失败: {str(e)}")


def generate_audio(api_key, model, text, voice):
    """
    主函数：生成语音
    """
    try:
        # 检查输入参数
        if not api_key:
            return "❌ 请输入API KEY", None
        if not text:
            return "❌ 请输入要转换的文本", None
        if len(text) > 600:
            return "❌ 文本长度不能超过600字符", None
        
        # 使用 OpenAI SDK 生成语音，返回文件路径
        audio_file_path = generate_speech(api_key, model, text, voice)
        
        return f"✅ 语音生成完毕", audio_file_path
        
    except Exception as e:
        return f"❌ 发生错误: {str(e)}", None


def generate_video(api_key, first_frame_image, last_frame_image, prompt, size, duration, model, negative_prompt=None, seed=None, enable_prompt_expansion=False, audio_url=None, video_state=None):
    """
    主函数：上传图片，提交任务，轮询状态并下载结果
    """
    try:
        # 提交任务
        task_id = submit_task(api_key, first_frame_image, last_frame_image, prompt, size, duration, model, negative_prompt, seed, enable_prompt_expansion, audio_url)
        # 确保 video_state 是列表
        current_videos = video_state if video_state is not None else []
        yield f"任务已提交，任务ID: {task_id}", current_videos, current_videos
        
        # 轮询任务状态
        while True:
            time.sleep(5)  # 每5秒查询一次
            
            status_result = check_task_status(api_key, task_id)
            task_status = status_result["output"]["task_status"]
            
            if task_status == "Success":
                video_urls = status_result["output"]["urls"]
                # 将新视频添加到现有视频列表中
                updated_videos = current_videos + video_urls
                yield f"✅ 视频生成完毕", updated_videos, updated_videos
                break
            elif task_status == "Failure":
                error_msg = status_result["output"].get("error_message", "未知错误")
                yield f"❌ 任务失败: {error_msg}", current_videos, current_videos
                break
            else:
                yield f"任务状态: {task_status}...", current_videos, current_videos
                
    except Exception as e:
        current_videos = video_state if video_state is not None else []
        yield f"❌ 发生错误: {str(e)}", current_videos, current_videos


def update_visibility(model):
    # 根据选择的模型更新界面元素的可见性
    if model == "Wan-AI/Wan2.2-I2V":
        # 只有Wan-AI/Wan2.2-I2V支持尾帧
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    elif "image-to-video" in model or model in ["Wan-AI/Wan2.5-I2V"]:
        # 其他i2v模型不支持尾帧
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def update_negative_prompt_visibility(model):
    # 更新负向提示词的可见性
    # 只有Wan-AI模型支持negative_prompt
    return gr.update(visible=model.startswith("Wan-AI"))
    

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
    elif model in ["openai/sora-2/image-to-video-pro", "openai/sora-2/text-to-video-pro"]:
        return gr.update(
            minimum=4,
            maximum=12,
            step=4,
            value=4,
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


def update_prompt_expansion_visibility(model):
    """根据模型类型更新提示词优化的可见性"""
    # Wan-AI 2.5-I2V 和 Wan2.5-T2V 模型都支持提示词优化
    # Wan2.2模型没有这个参数
    return gr.update(visible=model in ["Wan-AI/Wan2.5-I2V", "Wan-AI/Wan2.5-T2V"])


def update_audio_url_visibility(model):
    """根据模型类型更新音频URL的可见性"""
    # 只有Wan-AI 2.5模型支持音频URL
    return gr.update(visible=model in ["Wan-AI/Wan2.5-I2V", "Wan-AI/Wan2.5-T2V"])


def update_seed_visibility(model):
    """根据模型类型更新随机数种子的可见性"""
    # 只有Wan-AI模型支持seed
    return gr.update(visible=model.startswith("Wan-AI"))


with gr.Blocks(title="优云智算 API调用 在线体验", theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")])) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">优云智算 API调用 在线体验</h2>
            </div>
            <div style="text-align: center;">
                使用说明：体验前请先前往 <b><a href="https://www.compshare.cn/?ytag=GPU_YY-SZY_Gradio">优云智算</a></b> 平台注册实名，新用户立得10元赠金。
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 本工具仅提供API调用界面，用户需对生成内容承担全部责任。请确保遵守当地法律法规，不生成任何违法违规内容。
            </div>
            """)
    
    # 创建状态变量来存储已生成的视频
    video_state = gr.State([])
    api_key_input = gr.Textbox(
                label="API KEY（必填）",
                info="(请先去 [优云智算](https://console.compshare.cn/light-gpu/api-keys?ytag=GPU_YY-SZY_Gradio) 创建API KEY)",
                placeholder="请输入您的API KEY...",
                type="password"
            )
    with gr.Tabs():
        with gr.TabItem("视频生成"):
            with gr.Row():
                with gr.Column():
                    model_choice = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="openai/sora-2/image-to-video",
                        label="选择模型"
                    )
                    with gr.Row():
                        first_frame = gr.Image(type="pil", label="首帧图片", visible=True, height=300)
                        last_frame = gr.Image(type="pil", label="尾帧图片（可选）", visible=False, height=300)
                    prompt = gr.Textbox(label="提示词", placeholder="请输入提示词指导视频生成...")
                    negative_prompt = gr.Textbox(label="负面提示词（可选）", placeholder="请输入不希望出现的内容...", visible=False)
                    audio_url = gr.Textbox(label="音频 URL（可选）", placeholder="请输入音频文件 URL（可选）...", visible=False)
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
                    seed = gr.Number(label="种子", value=-1, info="-1表示随机", visible=False)
                    enable_prompt_expansion = gr.Checkbox(label="启用提示词优化", visible=False)
                    submit_btn = gr.Button("🎬 开始生成", variant="primary")
                with gr.Column():
                    status_output = gr.Textbox(label="任务状态", interactive=False)
                    gr.Markdown("视频生成后，请点击下载按钮手动保存。刷新界面会导致视频生成结果丢失。")
                    video_output = gr.Gallery(label="视频生成", columns=2, height=800, object_fit="contain")
                    gr.Markdown("更多使用方法详见[API调用文档](https://www.compshare.cn/docs/modelverse/models/audio_api/ttts/?ytag=GPU_YY-SZY_Gradio)")
        
        with gr.TabItem("音频生成"):
            with gr.Row():
                with gr.Column():
                    tts_model_choice = gr.Dropdown(
                        choices=TTS_MODEL_CHOICES,
                        value="IndexTeam/IndexTTS-2",
                        label="选择语音模型"
                    )
                    voice_choice = gr.Dropdown(
                        choices=TTS_VOICE_CHOICES,
                        value="jack_cheng",
                        label="选择音色"
                    )
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入要转换为语音的文本内容（最大支持600字符）...",
                    )
                    submit_audio_btn = gr.Button("🎵 生成语音", variant="primary")
                with gr.Column():
                    audio_status_output = gr.Textbox(label="生成状态", interactive=False)
                    audio_output = gr.Audio(label="生成的语音", type="filepath", interactive=False, autoplay=True, show_download_button=True)
                    gr.Markdown("更多使用方法详见[API调用文档](https://www.compshare.cn/docs/modelverse/models/audio_api/ttts/?ytag=GPU_YY-SZY_Gradio)")

    model_choice.change(
        fn=lambda model: [
            update_visibility(model)[0],
            update_visibility(model)[1],
            update_visibility(model)[2],
            update_negative_prompt_visibility(model),
            update_resolution_choices(model),
            update_duration_slider(model),
            update_seed_visibility(model),
            update_prompt_expansion_visibility(model),
            update_audio_url_visibility(model)
        ],
        inputs=model_choice,
        outputs=[first_frame, prompt, last_frame, negative_prompt, size, duration, seed, enable_prompt_expansion, audio_url]
    )

    # 视频生成事件处理
    gr.on(
        triggers=[submit_btn.click, prompt.submit],
        fn=generate_video,
        inputs=[api_key_input, first_frame, last_frame, prompt, size, duration, model_choice, negative_prompt, seed, enable_prompt_expansion, audio_url, video_state],
        outputs=[status_output, video_output, video_state]
    )
    
    # 音频生成事件处理
    gr.on(
        triggers=[submit_audio_btn.click, text_input.submit],
        fn=generate_audio,
        inputs=[api_key_input, tts_model_choice, text_input, voice_choice],
        outputs=[audio_status_output, audio_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        mcp_server=args.mcp_server,
        inbrowser=True,
    )