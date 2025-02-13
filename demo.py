
import os
import time
import base64
import torch
import wave
import numpy as np
from PIL import Image
import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
import torchaudio
import whisper
from openomni.constants import SPEECH_TOKEN_INDEX, IMAGE_TOKEN_INDEX
from openomni.conversation import SeparatorStyle
from openomni.mm_utils import process_images
from openomni.model.builder import load_pretrained_qwen_model
from openomni.flow_inference import AudioDecoder
from openomni.utils import disable_torch_init

# Constants
TEMP_FILES_PATH = "./assets"
if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)

# Global Variables
last_input_text = ""
last_input_audio = None
last_upload_image = None
temp_files = []

# Initialize Model and Dependencies
disable_torch_init()

model_path = "checkpoints/Tongyi-ConvAI/OpenOmni/qwen2"
voice_config_path = "./cosyvoice/vocab_16K.yaml"
flow_ckpt_path = "checkpoints/THUDM/glm-4-voice-decoder/flow.pt"
hift_ckpt_path = "checkpoints/THUDM/glm-4-voice-decoder/hift.pt"

assert all([model_path, voice_config_path, flow_ckpt_path, hift_ckpt_path]), "Model paths must be set."

audio_decoder = AudioDecoder(
    config_path=voice_config_path,
    flow_ckpt_path=flow_ckpt_path,
    hift_ckpt_path=hift_ckpt_path,
    device="cuda"
)

model_path = os.path.expanduser(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, None)
tokenizer.add_tokens(["<image>"], special_tokens=True)
tokenizer.add_tokens(["<speech>"], special_tokens=True)

image_token_index = tokenizer.convert_tokens_to_ids("<image>")
speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")

# Utility Functions
def extract_message_content(message, key):
    """
    Safely extract content from a message object (dict or MultimodalMessage).
    """
    try:
        if isinstance(message, dict):
            return message.get(key, "")
        elif hasattr(message, key):
            return getattr(message, key)
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")
    except Exception as e:
        print(f"[ERROR] Failed to extract '{key}' from message: {e}")
        return ""


def save_audio(audio, filename):
    """
    Save audio data to a WAV file.
    """
    try:
        filename = os.path.join(TEMP_FILES_PATH, filename)
        torchaudio.save(filename, audio[1], audio[0], format="wav")
        temp_files.append(filename)
        return filename
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None


def save_image(image_array, filename):
    """
    Save an image array to a file.
    """
    try:
        filename = os.path.join(TEMP_FILES_PATH, filename)
        image = Image.fromarray(image_array)
        image.save(filename)
        temp_files.append(filename)
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def clean_temp_files():
    """
    Remove all temporary files.
    """
    for file in temp_files:
        try:
            print(f"Removing {file}")
            os.remove(file)
        except Exception as e:
            print(f"Error removing file {file}: {e}")
    temp_files.clear()


def get_user_msg(input_text, input_audio, upload_image):
    """
    Generate a user message dictionary from inputs.
    """
    user_msg = {"text": input_text, "files": []}

    if input_audio is not None:
        audio_file = save_audio(input_audio, f"audio_{time.time()}.wav")
        if audio_file:
            user_msg["files"].append({"file": FileData(path=audio_file)})

    if upload_image is not None:
        image_file = save_image(upload_image, f"image_{time.time()}.jpg")
        if image_file:
            user_msg["files"].append({"file": FileData(path=image_file)})

    return user_msg


def get_bot_msg(text_response, audio_response, fs):
    """
    Generate a bot message dictionary from outputs.
    """
    bot_msg = {"text": text_response, "files": []}

    if audio_response is not None:
        audio_file = save_audio((fs, audio_response), f"resp_audio_{time.time()}.wav")
        if audio_file:
            bot_msg["files"].append({"file": FileData(path=audio_file)})

    return bot_msg

def preprocess_history(history):
    """
    Preprocess the conversation history to extract prompts.
    """
    prompt = []
    images = []
    speechs = []

    for usr, bot in history:
        usr_turn = {
            'role': 'user',
            'content': extract_message_content(usr, 'text')  # 使用工具函数提取内容
        }
        bot_turn = {
            'role': 'assistant',
            'content': extract_message_content(bot, 'text')  # 使用工具函数提取内容
        }

        # 提取文件信息
        usr_files = extract_message_content(usr, 'files')
        if isinstance(usr_files, list):
            for file_data in usr_files:
                # 修复对 FileMessage 对象的访问
                if hasattr(file_data, 'file') and isinstance(file_data.file, FileData):
                    file_path = file_data.file.path
                    if file_path.endswith(('.jpg', '.png')):
                        images.append(file_path)
                    elif file_path.endswith('.wav'):
                        speechs.append(file_path)
                        usr_turn['content'] = "<speech>\n Please answer the questions in the user's input speech"

        prompt.append(usr_turn)
        prompt.append(bot_turn)

    return prompt, images, speechs


def chat_response(input_text, input_audio, upload_image, temperature, top_p, max_output_tokens, history):
    """
    Process user input and generate a response.
    """
    global last_input_text, last_input_audio, last_upload_image
    last_input_text = input_text
    last_input_audio = input_audio
    last_upload_image = upload_image

    usr_msg = get_user_msg(input_text, input_audio, upload_image)
    prompt, images, speechs = preprocess_history(history)

    # Append current user message
    usr_turn = {"role": "user", "content": usr_msg["text"]}
    usr_files = usr_msg.get('files', [])
    for file_data in usr_files:
        if isinstance(file_data['file'], FileData):
            file_path = file_data['file'].path
            if file_path.endswith(('.jpg', '.png')):
                images.append(file_path)
            elif file_path.endswith('.wav'):
                speechs.append(file_path)
                usr_turn['content'] = "<speech>\n Please answer the questions in the user's input speech"
    prompt.append(usr_turn)

    # 如果有图片，添加图片标识
    if images:
        prompt[0]['content'] = '<image>\n' + prompt[0]['content']
        image_file = images[-1]

    # 如果没有语音输入，使用默认语音文件
    if not speechs:
        speechs = ["./assets/question.wav"]

    # 生成模型输入
    system_message = (
        "You are a helpful language, vision, and speech assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language or speech."
    )
    input_id = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}] + prompt,
        add_generation_prompt=True
    )
    # 替换特殊 token
    input_id = [
        IMAGE_TOKEN_INDEX if token == image_token_index else SPEECH_TOKEN_INDEX if token == speech_token_index else token
        for token in input_id
    ]
    input_ids = torch.tensor([input_id], dtype=torch.long).to(device="cuda")

    # 处理图片输入
    image_tensor = None
    image_sizes = []
    if images:
        image = Image.open(image_file).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_sizes = [image.size]  # 获取图片尺寸

    # 处理语音输入
    speech_tensors = []
    speech_lengths = []
    for speech_file in speechs:
        speech = whisper.load_audio(speech_file)
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_tensors.append(speech_tensor)
        speech_lengths.append(speech_tensor.size(0))  # 获取语音长度

    speech_lengths = torch.LongTensor(speech_lengths).to(device=torch.cuda.current_device(), non_blocking=True)
    if speech_tensors:
        speech_tensors = torch.stack(speech_tensors).to(dtype=torch.float16, device="cuda")

    # 调用模型生成输出
    with torch.inference_mode():
        # 记录起始时间
        time1 = time.time()

        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=image_sizes,  # 传递图片尺寸
            speech=speech_tensors,
            speech_lengths=speech_lengths,  # 传递语音长度
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,  # 默认使用单束搜索
            max_new_tokens=max_output_tokens,
            use_cache=True,  # 启用缓存
            pad_token_id=tokenizer.pad_token_id,
            streaming_unit_gen=False,  # 禁用流式生成
            faster_infer=False  # 禁用快速推理
        )

        # 记录结束时间
        time2 = time.time()
        print(f"Model generation time: {time2 - time1:.2f} seconds")

        # 解码生成的输出
        output_ids, output_units = outputs
        text_response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # 解码语音输出
        tts_speechs = []
        if output_units is not None:
            # 将 output_units 转换为整数 token 列表
            audio_tokens = [int(x) for x in output_units.split(' ')]
            tts_token = torch.tensor(audio_tokens, device='cuda').unsqueeze(0)

            # 使用 audio_decoder 解码语音特征
            tts_speech = audio_decoder.offline_inference(tts_token)
            tts_speechs.append(tts_speech.squeeze())

        # 合并解码后的语音
        if tts_speechs:
            tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        else:
            tts_speech = None

    # 生成机器人回复消息
    bot_msg = get_bot_msg(text_response, tts_speech.unsqueeze(0) if tts_speech is not None else None, 22050)

    # 更新历史记录
    history.append([usr_msg, bot_msg])
    return history, history, "", None, None

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with open("./assets/logo.png", "rb") as image_file:
            logo = base64.b64encode(image_file.read()).decode()
            gr.Markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: center;">
                    <img src="data:image/jpeg;base64,{logo}" alt="Logo" style="width: 100px; height: 100px; margin-right: 20px;">
                    <h1 style="margin: 0;">OpenOmni: A Fully Open-Source Omni Large Language Model with Real-time Self-Aware Emotional Speech Synthesis</h1>
                </div>
                """
            )

    with gr.Row():
        with gr.Column(scale=1):
            upload_image = gr.Image(height=224, width=224, label="Image")
            temperature = gr.Slider(0.1, 1.0, step=0.1, value=0.5, label="Temperature", interactive=True)
            top_p = gr.Slider(0.1, 1.0, step=0.1, value=0.5, label="Top P", interactive=True)
            max_output_tokens = gr.Slider(512, 16384, step=256, value=2048, label="Max Output Tokens", interactive=True)

        with gr.Column(scale=2):
            chatbot = MultimodalChatbot(height=400)
            input_text = gr.Textbox(lines=5, label="Input Text")
            input_audio = gr.Audio(label="Audio")
            with gr.Row():
                chat = gr.Button("Chat")
                regenerate = gr.Button("Regenerate")
                clear = gr.Button("Clear")

    chat.click(
        fn=chat_response,
        inputs=[input_text, input_audio, upload_image, temperature, top_p, max_output_tokens, chatbot],
        outputs=[chatbot, chatbot, input_text, input_audio, upload_image],
    )

    clear.click(
        fn=clean_temp_files,
        inputs=[],
        outputs=[chatbot, input_text, input_audio, upload_image],
    )

demo.launch()
