import streamlit as st
import os
import time
import json
import google.generativeai as genai
from PIL import Image
import io
from google.cloud import texttospeech

# 配置 Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_tts_audio(text_to_speak):
    """
    使用 Google Cloud Text-to-Speech API 将文本转换为语音
    
    Args:
        text_to_speak: 要转换的英文文本
        
    Returns:
        bytes: 音频数据，如果转换失败则返回 None
    """
    try:
        # 初始化 Text-to-Speech 客户端
        client = texttospeech.TextToSpeechClient()
        
        # 设置要合成的文本
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        
        # 配置语音参数
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",  # 使用女声
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # 配置音频格式
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,  # 稍微放慢语速，使其更清晰
            pitch=0  # 默认音高
        )
        
        # 执行文本到语音的转换
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # 返回音频内容
        return response.audio_content
        
    except Exception as e:
        st.error(f"文本转语音失败: {str(e)}")
        return None

def analyze_image_for_words(image_bytes):
    """
    分析图片中的物体并返回英文和中文名称
    
    Args:
        image_bytes: 图片的字节数据
        
    Returns:
        list: 包含物体信息的字典列表，每个字典包含 'english_name' 和 'chinese_name'
    """
    try:
        # 将字节数据转换为PIL Image对象
        image = Image.open(io.BytesIO(image_bytes))
        
        # 配置模型
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # 构建提示词
        prompt = """Identify the main objects in the provided image. For each object, give its name in English and its translation in Simplified Chinese. 
        Please return the result as a JSON list of objects, where each object has 'english_name' and 'chinese_name' keys. 
        For example: [{'english_name': 'mug', 'chinese_name': '马克杯'}, {'english_name': 'key', 'chinese_name': '钥匙'}]. 
        Only return the JSON list."""
        
        # 生成响应
        response = model.generate_content([prompt, image])
        
        # 尝试解析JSON响应
        try:
            # 提取响应文本并解析JSON
            response_text = response.text.strip()
            # 确保响应文本是有效的JSON字符串
            if response_text.startswith('[') and response_text.endswith(']'):
                objects_list = json.loads(response_text)
                return objects_list
            else:
                st.error("AI响应格式不正确")
                return []
        except json.JSONDecodeError:
            st.error("无法解析AI响应为JSON格式")
            return []
            
    except Exception as e:
        st.error(f"分析图片时发生错误: {str(e)}")
        return []

st.set_page_config(layout="wide")

# 初始化会话状态变量
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'analyzed_objects' not in st.session_state:
    st.session_state.analyzed_objects = []

# 标题
st.title("LingoLens MVP Simulator")

# 上下文捕获部分
st.subheader("Capture Context (Optional)")
img_file_buffer = st.camera_input("Take Photo")
if img_file_buffer is not None:
    st.session_state.current_image = img_file_buffer
    st.image(img_file_buffer, caption="已捕获的图像", width=300)
    
    # 添加分析按钮
    if st.button("Analyze Photo for Words"):
        if st.session_state.current_image is not None:
            with st.spinner('Analyzing image...'):
                # 调用分析函数
                analysis_results = analyze_image_for_words(st.session_state.current_image.getvalue())
                # 存储分析结果到会话状态
                st.session_state.analyzed_objects = analysis_results
                # 触发页面刷新
                st.rerun()
    
    # 创建分析结果显示区域
    object_analysis_area = st.container()
    with object_analysis_area:
        st.markdown("### Analysis Results")
        if st.session_state.analyzed_objects:
            st.markdown("#### 识别到的物体：")
            for obj in st.session_state.analyzed_objects:
                # 为每个物体创建一个容器
                with st.container():
                    # 创建三列布局
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown(f"**{obj['english_name']}**")
                    with col2:
                        st.markdown(f"{obj['chinese_name']}")
                    with col3:
                        # 使用英文名作为按钮的唯一键
                        if st.button("🔊", key=f"pronounce_{obj['english_name']}"):
                            # 显示生成提示
                            with st.spinner(f'正在生成 {obj["english_name"]} 的发音...'):
                                # 获取音频数据
                                audio_bytes = get_tts_audio(obj['english_name'])
                                if audio_bytes:
                                    # 播放音频
                                    st.audio(audio_bytes, format='audio/mp3')
                                else:
                                    st.error(f"无法生成 {obj['english_name']} 的发音")
                # 添加分隔线
                st.markdown("---")
        else:
            st.info("点击 'Analyze Photo for Words' 按钮开始分析图片")

# 主交互区域
st.subheader("Talk to LingoLens AI")

# 聊天历史容器
chat_container = st.container()
with chat_container:
    # 设置聊天历史容器的高度
    st.markdown("""
        <style>
        .chat-container {
            height: 400px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 显示聊天历史
    for message in st.session_state.conversation:
        with st.chat_message(message['role']):
            st.write(message['text'])

# 输入区域
st.subheader("Input")
record_col1, record_col2 = st.columns(2)
with record_col1:
    st.button("🎤 Start Recording")
with record_col2:
    st.button("⏹️ Stop Recording")

# 添加音频播放器
if st.session_state.audio_filename and os.path.exists(st.session_state.audio_filename):
    try:
        with open(st.session_state.audio_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
    except FileNotFoundError:
        st.error("音频文件未找到")
    except Exception as e:
        st.error(f"播放音频时发生错误: {str(e)}")
    finally:
        # 可选：删除临时音频文件
        try:
            os.remove(st.session_state.audio_filename)
        except:
            pass

# 文本输入和发送按钮
user_input = st.text_input("Or type your message here:")
st.button("Send to AI")

# 教练反馈部分
st.subheader("Coach Feedback")
feedback_container = st.container()
with feedback_container:
    # 设置反馈容器的高度
    st.markdown("""
        <style>
        .feedback-container {
            height: 200px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    for feedback in st.session_state.feedback_log:
        st.write(feedback)








