"""
心伴AI - 完整版（适配 Streamlit Cloud 部署）
功能：情感纠偏 + 情绪仪表盘 + 主动关怀模拟
布局：用户消息在右，AI回复在左，带头像
"""

import streamlit as st
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
from openai import OpenAI

# ==================== 页面配置 ====================
st.set_page_config(page_title="心伴AI", page_icon="❤️", layout="wide")

# ==================== 自定义CSS（左右对话布局，显示头像） ====================
st.markdown("""
<style>
    /* 用户消息容器 - 右对齐 */
    [data-testid="stChatMessage"][aria-label="user"] {
        background-color: #dcf8c5;
        border-radius: 18px;
        padding: 10px 16px;
        margin-left: auto;
        margin-right: 0;
        width: fit-content;
        max-width: 80%;
        text-align: left;
        display: flex;
        flex-direction: row-reverse;  /* 头像在右，消息在左 */
        gap: 8px;
    }
    
    /* AI消息容器 - 左对齐 */
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 10px 16px;
        margin-left: 0;
        margin-right: auto;
        width: fit-content;
        max-width: 80%;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: row;
        gap: 8px;
    }
    
    /* 头像样式 */
    [data-testid="stChatMessageAvatar"] {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        object-fit: cover;
        flex-shrink: 0;
    }
    
    /* 用户头像靠右，AI头像靠左 - 通过父容器的 flex-direction 已经处理 */
    
    /* 消息文本区域 */
    [data-testid="stChatMessageContent"] {
        flex: 1;
    }
    
    /* 调整消息之间的间距 */
    .stChatMessage {
        margin-bottom: 12px;
    }
    
    /* 侧边栏宽度调整 */
    [data-testid="stSidebar"] {
        width: 280px;
    }
    
    /* 主内容区域底部留空，避免被输入框遮挡 */
    .main .block-container {
        padding-bottom: 80px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 配置 ====================
# 从 secrets 中安全读取 API 密钥
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    st.error("❌ 未找到 API 密钥，请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
    st.stop()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 数据存储目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
EMOTION_LOG_PATH = os.path.join(DATA_DIR, "emotion_log.json")

# 系统提示词
SYSTEM_PROMPT = """你是"心伴AI"，一个温暖的情感陪伴助手。

【重要】你的回复中不要输出任何内部判断、情绪检测结果、模式标识。直接输出自然、温暖的对话内容即可。

【纠偏模式触发条件】
- 用户连续2轮表现出消极/焦虑情绪
- 用户出现自我贬低表述（如"我真没用""我太笨了"）

【纠偏模式回复格式（必须遵守）】
第一步：共情句（承认用户感受）
第二步：引导句（用问句挑战负面认知）

【正确示例】
用户："我考砸了，我太笨了"
回复："我能理解你现在的失落。但一次考试真的能定义你笨不笨吗？有没有可能是复习方法的问题？"

【禁止使用的回复】
- "别难过了"
- "加油你会更好的"
"""

def load_emotion_log():
    if os.path.exists(EMOTION_LOG_PATH):
        with open(EMOTION_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_emotion_log(log):
    with open(EMOTION_LOG_PATH, "w") as f:
        json.dump(log[-100:], f, ensure_ascii=False, indent=2)

def detect_emotion(user_text, recent_emotions):
    negative_keywords = ['废物', '没用', '太差', '崩溃', '绝望', '焦虑', '烦死了', '好烦', '我太笨', '我真蠢']
    for kw in negative_keywords:
        if kw in user_text:
            need_correction = len([e for e in recent_emotions[-2:] if e in ['消极', '焦虑']]) >= 1
            if '笨' in user_text or '没用' in user_text:
                need_correction = True
            return {'label': '消极', 'need_correction': need_correction}
    return {'label': '平静', 'need_correction': False}

def get_ai_reply(user_input, history, need_correction):
    mode_instruction = "【当前模式：纠偏模式】先共情，再用问句引导换角度思考。" if need_correction else "【当前模式：普通模式】正常共情回应。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + mode_instruction},
        *history[-6:],
        {"role": "user", "content": user_input}
    ]
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.8)
        return response.choices[0].message.content
    except Exception as e:
        return "我能理解你的感受。不过，也许我们可以换个角度看看？你觉得呢？" if need_correction else "我在这里陪着你。"

def check_active_care():
    logs = load_emotion_log()
    if len(logs) < 1:
        return None
    now = datetime.now()
    last_night = now.replace(hour=22, minute=0) - timedelta(days=1)
    for log in logs[-10:]:
        log_time = datetime.fromisoformat(log['timestamp'])
        if log_time > last_night and log['emotion'] in ['消极', '焦虑']:
            return "🌅 早安～昨晚你看起来有点难过，今天天气不错，希望你有好心情！"
    return None

# ==================== UI ====================
st.title("❤️ 心伴AI")
st.caption("会引导、会主动关心的AI情感陪伴系统")

# 侧边栏仪表盘
with st.sidebar:
    st.header("📊 情绪仪表盘")
    emotion_log = load_emotion_log()
    if emotion_log:
        df = pd.DataFrame(emotion_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig1 = px.pie(df, names='emotion', title="情绪分布")
        st.plotly_chart(fig1, use_container_width=True)
        
        # 最近7天趋势
        last7 = df[df['timestamp'] > datetime.now() - timedelta(days=7)]
        if not last7.empty:
            trend = last7.groupby(['date', 'emotion']).size().unstack().fillna(0)
            fig2 = px.line(trend, title="近7天情绪趋势", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无数据，开始对话后会自动记录")
    
    if st.button("🗑️ 清除所有记忆", use_container_width=True):
        save_emotion_log([])
        st.rerun()

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    care_msg = check_active_care()
    if care_msg:
        st.session_state.messages.append({"role": "assistant", "content": f"🔔 {care_msg}"})

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("说点什么..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 情绪识别
    emotion_log = load_emotion_log()
    recent_emotions = [log['emotion'] for log in emotion_log[-5:]]
    emotion_result = detect_emotion(prompt, recent_emotions)
    
    # 记录情绪
    emotion_log.append({
        "timestamp": datetime.now().isoformat(),
        "emotion": emotion_result['label'],
        "text": prompt[:50]
    })
    save_emotion_log(emotion_log)
    
    # 获取AI回复
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    reply = get_ai_reply(prompt, history, emotion_result['need_correction'])
    
    # 显示AI回复
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("💡 试试输入：「我考砸了，我太笨了」或「我好焦虑」→ 系统会先共情再引导")
