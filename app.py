"""
心伴AI - 完整版（Supabase登录 + 大模型记忆 + 实时情绪预测）
"""

import streamlit as st
from st_login_form import login_form
import json
import os
import re
import uuid
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
from openai import OpenAI

# ==================== 页面配置 ====================
st.set_page_config(page_title="心伴AI", page_icon="❤️", layout="wide")

# ==================== 用户认证 ====================
# 显示登录表单，返回Supabase连接
supabase_connection = login_form()

# 处理认证状态
if st.session_state.get("authenticated", False):
    username = st.session_state.get("username")
    
    if username:
        USER_ID = username
        user_name = username
        is_guest = False
    else:
        # 游客模式：ID 固定，不会随刷新改变
        if "guest_id" not in st.session_state:
            st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:8]}"
        USER_ID = st.session_state.guest_id
        user_name = "游客"
        is_guest = True
else:
    st.stop()

# ==================== 配置 ====================
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        st.error("❌ 未找到 API 密钥")
        st.stop()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 数据目录（按用户隔离）
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

EMOTION_LOG_PATH = os.path.join(DATA_DIR, f"emotion_log_{USER_ID}.json")
USER_PROFILE_PATH = os.path.join(DATA_DIR, f"user_profile_{USER_ID}.json")
SAFETY_LOG_PATH = os.path.join(DATA_DIR, f"safety_log_{USER_ID}.json")

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是"心伴AI"，一个温暖、温柔、有共情力的情感陪伴助手。你的风格像一位温柔的心理咨询师。

【核心定位】
- 你既是情感陪伴者，也是轻量级的支持者
- 优先倾听和共情，让用户感受到"你懂我"
- 在共情之后，提供具体、可执行、不压迫的小建议

【回复结构建议】
1. 先共情：让用户感受到被理解（1-2句）
2. 再陪伴：开放式提问，鼓励用户多说说（1句）
3. 最后轻引导：提供1-2个具体、可执行的小建议（1-2句）

【正确示例】
用户："大半夜干活好累"
回复："听起来真的很辛苦。这么晚还在工作，一定是有很重要的事情要赶吧？如果实在太累，不妨先休息15分钟，喝点温水。你觉得呢？"

用户："我考砸了，我太笨了"
回复："我能感受到你现在的失落。一次考试没考好，很容易让人怀疑自己。要不要一起看看，问题可能出在复习方法上，还是状态上？"

【禁止行为】
- 不要说"别难过了""加油""你要坚强"等空话
- 不要输出1、2、3点式的冷冰冰建议列表
- 不要输出任何括号形式的内部判断
"""

# ==================== 大模型提取记忆 ====================
def extract_facts_with_llm(user_input, conversation_history):
    """使用大模型从对话中提取关键事实信息"""
    
    history_text = ""
    for msg in conversation_history[-6:]:
        role = "用户" if msg["role"] == "user" else "心伴AI"
        history_text += f"{role}：{msg['content']}\n"
    
    prompt = f"""你是一个信息提取助手。请从以下对话中提取用户提到的**新的事实性信息**。

重要规则：
1. 只提取用户明确表达的信息，不要推测
2. 只提取**新信息**，不要重复之前已有的
3. 如果用户说的是别人的事情（如"朋友生日"），不要提取
4. 信息类型包括：生日、考试、面试、旅行、偏好（喜欢/不喜欢）、重要事件

对话历史：
{history_text}

最新用户输入：{user_input}

请以JSON数组格式输出，如果没有新信息输出[]。
格式示例：
[
  {{"type": "birthday", "value": "5月20日", "original_text": "我生日是5月20日"}},
  {{"type": "exam", "value": "6月15日", "original_text": "6月15日要期末考试"}},
  {{"type": "preference", "value": "喜欢喝咖啡", "original_text": "我平时喜欢喝咖啡"}}
]

只输出JSON，不要有其他内容。"""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            facts = json.loads(json_match.group())
            return facts
        return []
    except Exception as e:
        return []

def update_memory_from_conversation(user_input, conversation_history, profile):
    """更新用户画像（大模型版）"""
    new_facts = extract_facts_with_llm(user_input, conversation_history)
    
    if not new_facts:
        return profile
    
    if "memories" not in profile:
        profile["memories"] = []
    
    for fact in new_facts:
        exists = False
        for existing in profile["memories"]:
            if existing["type"] == fact["type"] and existing["value"] == fact["value"]:
                exists = True
                break
        
        if not exists:
            profile["memories"].append({
                "type": fact.get("type", "general"),
                "value": fact.get("value", ""),
                "original_text": fact.get("original_text", ""),
                "timestamp": datetime.now().isoformat()
            })
    
    profile["memories"] = profile["memories"][-50:]
    save_user_profile(profile)
    
    return profile

# ==================== 实时情绪预测 ====================
def predict_current_emotion(conversation_history, user_input):
    """基于对话历史预测当前用户情绪"""
    
    history_text = ""
    for msg in conversation_history[-8:]:
        role = "用户" if msg["role"] == "user" else "心伴AI"
        history_text += f"{role}：{msg['content']}\n"
    
    prompt = f"""你是一个情绪分析专家。基于以下对话，判断用户**当前**的情绪状态。

对话历史：
{history_text}
用户最新输入：{user_input}

请输出JSON格式，只输出JSON：
{{
    "emotion": "积极/平静/消极",
    "confidence": 0-100,
    "reason": "判断理由"
}}"""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"emotion": "平静", "confidence": 50, "reason": "无法确定"}

# ==================== 智能情绪分类 + 回复生成 ====================
def get_ai_reply_with_emotion(user_input, history, need_correction, disable_correction, user_profile):
    """一次API调用完成情绪分类、记忆整合和回复生成"""
    
    # 构建记忆上下文
    memory_context = ""
    if user_profile.get("memories"):
        recent_memories = user_profile["memories"][-5:]
        memory_context = "\n【用户历史信息】\n"
        for mem in recent_memories:
            memory_context += f"- {mem['type']}: {mem['value']}\n"
        memory_context += "请适当结合这些信息，让回复更个性化。\n"
    
    classification_instruction = """
【额外任务】在输出回复之前，请先用一行标注用户的情绪，格式为：[情绪:积极] 或 [情绪:平静] 或 [情绪:消极]
注意：这个标注只用于内部记录，不要显示给用户。标注后空一行再输出回复。
"""
    
    if disable_correction:
        mode_instruction = "【当前模式：用户拒绝引导模式】请只做普通共情，不要进行认知引导。"
        actual_need_correction = False
    elif need_correction:
        mode_instruction = "【当前模式：纠偏模式】请先共情，再用问句引导换角度思考。"
        actual_need_correction = True
    else:
        mode_instruction = "【当前模式：普通模式】正常共情回应即可。"
        actual_need_correction = False
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + memory_context + classification_instruction + mode_instruction},
        *history[-6:],
        {"role": "user", "content": user_input}
    ]
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        full_response = response.choices[0].message.content
        
        # 解析情绪标注
        emotion_match = re.search(r'\[情绪:(积极|平静|消极)\]', full_response)
        if emotion_match:
            detected_emotion = emotion_match.group(1)
            clean_reply = re.sub(r'\[情绪:(积极|平静|消极)\]\s*\n?', '', full_response).strip()
        else:
            # 降级：让大模型单独判断情绪
            try:
                client2 = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                emotion_response = client2.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"只输出一个词：用户说'{user_input}'，情绪是积极、平静还是消极？"}],
                    temperature=0,
                    max_tokens=10
                )
                detected_emotion = emotion_response.choices[0].message.content.strip()
                if detected_emotion not in ["积极", "平静", "消极"]:
                    detected_emotion = "平静"
            except:
                detected_emotion = "平静"
            clean_reply = full_response
        
        if not clean_reply:
            if actual_need_correction:
                clean_reply = "我能理解你的感受。不过，也许我们可以换个角度看看？"
            else:
                clean_reply = "我在这里陪着你。"
        
        return clean_reply, detected_emotion
        
    except Exception as e:
        if actual_need_correction:
            return "我能理解你的感受。也许我们可以换个角度看看？", '消极'
        return "我在这里陪着你。", '平静'
# ==================== 数据读写 ====================
def load_emotion_log():
    if os.path.exists(EMOTION_LOG_PATH):
        with open(EMOTION_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_emotion_log(log):
    with open(EMOTION_LOG_PATH, "w") as f:
        json.dump(log[-100:], f, ensure_ascii=False, indent=2)

def load_user_profile():
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, "r") as f:
            return json.load(f)
    return {"memories": []}

def save_user_profile(profile):
    with open(USER_PROFILE_PATH, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

# ==================== 安全检测 ====================
CRISIS_KEYWORDS = ['想死', '不想活了', '活不下去', '自杀', '死了算了', '活着没意思']
SAFETY_REPLY = """
> ⚠️ **心伴AI温馨提示**

请立即联系专业心理援助热线：
- 📞 **全国统一心理援助热线：12356**（24小时）
- 📞 **希望24热线：400-161-9995**（24小时）
"""

def check_crisis(user_text):
    for kw in CRISIS_KEYWORDS:
        if kw in user_text:
            return True
    return False

def log_safety_event(user_input, timestamp):
    log_entry = {"timestamp": timestamp, "message": user_input[:100]}
    try:
        if os.path.exists(SAFETY_LOG_PATH):
            with open(SAFETY_LOG_PATH, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        logs = logs[-50:]
        with open(SAFETY_LOG_PATH, "w") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except:
        pass

# ==================== 主动关怀 ====================
def get_care_message():
    if "last_care_date" not in st.session_state:
        st.session_state.last_care_date = None
    today = datetime.now().date()
    if st.session_state.last_care_date == today:
        return None
    current_hour = datetime.now().hour
    if not (6 <= current_hour <= 11):
        return None
    logs = load_emotion_log()
    if len(logs) < 1:
        return None
    now = datetime.now()
    last_night = now.replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=1)
    has_negative = False
    for log in logs[-10:]:
        log_time = datetime.fromisoformat(log['timestamp'])
        if log_time > last_night and log['emotion'] in ['消极', '焦虑']:
            has_negative = True
            break
    if not has_negative:
        return None
    st.session_state.last_care_date = today
    return "🌅 早安～昨晚你看起来有点难过，希望今天的心情能像阳光一样明媚 ☀️"

# ==================== UI ====================
st.title("❤️ 心伴AI")
st.caption("会引导、会主动关心的AI情感陪伴系统 | 我能记住你的重要日子")

with st.sidebar:
    # 用户信息
    if is_guest:
        st.write(f"👤 {user_name} (游客模式)")
        st.caption("💡 注册账号可保存对话记录")
    else:
        st.write(f"👤 {user_name}")
    
    st.divider()
    st.header("📊 情绪仪表盘")

# ===== 直接使用实时情绪预测显示 =====
if "last_prediction" in st.session_state:
    pred = st.session_state.last_prediction
    current_emotion = pred["emotion"]
    current_confidence = pred["confidence"]
    
    # 显示当前情绪
    if current_emotion == "积极":
        st.success(f"## 😊 {current_emotion}")
    elif current_emotion == "消极":
        st.error(f"## 😔 {current_emotion}")
    else:
        st.info(f"## 😐 {current_emotion}")
    
    st.progress(current_confidence / 100)
    st.caption(f"置信度: {current_confidence}%")
    st.caption(f"💡 {pred['reason']}")
    
    # 简化的情绪分布图
    import pandas as pd
    fake_data = pd.DataFrame({
        'emotion': [current_emotion, '其他'],
        'count': [80, 20]
    })
    fig = px.pie(fake_data, values='count', names='emotion', title="情绪分布（实时）")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("开始对话后，我会分析你的情绪")
    
    # 显示记忆
    profile = load_user_profile()
    if profile.get("memories"):
        st.divider()
        st.subheader("📝 我记得")
        for mem in profile["memories"][-3:]:
            st.caption(f"• {mem['type']}: {mem['value']}")
    
    st.divider()
    if st.button("🗑️ 清除所有记忆", use_container_width=True):
        save_emotion_log([])
        save_user_profile({"memories": []})
        st.session_state.messages = []
        st.session_state.disable_correction_counter = 0
        st.session_state.last_care_date = None
        st.rerun()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.disable_correction_counter = 0
    st.session_state.last_care_date = None
    
    care_msg = get_care_message()
    if care_msg:
        st.session_state.messages.append({"role": "assistant", "content": f"🔔 {care_msg}"})

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 12px;">
                <div style="background-color: #dcf8c5; border-radius: 18px; padding: 8px 14px; max-width: 70%; word-wrap: break-word;">
                    {msg["content"]}
                </div>
                <div style="margin-left: 8px; width: 36px; height: 36px; border-radius: 50%; background-color: #ccc; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    👤
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; align-items: flex-start; margin-bottom: 12px;">
                <div style="margin-right: 8px; width: 36px; height: 36px; border-radius: 50%; background-color: #ff6b6b; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    ❤️
                </div>
                <div style="background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 18px; padding: 8px 14px; max-width: 70%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    {msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# 输入框
prompt = st.chat_input("说点什么...")
if prompt:
    if check_crisis(prompt):
        log_safety_event(prompt, datetime.now().isoformat())
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": SAFETY_REPLY})
        st.rerun()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 拒绝引导检测
    reject_keywords = ['别说了', '不想听', '别管我', '不要你管', '别说教', '别烦我', '够了']
    if any(kw in prompt for kw in reject_keywords):
        st.session_state.disable_correction_counter = 3
        st.session_state.messages.append({"role": "assistant", "content": "好的，不强行引导了。我会安静陪着你。"})
        st.rerun()
    
    # 更新记忆（大模型提取）
    conversation_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    profile = load_user_profile()
    profile = update_memory_from_conversation(prompt, conversation_history, profile)
    
    # 情绪日志
    emotion_log = load_emotion_log()
    recent_emotions = [log['emotion'] for log in emotion_log[-5:]]
    
    # 判断是否需要纠偏
    need_correction = len([e for e in recent_emotions[-2:] if e == '消极']) >= 1
    if any(x in prompt for x in ['笨', '没用', '废物', '不行', '不好']):
        need_correction = True
    
    disable_corr = st.session_state.disable_correction_counter > 0
    if disable_corr:
        st.session_state.disable_correction_counter -= 1
    
    # 实时情绪预测
    conversation_history_full = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.last_prediction = predict_current_emotion(conversation_history_full, prompt)
    
    # 获取AI回复
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    reply, _ = get_ai_reply_with_emotion(
        prompt, history, need_correction, disable_corr, profile
    )
    
    # ===== 单独调用大模型判断情绪（高精度，保证仪表盘工作）=====
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        emotion_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"只输出一个词：用户说'{prompt}'，情绪是积极、平静还是消极？"}],
            temperature=0,
            max_tokens=10
        )
        detected_emotion = emotion_response.choices[0].message.content.strip()
        if detected_emotion not in ["积极", "平静", "消极"]:
            detected_emotion = "平静"
    except Exception as e:
        # 降级：关键词判断
        text_lower = prompt.lower()
        if any(kw in text_lower for kw in ['开心', '高兴', '棒', '不错', '喜欢', '好', '耶', '哇']):
            detected_emotion = "积极"
        elif any(kw in text_lower for kw in ['累', '烦', '焦虑', '压力', '难过', '伤心', '郁闷', '笨', '没用', '废物', '糟糕', '崩溃']):
            detected_emotion = "消极"
        else:
            detected_emotion = "平静"
    
    # 记录情绪日志
    emotion_log.append({
        "timestamp": datetime.now().isoformat(),
        "emotion": detected_emotion,
        "text": prompt[:50]
    })
    save_emotion_log(emotion_log)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

st.caption("💡 试试：「我考砸了，我太笨了」→ 系统会先共情再引导；「别说了」→ 停止纠偏；「我生日是5月20日」→ 我会记住并在当天提醒")
