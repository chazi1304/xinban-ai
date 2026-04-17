"""
心伴AI - 完整版（含安全检测 + 长期记忆 + 主动提醒 + 智能关怀 + 多用户数据隔离）
功能：情感纠偏、情绪仪表盘、主动关怀（仅早晨、每天一次）、用户拒绝反馈、
      极端危机安全协议、重要日期记忆与提醒、每个用户独立数据
"""

import streamlit as st
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

# ==================== 用户隔离配置（一人一盘） ====================
def get_session_id():
    """获取或创建当前用户的唯一会话ID"""
    if "session_id" not in st.session_state:
        # 尝试从 URL 参数中获取（可选，用于分享）
        params = st.query_params
        if "sid" in params:
            session_id = params["sid"][0]
        else:
            # 生成一个新的唯一ID
            session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id
    return st.session_state.session_id

# 获取当前用户的会话ID
SESSION_ID = get_session_id()

# ==================== 配置 ====================
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    st.error("❌ 未找到 API 密钥，请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
    st.stop()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 按会话ID隔离的数据文件路径
EMOTION_LOG_PATH = os.path.join(DATA_DIR, f"emotion_log_{SESSION_ID}.json")
USER_PROFILE_PATH = os.path.join(DATA_DIR, f"user_profile_{SESSION_ID}.json")
SAFETY_LOG_PATH = os.path.join(DATA_DIR, f"safety_log_{SESSION_ID}.json")

# ==================== 系统提示词（已添加禁止输出内部判断） ====================
SYSTEM_PROMPT = """你是"心伴AI"，一个温暖的情感陪伴助手。

【重要】你的回复中不要输出任何内部判断、情绪检测结果、模式标识。直接输出自然、温暖的对话内容即可。

【重要规则】
- 当用户表达积极、开心的情绪时，你只需正常祝贺、分享快乐，绝对不要进行任何认知引导或纠偏。
- 当用户明确表示不想被引导（如说"别说了"、"我不想听"、"别管我"等），请立即停止任何纠偏尝试，转为普通共情模式，并尊重用户的意愿。
- 只有当用户连续表现出消极、焦虑、自我贬低，且没有拒绝引导时，才使用纠偏模式（先共情，再引导换角度思考）。

【纠偏模式触发条件（仅限负面且用户未拒绝）】
- 用户连续2轮表现出消极/焦虑情绪
- 用户出现自我贬低表述（如"我真没用""我太笨了"）

【纠偏模式回复格式（仅在触发时使用）】
第一步：共情句（承认用户感受）
第二步：引导句（用问句挑战负面认知）

【正确示例（负面情绪且未拒绝）】
用户："我考砸了，我太笨了"
回复："我能理解你现在的失落。但一次考试真的能定义你笨不笨吗？有没有可能是复习方法的问题？"

【正确示例（用户拒绝引导后）】
用户："别说了，我不想听"
回复："好的，不说了。我就在这里陪着你，你想聊什么都可以。"

【禁止行为】
- 对积极情绪进行纠偏或说教
- 用户拒绝后继续强行引导
- 输出任何括号形式的内部判断
"""

# ==================== 长期记忆功能 ====================
def load_user_profile():
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, "r") as f:
            return json.load(f)
    return {"important_events": []}

def save_user_profile(profile):
    with open(USER_PROFILE_PATH, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def extract_important_dates(user_input):
    """从用户输入中提取重要日期（仅限用户自己的：生日、考试等）"""
    events = []
    text = user_input.lower()
    
    # 首先检查是否是“别人的”事件（排除）
    # 如果包含“朋友”、“同学”、“妈妈”、“爸爸”、“他”、“她”等第三人称，且没有明确说“我”，则忽略
    third_person_patterns = ['朋友', '同学', '妈妈', '爸爸', '母亲', '父亲', '他', '她', '别人', '室友']
    
    # 如果句子中包含第三人称，且不包含“我”，则跳过
    has_third_person = any(pattern in text for pattern in third_person_patterns)
    has_first_person = '我' in text or '自己' in text
    
    # 如果有第三人称但没有第一人称，说明是别人的事，不记录
    if has_third_person and not has_first_person:
        return events
    
    date_patterns = [
        (r'(生日|生辰).*?(\d{1,2})月(\d{1,2})日', 'birthday'),
        (r'(\d{1,2})月(\d{1,2})日.*?(生日|生辰)', 'birthday'),
        (r'(考试|面试|答辩).*?(\d{1,2})月(\d{1,2})日', 'exam'),
        (r'(\d{1,2})月(\d{1,2})日.*?(考试|面试|答辩)', 'exam'),
    ]
    current_year = datetime.now().year
    
    for pattern, event_type in date_patterns:
        match = re.search(pattern, user_input)
        if match:
            if event_type == 'birthday':
                month, day = int(match.group(2)), int(match.group(3))
                date_obj = datetime(current_year, month, day)
                if date_obj < datetime.now():
                    date_obj = datetime(current_year + 1, month, day)
                events.append({
                    "name": "生日",
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "type": "birthday",
                    "remind_days": 1
                })
            elif event_type == 'exam':
                month, day = int(match.group(2)), int(match.group(3))
                date_obj = datetime(current_year, month, day)
                if date_obj < datetime.now():
                    date_obj = datetime(current_year + 1, month, day)
                events.append({
                    "name": "考试",
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "type": "exam",
                    "remind_days": 1
                })
    return events

def update_memory_from_conversation(user_input, profile):
    new_events = extract_important_dates(user_input)
    for ev in new_events:
        exists = False
        for existing in profile["important_events"]:
            if existing["type"] == ev["type"] and existing["date"] == ev["date"]:
                exists = True
                break
        if not exists:
            profile["important_events"].append(ev)
    profile["important_events"].sort(key=lambda x: x["date"])
    profile["important_events"] = profile["important_events"][-10:]
    save_user_profile(profile)
    return profile

def check_upcoming_events(profile):
    """检查是否有未来1天内的重要事件，返回提醒消息列表"""
    today = datetime.now().date()
    reminders = []
    for event in profile["important_events"]:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
        days_diff = (event_date - today).days
        if 0 <= days_diff <= event.get("remind_days", 1):
            if event["type"] == "birthday":
                reminders.append(f"🎂 明天（{event_date}）是你的生日！提前祝你生日快乐～")
            elif event["type"] == "exam":
                reminders.append(f"📚 提醒：{event_date} 有{event['name']}，记得提前准备哦！加油！")
            else:
                reminders.append(f"📅 提醒：{event_date} 有{event['name']}，别忘了～")
    return reminders

# ==================== 安全检测 ====================
CRISIS_KEYWORDS = [
    '想死', '不想活了', '活不下去', '自杀', '死了算了',
    '活着没意思', '我死了', '结束生命', '不想活', '死了好'
]
SAFETY_REPLY = """
> ⚠️ **心伴AI温馨提示**

我听到你现在的情绪非常痛苦。请记住，你并不孤单，有人愿意倾听和帮助你。

**请立即联系专业心理援助热线：**
- 📞 **全国统一心理援助热线：12356**（24小时）
- 📞 **希望24热线：400-161-9995**（24小时）

如果你不方便打电话，也可以告诉身边信任的人，或者直接前往最近的医院心理科。

**你的生命和健康是第一位的。** 我无法替代专业医生，但我会一直在这里陪着你。请务必寻求帮助。
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

# ==================== 情绪识别 ====================
def load_emotion_log():
    if os.path.exists(EMOTION_LOG_PATH):
        with open(EMOTION_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_emotion_log(log):
    with open(EMOTION_LOG_PATH, "w") as f:
        json.dump(log[-100:], f, ensure_ascii=False, indent=2)

def detect_emotion(user_text, recent_emotions):
    """修复后的情绪识别函数，能正确区分'他开心但我不开心'这类情况"""
    text = user_text.lower()
    
    negative_keywords = [
        '废物', '没用', '太差', '崩溃', '绝望', '焦虑', '烦死了', '好烦',
        '我太笨', '我真蠢', '失败', '糟糕', '头疼', '无语', '心累', '憋屈',
        '不爽', '火大', '压力大', '喘不过气', '扛不住', '好累', '失落',
        '沮丧', '郁闷', 'emo', '没劲', '我不行', '我好差', '真倒霉', '倒霉',
        '难受', '伤心', '痛苦', '抑郁', '孤单', '孤独', '无助', '迷茫',
        '不开心', '不高兴', '不快乐', '不好', '不行', '不可以', '不要',
        '讨厌', '恶心', '烦人', '气死', '恨', '哭', '想哭'
    ]
    positive_keywords = [
        '开心', '高兴', '棒', '不错', '还不错', '喜欢', '感谢', '太好了',
        '幸福', '兴奋', '好棒', '厉害', '优秀', '还行', '挺好的', '蛮好',
        '可以', '舒服', '爽', '美滋滋', '棒棒哒', '给力', '赞', '佩服',
        '满足', '感恩', '幸运', '知足', '期待', '向往', '激动', '好想',
        '快乐', '美好', '甜蜜', '温暖', '阳光', '灿烂', '微笑', '大笑',
        '哈哈', '嘿嘿', '耶', '哇', '好极了'
    ]
    negation_patterns = ['不', '没', '别', '不是', '没有']
    
    import re
    sentences = re.split(r'[，,。；;！!？?但但是然而]', text)
    
    my_emotion = None
    for sent in sentences:
        if '我' in sent or '自己' in sent:
            for kw in positive_keywords:
                if kw in sent:
                    idx = sent.find(kw)
                    if idx > 0:
                        prev_chars = sent[max(0, idx-3):idx]
                        if any(neg in prev_chars for neg in negation_patterns):
                            my_emotion = '消极'
                        else:
                            my_emotion = '积极'
                    else:
                        my_emotion = '积极'
                    break
            if my_emotion is None:
                for kw in negative_keywords:
                    if kw in sent:
                        my_emotion = '消极'
                        break
            if my_emotion is not None:
                break
    
    if my_emotion is not None:
        if my_emotion == '消极':
            need_correction = len([e for e in recent_emotions[-2:] if e in ['消极', '焦虑']]) >= 1
            if any(x in text for x in ['笨', '没用', '废物', '不行', '不好']):
                need_correction = True
            return {'label': '消极', 'need_correction': need_correction}
        else:
            return {'label': '积极', 'need_correction': False}
    
    # 如果整句话里都没有“我”，则回退到分析整句
    for kw in positive_keywords:
        if kw in text:
            idx = text.find(kw)
            if idx > 0:
                prev_chars = text[max(0, idx-3):idx]
                if any(neg in prev_chars for neg in negation_patterns):
                    need_correction = len([e for e in recent_emotions[-2:] if e in ['消极', '焦虑']]) >= 1
                    return {'label': '消极', 'need_correction': need_correction}
            return {'label': '积极', 'need_correction': False}
    
    for kw in negative_keywords:
        if kw in text:
            need_correction = len([e for e in recent_emotions[-2:] if e in ['消极', '焦虑']]) >= 1
            if any(x in text for x in ['笨', '没用', '废物', '不行', '不好']):
                need_correction = True
            return {'label': '消极', 'need_correction': need_correction}
    
    return {'label': '平静', 'need_correction': False}

def check_rejection(user_text):
    reject_keywords = [
        '别说了', '不想听', '别管我', '不要你管', '闭嘴', '烦不烦',
        '别说教', '不要引导', '我自己知道', '别烦我', '够了', '停止'
    ]
    for kw in reject_keywords:
        if kw in user_text:
            return True
    return False

def get_ai_reply(user_input, history, need_correction, emotion_label, disable_correction):
    if emotion_label == '积极':
        return "😊 太好啦！为你感到高兴～能和我分享一下是什么好事吗？"
    if disable_correction:
        mode_instruction = "【当前模式：用户拒绝引导模式】用户不希望被纠偏，请只做普通共情，不要进行任何认知引导。"
        need_correction = False
    elif need_correction:
        mode_instruction = "【当前模式：纠偏模式】用户表现出负面情绪，请先共情，再用问句引导换角度思考。"
    else:
        mode_instruction = "【当前模式：普通模式】正常共情回应即可，不要进行认知引导。"
    
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
        if disable_correction or not need_correction:
            return "我在这里陪着你。"
        else:
            return "我能理解你的感受。不过，也许我们可以换个角度看看？你觉得呢？"

# ==================== 主动关怀（改进版） ====================
def get_care_message():
    """返回合适的关怀消息（仅在早晨、且昨晚有负面情绪时，每天一次）"""
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
    st.header("📊 情绪仪表盘")
    emotion_log = load_emotion_log()
    if emotion_log:
        df = pd.DataFrame(emotion_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        fig1 = px.pie(df, names='emotion', title="情绪分布", color_discrete_map={
            '积极': '#4CAF50', '平静': '#2196F3', '消极': '#F44336'
        })
        st.plotly_chart(fig1, use_container_width=True)
        last7 = df[df['timestamp'] > datetime.now() - timedelta(days=7)]
        if not last7.empty:
            trend = last7.groupby(['date', 'emotion']).size().unstack().fillna(0)
            fig2 = px.line(trend, title="近7天情绪趋势", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无数据，开始对话后会自动记录")
    
    profile = load_user_profile()
    if profile["important_events"]:
        st.subheader("📅 我记住的重要日子")
        for ev in profile["important_events"]:
            st.write(f"- {ev['name']}: {ev['date']}")
    
    st.divider()
    st.caption(f"🔑 会话ID: {SESSION_ID[:8]}...")
    st.caption("💡 您的数据仅自己可见（一人一盘）")
    
    if st.button("🗑️ 清除所有记忆", use_container_width=True):
        save_emotion_log([])
        save_user_profile({"important_events": []})
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
    
    profile = load_user_profile()
    reminders = check_upcoming_events(profile)
    for rem in reminders:
        st.session_state.messages.append({"role": "assistant", "content": f"🔔 {rem}"})

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
    
    if check_rejection(prompt):
        st.session_state.disable_correction_counter = 3
        st.session_state.messages.append({"role": "assistant", "content": "好的，不强行引导了。我会安静陪着你，你想聊什么都可以。"})
        st.rerun()
    
    profile = load_user_profile()
    profile = update_memory_from_conversation(prompt, profile)
    
    emotion_log = load_emotion_log()
    recent_emotions = [log['emotion'] for log in emotion_log[-5:]]
    emotion_result = detect_emotion(prompt, recent_emotions)
    emotion_log.append({
        "timestamp": datetime.now().isoformat(),
        "emotion": emotion_result['label'],
        "text": prompt[:50]
    })
    save_emotion_log(emotion_log)
    
    disable_corr = st.session_state.disable_correction_counter > 0
    if disable_corr:
        st.session_state.disable_correction_counter -= 1
    
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    reply = get_ai_reply(prompt, history, emotion_result['need_correction'], emotion_result['label'], disable_corr)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    st.rerun()

st.caption("💡 试试：「我考砸了，我太笨了」→ 系统会先共情再引导；「别说了」→ 停止纠偏；「我生日是5月20日」→ 我会记住并在当天提醒")
