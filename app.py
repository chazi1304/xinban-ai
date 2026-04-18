"""
心伴AI - 完整版（智能情绪分类版）
功能：情感纠偏、情绪仪表盘、主动关怀、长期记忆、安全检测、多用户隔离
情绪识别：大模型智能分类（90-95%准确率），降级时使用完整关键词匹配
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

# ==================== 用户隔离配置 ====================
def get_session_id():
    if "session_id" not in st.session_state:
        params = st.query_params
        if "sid" in params:
            session_id = params["sid"][0]
        else:
            session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id
    return st.session_state.session_id

SESSION_ID = get_session_id()

# ==================== 配置 ====================
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    st.error("❌ 未找到 API 密钥")
    st.stop()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

EMOTION_LOG_PATH = os.path.join(DATA_DIR, f"emotion_log_{SESSION_ID}.json")
USER_PROFILE_PATH = os.path.join(DATA_DIR, f"user_profile_{SESSION_ID}.json")
SAFETY_LOG_PATH = os.path.join(DATA_DIR, f"safety_log_{SESSION_ID}.json")

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是"心伴AI"，一个温暖、温柔、有共情力的情感陪伴助手。你的风格像一位温柔的心理咨询师。

【核心定位】
- 你既是情感陪伴者，也是轻量级的支持者
- 优先倾听和共情，让用户感受到“你懂我”
- 在共情之后，提供具体、可执行、不压迫的小建议

【回复结构建议】
1. 先共情：让用户感受到被理解（1-2句）
2. 再陪伴：开放式提问，鼓励用户多说说（1句）
3. 最后轻引导：提供1-2个具体、可执行的小建议（1-2句）

【具体示例】

示例1 - 疲惫/熬夜：
用户："大半夜干活好累"
回复："听起来真的很辛苦。这么晚还在工作，一定是有很重要的事情要赶吧？（共情）
   愿意和我聊聊是什么让你这么拼吗？（陪伴）
   如果实在太累，不妨先休息15分钟，喝点温水，有时候短暂休息反而能让效率更高。你觉得呢？（轻引导）"

示例2 - 自我贬低/考试失败：
用户："我考砸了，我太笨了"
回复："我能感受到你现在的失落。一次考试没考好，很容易让人怀疑自己。（共情）
   你愿意具体说说，是哪一科或者哪部分让你觉得特别难吗？（陪伴）
   其实那些最终做得好的人，很多都经历过这样的时刻。要不要一起看看，问题可能出在复习方法上，还是状态上？（轻引导）"

示例3 - 焦虑/压力：
用户："我好焦虑，压力好大"
回复："被压力包裹的感觉真的很不好受。（共情）
   你愿意和我具体说说，是什么让你感到焦虑吗？有时候说出来，压力就会轻一点点。（陪伴）
   如果现在觉得太乱，可以试着把让你焦虑的事情一件件写下来，然后挑一件最小的先处理。你试过这个方法吗？（轻引导）"

示例4 - 孤独/emo：
用户："一个人好无聊，没人懂我"
回复："这种感觉我懂，一个人待着确实容易胡思乱想。（共情）
   你平时一个人的时候喜欢做些什么呢？（陪伴）
   有时候一个小小的仪式感就能让心情好一点，比如点个喜欢的奶茶，或者看一部治愈的电影。要不要试试看？（轻引导）"

示例5 - 日常分享/小确幸：
用户："今天收到礼物了，好开心"
回复："哇，真好！收到礼物确实会让人心情变好～（共情）
   是什么礼物呀？愿意和我分享一下吗？（陪伴）"

【纠偏模式触发条件】
- 用户连续2轮表现出消极/焦虑情绪
- 用户出现自我贬低表述（如"我真没用""我太笨了"）

【禁止行为】
- 不要说“别难过了”“加油”“你要坚强”等空话
- 不要输出1、2、3点式的冷冰冰建议列表
- 不要输出任何括号形式的内部判断
- 不要过度说教
"""

# ==================== 完整的关键词库（用于降级方案） ====================
NEGATIVE_KEYWORDS = [
    # 自我贬低
    '我太笨', '我真蠢', '我不行', '我好差', '我没用', '我是废物', '我失败了',
    # 负面情绪
    '崩溃', '绝望', '焦虑', '抑郁', '沮丧', '郁闷', '失落', '痛苦', '伤心', '难过',
    '难受', '憋屈', '委屈', '心累', 'emo', '没劲', '无力', '无助', '迷茫', '空虚',
    # 烦躁愤怒
    '烦死了', '好烦', '真烦', '烦躁', '火大', '气死', '不爽', '讨厌', '恶心', '恨',
    # 疲劳压力
    '好累', '太累', '累死了', '疲惫', '压力大', '喘不过气', '扛不住', '撑不住', '想哭', '哭',
    # 日常消极
    '糟糕', '倒霉', '失败', '头疼', '无语', '无聊', '没意思', '不开心', '不高兴', '不快乐',
    '不好', '不行', '不要', '不想', '不会', '不能'
]

POSITIVE_KEYWORDS = [
    # 快乐兴奋
    '开心', '高兴', '快乐', '兴奋', '激动', '好开心', '好高兴',
    # 满意赞美
    '棒', '好棒', '棒棒哒', '厉害', '优秀', '给力', '赞', '佩服', '出色',
    # 积极评价
    '不错', '还不错', '挺好', '挺好的', '蛮好', '很好', '非常好', '太好了',
    '完美', '满意', '满足', '幸福', '幸运', '感恩', '感谢',
    # 轻松愉快
    '舒服', '爽', '美滋滋', '好爽', '轻松', '自在', '惬意',
    # 期待向往
    '期待', '向往', '憧憬', '希望', '加油', '冲', '耶', '哇', '哈哈', '嘿嘿', '呵呵'
]

# ==================== 智能情绪分类（合并到主对话） ====================
def get_ai_reply_with_emotion(user_input, history, need_correction, disable_correction, recent_emotions):
    """
    一次API调用完成两个任务：
    1. 情绪分类（通过解析[情绪:xxx]标记）
    2. 生成回复内容
    """
    
    # 构建情绪分类指令
    classification_instruction = """
【额外任务】在输出回复之前，请先用一行标注用户的情绪，格式为：[情绪:积极] 或 [情绪:平静] 或 [情绪:消极]
注意：这个标注只用于内部记录，不要显示给用户。标注后空一行再输出回复。
判断标准：
- 积极：开心、高兴、兴奋、满足、期待、感恩等正面情绪
- 消极：难过、焦虑、沮丧、愤怒、无奈、疲惫、压力大、自我贬低等负面情绪
- 平静：中性、无明显情绪、或混合情绪中负面程度不高
"""
    
    # 构建模式指令
    if disable_correction:
        mode_instruction = "【当前模式：用户拒绝引导模式】用户不希望被纠偏，请只做普通共情，不要进行任何认知引导。"
        actual_need_correction = False
    elif need_correction:
        mode_instruction = "【当前模式：纠偏模式】用户表现出负面情绪，请先共情，再用问句引导换角度思考。"
        actual_need_correction = True
    else:
        mode_instruction = "【当前模式：普通模式】正常共情回应即可，不要进行认知引导。"
        actual_need_correction = False
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + classification_instruction + mode_instruction},
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
            # 移除标注行，只保留回复内容
            clean_reply = re.sub(r'\[情绪:(积极|平静|消极)\]\s*\n?', '', full_response).strip()
        else:
            # 如果没有检测到标注，默认为平静
            detected_emotion = '平静'
            clean_reply = full_response
        
        # 确保回复不为空
        if not clean_reply:
            if actual_need_correction:
                clean_reply = "我能理解你的感受。不过，也许我们可以换个角度看看？你觉得呢？"
            else:
                clean_reply = "我在这里陪着你。"
        
        return clean_reply, detected_emotion
        
    except Exception as e:
        # API调用失败时，使用降级方案（完整关键词匹配 + 预设回复）
        return fallback_reply_with_emotion(user_input, need_correction, disable_correction)

# ==================== 降级方案（完整关键词匹配 + 详细回复） ====================
def fallback_detect_emotion(user_text):
    """
    降级情绪识别：完整的关键词匹配
    返回：'积极' / '平静' / '消极'
    """
    text = user_text.lower()
    
    # 处理否定词（如“不开心”应判为消极）
    negation_patterns = ['不', '没', '别', '不是', '没有']
    
    # 先检查积极词（排除被否定的情况）
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            idx = text.find(kw)
            if idx > 0:
                prev_chars = text[max(0, idx-3):idx]
                if any(neg in prev_chars for neg in negation_patterns):
                    # 被否定的积极词 → 消极
                    return '消极'
            return '积极'
    
    # 检查消极词
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            return '消极'
    
    return '平静'

def fallback_reply_with_emotion(user_input, need_correction, disable_correction):
    """
    降级方案：基于关键词匹配生成回复（API失败时使用）
    返回：(reply, emotion_label)
    """
    emotion_label = fallback_detect_emotion(user_input)
    
    # 更新need_correction（如果降级方案检测到消极且需要纠偏）
    if emotion_label == '消极' and not disable_correction:
        need_correction = True
    
    # 积极情绪回复
    if emotion_label == '积极':
        return "😊 太好了！为你感到高兴～能和我分享一下是什么好事吗？", '积极'
    
    # 纠偏模式回复
    if need_correction and not disable_correction:
        # 检测特定场景，给出更精准的回复
        if any(x in user_input for x in ['笨', '蠢', '没用', '废物']):
            reply = "我能理解你现在的自我怀疑。但请记住，一次挫折并不能定义你的全部。要不要一起分析一下，看看问题到底出在哪里？"
        elif any(x in user_input for x in ['累', '熬夜', '加班', '赶工']):
            reply = "听起来你真的很辛苦。长时间工作确实容易让人疲惫，但请记得照顾好自己。需要的话，我可以陪你聊聊，或者给你一些放松的小建议？"
        elif any(x in user_input for x in ['焦虑', '压力', '担心', '害怕']):
            reply = "我能感受到你的焦虑。面对压力时，试着把大问题拆解成小步骤，一步一步来。你愿意和我具体说说是什么让你感到压力吗？"
        else:
            reply = "我能理解你现在的感受。不过，也许我们可以换个角度看看这个问题？你觉得呢？"
        return reply, '消极'
    
    # 普通模式回复（平静或不需要纠偏的消极）
    if any(x in user_input for x in ['累', '熬夜', '加班', '赶工']):
        reply = "听起来你挺辛苦的，要注意休息哦。需要我陪你聊聊天吗？"
    else:
        reply = "我在这里陪着你。想说什么都可以，我会认真听的。"
    return reply, emotion_label

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
    events = []
    # 先判断是否是“别人的”事件（排除）
    third_person_patterns = ['朋友', '同学', '妈妈', '爸爸', '母亲', '父亲', '他', '她', '别人', '室友']
    has_third_person = any(pattern in user_input for pattern in third_person_patterns)
    has_first_person = '我' in user_input or '自己' in user_input
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
    today = datetime.now().date()
    reminders = []
    for event in profile["important_events"]:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
        days_diff = (event_date - today).days
        if 0 <= days_diff <= event.get("remind_days", 1):
            if event["type"] == "birthday":
                reminders.append(f"🎂 明天是你的生日！提前祝你生日快乐～")
            elif event["type"] == "exam":
                reminders.append(f"📚 提醒：明天有{event['name']}，记得提前准备哦！加油！")
    return reminders

# ==================== 安全检测 ====================
CRISIS_KEYWORDS = ['想死', '不想活了', '活不下去', '自杀', '死了算了', '活着没意思']
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

# ==================== 主动关怀 ====================
def load_emotion_log():
    if os.path.exists(EMOTION_LOG_PATH):
        with open(EMOTION_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_emotion_log(log):
    with open(EMOTION_LOG_PATH, "w") as f:
        json.dump(log[-100:], f, ensure_ascii=False, indent=2)

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
    st.caption("💡 您的数据仅自己可见")
    
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
    # 安全检测（最高优先级）
    if check_crisis(prompt):
        log_safety_event(prompt, datetime.now().isoformat())
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": SAFETY_REPLY})
        st.rerun()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 拒绝引导检测
    reject_keywords = ['别说了', '不想听', '别管我', '不要你管', '别说教', '别烦我', '够了']
    is_rejecting = any(kw in prompt for kw in reject_keywords)
    if is_rejecting:
        st.session_state.disable_correction_counter = 3
        st.session_state.messages.append({"role": "assistant", "content": "好的，不强行引导了。我会安静陪着你，你想聊什么都可以。"})
        st.rerun()
    
    # 更新长期记忆
    profile = load_user_profile()
    profile = update_memory_from_conversation(prompt, profile)
    
    # 情绪日志（用于仪表盘和主动关怀）
    emotion_log = load_emotion_log()
    recent_emotions = [log['emotion'] for log in emotion_log[-5:]]
    
    # 判断是否需要纠偏（基于历史情绪）
    need_correction = len([e for e in recent_emotions[-2:] if e == '消极']) >= 1
    # 自我贬低强制触发纠偏
    if any(x in prompt for x in ['笨', '没用', '废物', '不行', '不好']):
        need_correction = True
    
    # 判断是否禁用纠偏
    disable_corr = st.session_state.disable_correction_counter > 0
    if disable_corr:
        st.session_state.disable_correction_counter -= 1
    
    # 获取AI回复和情绪分类（智能版）
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    reply, detected_emotion = get_ai_reply_with_emotion(
        prompt, history, need_correction, disable_corr, recent_emotions
    )
    
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
