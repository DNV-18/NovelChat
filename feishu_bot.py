import os
import json
import logging
import time
import re
import threading
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from typing import Dict, List, Union, Optional, Tuple

# 引入你的最强大脑
from src.agent.main_agent import NovelAgent

# 显式加载 .env 到进程环境变量，确保 os.getenv 可读取
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="吞噬星空 飞书网关")

# ==========================================
# 1. 飞书配置与全局单例
# ==========================================
# 只从环境变量读取，避免在代码中硬编码敏感凭证
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "")

if not FEISHU_APP_ID or not FEISHU_APP_SECRET or not FEISHU_VERIFICATION_TOKEN:
    raise RuntimeError(
        "缺少飞书配置：请设置 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_VERIFICATION_TOKEN"
    )

# 全局 Agent 单例（共享数据库连接，节省资源）
agent = None
agent_lock = threading.RLock()


def get_agent() -> NovelAgent:
    """懒加载 Agent，避免导入阶段因外部依赖未就绪导致服务启动失败。"""
    global agent
    with agent_lock:
        if agent is None:
            agent = NovelAgent()
        return agent

# 多租户会话隔离池：{"open_id": [{"role": "user", "content": "..."}]}
session_pool: Dict[str, List[Dict]] = {}
session_pool_lock = threading.RLock()

# 简单去重池（避免飞书重试导致重复处理）
processed_message_ids: Dict[str, float] = {}
processed_lock = threading.RLock()

# 飞书 tenant_access_token 缓存
tenant_token_cache: Dict[str, Union[float, str]] = {"token": "", "expire_at": 0.0}
tenant_token_lock = threading.RLock()


def _cleanup_processed_ids(ttl_seconds: int = 600):
    now = time.time()
    with processed_lock:
        stale = [mid for mid, ts in processed_message_ids.items() if now - ts > ttl_seconds]
        for mid in stale:
            processed_message_ids.pop(mid, None)

# ==========================================
# 2. 飞书 API 交互工具函数
# ==========================================
def get_tenant_access_token() -> str:
    """获取飞书接口的通行证 (Token 2小时有效，生产环境建议加上 Redis 缓存)"""
    now = time.time()
    with tenant_token_lock:
        cached_token = str(tenant_token_cache.get("token") or "")
        expire_at = float(tenant_token_cache.get("expire_at") or 0.0)
        if cached_token and now < expire_at:
            return cached_token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET,
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        res = resp.json()
        if res.get("code") != 0:
            raise RuntimeError(f"获取 tenant_access_token 失败: {res}")

        token = str(res.get("tenant_access_token") or "")
        expire = int(res.get("expire") or 7200)
        if not token:
            raise RuntimeError("获取 tenant_access_token 失败: 返回空 token")

        tenant_token_cache["token"] = token
        tenant_token_cache["expire_at"] = now + max(60, expire - 60)
        return token

def reply_feishu_message(message_id: str, content: str):
    """主动向飞书用户发送文本消息"""
    token = get_tenant_access_token()
    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "msg_type": "text",
        "content": json.dumps({"text": content}),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"飞书回复失败: {body}")


def _sanitize_feishu_text(text: str) -> str:
    """清理飞书文本中的 @ 机器人片段和多余空白。"""
    content = (text or "").strip()
    # 常见格式：@_user_1 或 @ou_xxx
    content = re.sub(r"@[_a-zA-Z0-9]+", "", content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def _extract_text_from_content(content_raw: str) -> str:
    """兼容解析 message.content 字段（可能是 JSON 字符串，也可能是纯文本）。"""
    if not content_raw:
        return ""
    try:
        parsed = json.loads(content_raw)
        if isinstance(parsed, dict):
            return str(parsed.get("text", "") or "")
    except Exception:
        pass
    return str(content_raw)


def _extract_message_event(payload: Dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    返回 (msg_type, message_id, user_id, user_text)
    - 支持飞书 schema 2.0（header/event）
    - 兼容旧版 event 结构（event.type == message）
    """
    header = payload.get("header", {}) or {}
    event = payload.get("event", {}) or {}

    # schema 2.0: im.message.receive_v1
    if header.get("event_type") == "im.message.receive_v1":
        message = event.get("message", {}) or {}
        sender = event.get("sender", {}) or {}
        msg_type = message.get("message_type")
        message_id = message.get("message_id") or ""
        user_id = sender.get("sender_id", {}).get("open_id", "unknown_user")
        raw_content = message.get("content", "{}")
        user_text = _extract_text_from_content(raw_content)
        return msg_type, message_id, user_id, user_text

    # 旧版 event callback 兼容
    if event.get("type") == "message":
        msg_type = event.get("msg_type") or event.get("message_type") or "text"
        message_id = event.get("open_message_id") or event.get("message_id") or ""
        user_id = (
            event.get("open_id")
            or event.get("user_open_id")
            or event.get("sender", {}).get("sender_id", {}).get("open_id")
            or "unknown_user"
        )
        user_text = event.get("text") or _extract_text_from_content(event.get("content", ""))
        return msg_type, message_id, user_id, str(user_text or "")

    return None, None, None, None

# ==========================================
# 3. 异步后台任务：执行硬核 RAG 流水线
# ==========================================
def process_rag_and_reply(user_id: str, message_id: str, text: str):
    """
    这个函数在后台运行，无论跑多久都不会导致飞书超时报错！
    """
    logging.info(f"🚀 开始为用户 {user_id} 执行 RAG 检索...")
    
    # 1. 获取该用户的专属历史记录
    with session_pool_lock:
        if user_id not in session_pool:
            session_pool[user_id] = []
        # 复制一份，避免并发线程同时操作同一 list
        chat_history = list(session_pool[user_id])
    
    try:
        # 2. 调用你的超级 Agent 思考
        reply_text = get_agent().chat(user_message=text, chat_history=chat_history)
        
        # 3. 维护上下文（写回时加锁）
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": reply_text})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
        with session_pool_lock:
            session_pool[user_id] = chat_history
            
        # 4. 把大模型生成的回复发回飞书
        reply_feishu_message(message_id, reply_text)
        
    except Exception as e:
        logging.error(f"处理失败: {e}")
        try:
            reply_feishu_message(message_id, "系统引擎过载，请稍后再试。")
        except Exception as send_err:
            logging.error(f"飞书回复失败: {send_err}")

# ==========================================
# 4. FastAPI 路由入口 (接收飞书的 Webhook 推送)
# ==========================================
@app.on_event("shutdown")
def on_shutdown():
    """关闭进程时释放 Agent 资源。"""
    try:
        global agent
        if agent is not None:
            agent.close()
            agent = None
    except Exception as e:
        logging.warning(f"关闭 Agent 失败: {e}")


@app.post("/feishu/webhook")
async def feishu_webhook(request: Request, background_tasks: BackgroundTasks):
    """飞书事件订阅入口"""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 若启用“事件加密”，飞书只会推送 encrypt 字段；当前实现未做解密。
    if "encrypt" in payload:
        logging.error("收到加密事件，但当前未配置解密逻辑。请在飞书后台关闭事件加密，或补充解密实现。")
        return {"msg": "ok"}

    # 1. 处理飞书 URL 验证挑战 (第一次配置事件订阅时触发)
    if "challenge" in payload:
        if payload.get("token") != FEISHU_VERIFICATION_TOKEN:
            raise HTTPException(status_code=403, detail="Token Invalid")
        return {"challenge": payload["challenge"]}

    # 普通事件也执行 token 校验（若飞书推送包含 token 字段）
    if payload.get("token") and payload.get("token") != FEISHU_VERIFICATION_TOKEN:
        raise HTTPException(status_code=403, detail="Token Invalid")

    # 2. 处理消息推送事件（兼容 schema 2.0 / 旧版事件）
    msg_type, message_id, user_id, user_text_raw = _extract_message_event(payload)
    if msg_type is None:
        logging.info("收到非消息事件，已忽略。payload keys=%s", list(payload.keys()))
        return {"msg": "ok"}

    if msg_type != "text":
        logging.info("收到非 text 消息，已忽略。msg_type=%s", msg_type)
        return {"msg": "ok"}

    message_id = message_id or ""

    # 幂等去重：飞书重试时避免重复处理
    _cleanup_processed_ids()
    with processed_lock:
        if message_id and message_id in processed_message_ids:
            logging.info(f"♻️ 重复消息已忽略: {message_id}")
            return {"msg": "ok"}
        if message_id:
            processed_message_ids[message_id] = time.time()

    user_text = _sanitize_feishu_text(user_text_raw or "")
    if not user_text:
        logging.info("消息文本为空，已忽略。message_id=%s", message_id)
        return {"msg": "ok"}

    logging.info(f"📥 收到飞书消息 [{user_id}]: {user_text}")

    # 【架构精髓】：把耗时的 RAG 任务挂到后台，立刻返回 200 给飞书，终结超时困扰！
    background_tasks.add_task(process_rag_and_reply, user_id, message_id, user_text)

    return {"msg": "ok"}