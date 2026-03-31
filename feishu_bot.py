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
from uuid import uuid4

# 引入你的最强大脑
from src.agent.main_agent import NovelAgent

# 显式加载 .env 到进程环境变量，确保 os.getenv 可读取
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("feishu_bot")

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
memory_op_lock = threading.RLock()

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
            logger.debug("tenant_access_token 命中缓存，剩余有效期 %.1fs", expire_at - now)
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
        logger.info("tenant_access_token 已刷新，预计 %.1f 分钟后过期", (tenant_token_cache["expire_at"] - now) / 60)
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
    logger.info("飞书回复成功 | message_id=%s | text_len=%d", message_id, len(content or ""))


def _sanitize_feishu_text(text: str) -> str:
    """清理飞书文本中的 @ 机器人片段和多余空白。"""
    content = (text or "").strip()
    # 【精准打击】：只匹配飞书特征的 @_user_xxx 或 @ou_xxx，避免误杀正常邮箱地址
    content = re.sub(r"@(_user_[0-9]+|ou_[a-zA-Z0-9]+)", "", content)
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


def _parse_memory_command(user_text: str) -> Optional[str]:
    """解析飞书文本中的记忆删除命令，返回 1 / 2 / 12 或 None。"""
    text = (user_text or "").strip().replace("：", ":")
    if not text:
        return None

    compact = text.replace(" ", "")
    # 支持：删除记忆、删除记忆1、删除记忆 2、删除记忆:12
    if compact in {"删除记忆", "清空记忆"}:
        return "help"
    if compact in {"删除记忆1", "删除记忆:1", "清空记忆1", "清空记忆:1"}:
        return "1"
    if compact in {"删除记忆2", "删除记忆:2", "清空记忆2", "清空记忆:2"}:
        return "2"
    if compact in {
        "删除记忆12",
        "删除记忆21",
        "删除记忆:12",
        "删除记忆:21",
        "清空记忆12",
        "清空记忆21",
        "清空记忆:12",
        "清空记忆:21",
    }:
        return "12"
    return None


def process_memory_command_and_reply(user_id: str, message_id: str, command: str):
    """后台执行飞书记忆删除命令并回包。"""
    logger.info("记忆命令开始 | user_id=%s | message_id=%s | command=%s", user_id, message_id, command)
    try:
        if command == "help":
            help_text = (
                "删除记忆指令如下:\n"
                "1) 删除记忆 1：清空 KV profile\n"
                "2) 删除记忆 2：删除图谱边 + 重建记忆摘要集合\n"
                "3) 删除记忆 12：同时执行 1 和 2"
            )
            reply_feishu_message(message_id, help_text)
            return

        with memory_op_lock:
            mm = get_agent().memory_manager
            if command == "1":
                result = mm.clear_kv_profile()
            elif command == "2":
                result = mm.clear_user_graph_and_memory_db()
            elif command == "12":
                result = mm.clear_all_memories()
            else:
                result = "无效删除指令。请发送：删除记忆 1 / 2 / 12"

        with session_pool_lock:
            session_pool[user_id] = []

        reply_feishu_message(message_id, f"🧹 {result}")
        logger.info("记忆命令完成 | user_id=%s | message_id=%s | command=%s", user_id, message_id, command)
    except Exception as e:
        logger.exception("记忆命令失败 | user_id=%s | message_id=%s | command=%s | err=%s", user_id, message_id, command, e)
        try:
            reply_feishu_message(message_id, "记忆删除失败，请稍后重试。")
        except Exception:
            logger.exception("记忆命令失败后回包失败 | message_id=%s", message_id)

# ==========================================
# 3. 异步后台任务：执行硬核 RAG 流水线
# ==========================================
def process_rag_and_reply(user_id: str, message_id: str, text: str):
    """
    这个函数在后台运行，无论跑多久都不会导致飞书超时报错！
    """
    start_ts = time.time()
    logger.info("后台任务开始 | user_id=%s | message_id=%s | text_len=%d", user_id, message_id, len(text or ""))
    
    # 1. 获取该用户的专属历史记录
    with session_pool_lock:
        if user_id not in session_pool:
            session_pool[user_id] = []
        # 复制一份，避免并发线程同时操作同一 list
        chat_history = list(session_pool[user_id])
    logger.info("会话上下文就绪 | user_id=%s | history_messages=%d", user_id, len(chat_history))
    
    try:
        # 2. 调用你的超级 Agent 思考
        reply_text = get_agent().chat(user_message=text, chat_history=chat_history)
        logger.info(
            "Agent 生成完成 | user_id=%s | message_id=%s | cost_ms=%d | reply_len=%d",
            user_id,
            message_id,
            int((time.time() - start_ts) * 1000),
            len(reply_text or ""),
        )
        
        # 3. 维护上下文（写回时加锁）
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": reply_text})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
        with session_pool_lock:
            session_pool[user_id] = chat_history
            
        # 4. 把大模型生成的回复发回飞书
        reply_feishu_message(message_id, reply_text)
        logger.info(
            "后台任务完成 | user_id=%s | message_id=%s | total_cost_ms=%d",
            user_id,
            message_id,
            int((time.time() - start_ts) * 1000),
        )
        
    except Exception as e:
        logger.exception(
            "后台任务失败 | user_id=%s | message_id=%s | err=%s",
            user_id,
            message_id,
            e,
        )
        try:
            reply_feishu_message(message_id, "系统引擎过载，请稍后再试。")
        except Exception as send_err:
            logger.exception("失败兜底消息发送失败 | message_id=%s | err=%s", message_id, send_err)

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
        logger.info("服务关闭：Agent 资源已释放")
    except Exception as e:
        logger.warning("关闭 Agent 失败: %s", e)


@app.post("/feishu/webhook")
async def feishu_webhook(request: Request, background_tasks: BackgroundTasks):
    """飞书事件订阅入口"""
    req_id = uuid4().hex[:8]
    req_start = time.time()
    logger.info("Webhook 请求进入 | req_id=%s", req_id)

    try:
        payload = await request.json()
    except Exception:
        logger.warning("Webhook JSON 解析失败 | req_id=%s", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(
        "Webhook 载荷摘要 | req_id=%s | top_keys=%s | event_type=%s | has_encrypt=%s",
        req_id,
        list(payload.keys()),
        payload.get("header", {}).get("event_type") or payload.get("event", {}).get("type"),
        "encrypt" in payload,
    )

    # 若启用“事件加密”，飞书只会推送 encrypt 字段；当前实现未做解密。
    if "encrypt" in payload:
        logger.error("收到加密事件但未配置解密 | req_id=%s", req_id)
        return {"msg": "ok"}

    # 1. 处理飞书 URL 验证挑战 (第一次配置事件订阅时触发)
    if "challenge" in payload:
        if payload.get("token") != FEISHU_VERIFICATION_TOKEN:
            logger.warning("challenge token 校验失败 | req_id=%s", req_id)
            raise HTTPException(status_code=403, detail="Token Invalid")
        logger.info("challenge 校验成功 | req_id=%s", req_id)
        return {"challenge": payload["challenge"]}

    # 普通事件也执行 token 校验（若飞书推送包含 token 字段）
    event_token = payload.get("token") or payload.get("header", {}).get("token")
    if event_token and event_token != FEISHU_VERIFICATION_TOKEN:
        logger.warning("事件 token 校验失败 | req_id=%s", req_id)
        raise HTTPException(status_code=403, detail="Token Invalid")

    # 2. 处理消息推送事件（兼容 schema 2.0 / 旧版事件）
    msg_type, message_id, user_id, user_text_raw = _extract_message_event(payload)
    if msg_type is None:
        logger.info("非消息事件已忽略 | req_id=%s | payload_keys=%s", req_id, list(payload.keys()))
        return {"msg": "ok"}

    logger.info(
        "消息事件解析结果 | req_id=%s | msg_type=%s | message_id=%s | user_id=%s",
        req_id,
        msg_type,
        message_id,
        user_id,
    )

    if msg_type != "text":
        logger.info("非 text 消息已忽略 | req_id=%s | msg_type=%s", req_id, msg_type)
        return {"msg": "ok"}

    message_id = message_id or ""

    # 幂等去重：飞书重试时避免重复处理
    _cleanup_processed_ids()
    with processed_lock:
        if message_id and message_id in processed_message_ids:
            logger.info("重复消息已忽略 | req_id=%s | message_id=%s", req_id, message_id)
            return {"msg": "ok"}
        if message_id:
            processed_message_ids[message_id] = time.time()

    user_text = _sanitize_feishu_text(user_text_raw or "")
    if not user_text:
        logger.info("消息文本为空已忽略 | req_id=%s | message_id=%s", req_id, message_id)
        return {"msg": "ok"}

    logger.info(
        "收到文本消息 | req_id=%s | user_id=%s | message_id=%s | text_len=%d",
        req_id,
        user_id,
        message_id,
        len(user_text),
    )

    # 支持飞书命令式记忆删除（原 CLI 的“删除记忆”菜单在飞书场景不可交互）
    memory_cmd = _parse_memory_command(user_text)
    if memory_cmd is not None:
        background_tasks.add_task(process_memory_command_and_reply, user_id, message_id, memory_cmd)
        logger.info(
            "记忆命令已提交 | req_id=%s | user_id=%s | message_id=%s | command=%s | ack_cost_ms=%d",
            req_id,
            user_id,
            message_id,
            memory_cmd,
            int((time.time() - req_start) * 1000),
        )
        return {"msg": "ok"}

    # 【架构精髓】：把耗时的 RAG 任务挂到后台，立刻返回 200 给飞书，终结超时困扰！
    background_tasks.add_task(process_rag_and_reply, user_id, message_id, user_text)
    logger.info("后台任务已提交 | req_id=%s | ack_cost_ms=%d", req_id, int((time.time() - req_start) * 1000))

    return {"msg": "ok"}