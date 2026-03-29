import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from src.config import settings
from src.utils.model_factory import ModelFactory
from src.utils.prompts import (
    QUERY_ROUTER_SYSTEM_PROMPT,
    build_query_router_user_prompt,
)

# 配置简单的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RouteResult(BaseModel):
    """Router 必须严格输出的结构化 JSON 格式"""
    # 【优化点1】：增加推理字段。强制大模型先思考再输出 mode，准确率大幅提升！
    reasoning: str = Field(..., description="简短的思考过程：为什么选择这个模式？代词指代了谁？")
    mode: Literal["LOCAL", "GLOBAL", "MEMORY", "DIRECT"] = Field(..., description="检索模式：LOCAL, GLOBAL, MEMORY 或 DIRECT")
    rewritten_query: str = Field(..., description="指代消解后、适合作为独立检索的搜索词")
    entities: List[str] = Field(default_factory=list, description="提取出的核心实体列表")

class QueryRouter:
    """
    Phase 4: 智能查询路由器 (Query Router)
    拦截用户的原始提问，结合最近的聊天记录，使用便宜的小模型快速判定检索意图并提取搜索参数。
    """
    def __init__(self, model_tier: Optional[str] = None):
        # 为兼容旧调用方保留 llm_client 参数，但统一改为走 ModelFactory。
        self.model_tier = model_tier or settings.query_router_model_tier
        self.history_turns = settings.query_router_history_turns

    @staticmethod
    def _extract_json_block(raw_text: str) -> str:
        """从模型输出中提取 JSON 文本。"""
        if not raw_text:
            return ""
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def route_query(self, user_query: str, chat_history: List[dict] = None) -> RouteResult:
        """
        核心路由判定方法
        """
        history_text = ""
        if chat_history:
            # 【优化点2】：优化历史记录的拼接格式，使得大模型更容易理解对话上下文
            recent_history = chat_history[-self.history_turns :]
            history_text = "\n".join([
                f"{'用户' if msg['role'] == 'user' else 'AI助理'}: {msg['content']}" 
                for msg in recent_history
            ])

        prompt = build_query_router_user_prompt(user_query=user_query, history_text=history_text)

        try:
            response = ModelFactory.chat_completion(
                messages=[
                    {"role": "system", "content": QUERY_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model_tier=self.model_tier,
                temperature=0.0,
                max_tokens=16248,
                extra_body={"response_format": {"type": "json_object"}},
                enable_thinking=False,
            )

            raw_json_str = response.choices[0].message.content
            if not isinstance(raw_json_str, str):
                raw_json_str = str(raw_json_str)
            raw_json_str = self._extract_json_block(raw_json_str)
            result_dict = json.loads(raw_json_str)
            
            route_result = RouteResult(**result_dict)
            logging.info(f"🚦 路由判定 -> 模式: [{route_result.mode}], 实体: {route_result.entities}")
            return route_result
            
        except Exception as e:
            logging.error(f"⚠️ Router 路由失败，启动安全降级策略... 错误: {e}")
            # 降级策略：一旦发生断网或大模型解析崩溃，默认走 LOCAL 检索，防止业务中断
            return RouteResult(
                reasoning="系统异常降级，默认进入局部检索",
                mode="LOCAL",
                rewritten_query=user_query,
                entities=[]
            )

# ==========================================
# 独立测试入口
# ==========================================
if __name__ == "__main__":
    router = QueryRouter()
    
    # 模拟场景 1: 带有代词的局部检索
    history = [
        {"role": "user", "content": "罗峰加入极限武馆了吗？"}, 
        {"role": "assistant", "content": "是的，他通过了准武者考核并加入了极限武馆。"}
    ]
    q1 = "那他在里面学了什么刀法？"
    res1 = router.route_query(q1, history)
    print(f"\n👱 历史上下文: {history}")
    print(f"👱 最新提问: {q1}")
    print(f"🤖 路由结果: \n{json.dumps(res1.model_dump(), indent=2, ensure_ascii=False)}")
    
    # 模拟场景 2: 宏观全局检索
    q2 = "地球篇的人类势力是怎么划分的？"
    res2 = router.route_query(q2)
    print(f"\n👱 最新提问: {q2}")
    print(f"🤖 路由结果: \n{json.dumps(res2.model_dump(), indent=2, ensure_ascii=False)}")