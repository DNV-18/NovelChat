import json
from typing import List, Dict

# 假设已导入你的模块
from src.config import settings
from src.utils.model_factory import ModelFactory
from src.agent.query_router import QueryRouter
from src.memory.tools import AGENT_MEMORY_TOOLS, MemoryManager, get_current_kv_profile

class NovelAgent:
    """
    系统的大脑：统领路由、检索、以及带着 Tools 进行记忆读写与最终回答。
    """
    def __init__(self):
        # 初始化模型
        self.smart_llm = ModelFactory.get_llm(model_tier="smart")
        self.cheap_llm = ModelFactory.get_llm(model_tier="cheap")
        
        # 初始化组件
        self.router = QueryRouter()
        self.memory_manager = MemoryManager()
        
    def _execute_tool_calls(self, tool_calls) -> List[str]:
        """
        后台静默执行大模型下发的记忆工具调用
        """
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"\n⚙️ [后台动作] 大模型触发了记忆操作: {function_name}({arguments})")
            
            if function_name == "update_kv_profile":
                res = self.memory_manager.update_kv_profile(arguments["key"], arguments["value"])
            elif function_name == "save_event_to_long_term":
                res = self.memory_manager.save_event_to_long_term(arguments["event_description"])
            elif function_name == "add_graph_memory":
                res = self.memory_manager.add_graph_memory(arguments["source_entity"], arguments["target_entity"], arguments["relation"])
            else:
                res = f"未知工具: {function_name}"
            
            results.append(res)
        return results

    def chat(self, user_message: str, chat_history: List[Dict] = None):
        """
        一次完整的问答生命周期
        """
        if chat_history is None:
            chat_history = []

        # =======================================================
        # 1. 极速路由判定 (Router)
        # =======================================================
        print(f"🚦 正在路由用户请求...")
        route_result = self.router.route_query(user_message, chat_history)
        print(f"   -> 模式: {route_result.mode} | 改写后: {route_result.rewritten_query}")

        # =======================================================
        # 2. 知识召回 (Retrieval) - 这里只有读，没有写！
        # =======================================================
        retrieved_context = ""
        if route_result.mode == "LOCAL":
            # TODO: 去查 Neo4j 实体 和 Milvus 对应 Chunk
            retrieved_context = "模拟检索到的局部微观证据..."
        elif route_result.mode == "GLOBAL":
            # TODO: 去查 Milvus 里的 Community Summaries
            retrieved_context = "模拟检索到的宏观社区摘要..."
        elif route_result.mode == "MEMORY":
            # TODO: 去查 Milvus 里的 long_term_events
            retrieved_context = "模拟检索到的用户过往长期记忆..."
        # DIRECT 模式不需要查数据库

        # =======================================================
        # 3. 构建 System Prompt (静默注入当前 KV 记忆)
        # =======================================================
        current_kv = get_current_kv_profile()
        system_prompt = f"""你是一个科幻小说《吞噬星空》的专属超级 AI 助理。

【用户专属偏好设定（必须遵守）】：
{current_kv}

【检索到的参考资料】：
{retrieved_context if retrieved_context else "无"}

请根据以上参考资料和你的知识回答用户。如果用户刚才的话语中透露了新的长期设定、偏好、或者重要事实，请调用相应的工具（Tools）保存它们！"""

        messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_message}]

        # =======================================================
        # 4. 召唤主控大模型，并赋予它 Tools 的能力！
        # =======================================================
        print("🧠 主控大模型正在思考与生成...")
        response = self.smart_llm.chat.completions.create(
            model=getattr(self.smart_llm, "default_model", "gpt-4o"),
            messages=messages,
            # 【核心秘密】：在这里把你的那 3 个记忆工具挂给大模型！
            tools=AGENT_MEMORY_TOOLS,
            tool_choice="auto", # 让大模型自己决定用不用工具
            temperature=0.7
        )

        response_message = response.choices[0].message

        # 检查大模型是否觉得需要“记笔记”
        if response_message.tool_calls:
            # 大模型决定调用工具记录记忆
            # 在实际的 Web 服务中，这一步可以通过 Celery 或 Async 放到后台执行，不阻塞用户响应
            self._execute_tool_calls(response_message.tool_calls)
            
            # OpenAI 的机制：如果它调了工具，它可能把话放在内容里，也可能没有内容。
            # 为了简单展示，我们只返回其内容（如果有的话）
            if response_message.content:
                return response_message.content
            else:
                return "好的，我已经记下你的设定位了！还有什么想了解的吗？"
        else:
            # 大模型觉得这句话不值得记录，只是正常的问答
            return response_message.content


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    agent = NovelAgent()
    
    # 测试场景：用户突然说了一句设定
    user_say = "以后请叫我‘地球领主’，另外，我特别讨厌别人剧透大结局。"
    print(f"\n👱 用户: {user_say}")
    reply = agent.chat(user_say)
    print(f"\n🤖 助理: {reply}")
    
    # 你会看到控制台打印：
    # ⚙️ [后台动作] 大模型触发了记忆操作: update_kv_profile({'key': '称呼', 'value': '地球领主'})
    # ⚙️ [后台动作] 大模型触发了记忆操作: update_kv_profile({'key': '忌讳', 'value': '讨厌被剧透大结局'})