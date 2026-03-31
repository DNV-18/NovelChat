import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

try:
    from prompt_toolkit import prompt as pt_prompt
except Exception:
    pt_prompt = None

# 假设已导入你的模块
from src.config import settings
from src.utils.model_factory import ModelFactory
from langsmith import traceable
from src.utils.prompts import build_main_agent_system_prompt
from src.agent.query_router import QueryRouter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.memory.tools import AGENT_MEMORY_TOOLS, MemoryManager, get_current_kv_profile

class NovelAgent:
    """
    系统的大脑：统领路由、检索、以及带着 Tools 进行记忆读写与最终回答。
    """
    def __init__(self):
        # 初始化组件
        self.router = QueryRouter()
        # 先初始化 MemoryManager，确保 Milvus 记忆集合存在，再初始化检索器
        self.memory_manager = MemoryManager()
        self.retriever = HybridRetriever()
        # 记忆写入与工具执行改为后台线程，避免阻塞主回复
        self._memory_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-bg")
        self._memory_lock = threading.RLock()

    @staticmethod
    def _prepare_recent_history(chat_history: List[Dict], max_rounds: int = 2) -> List[Dict]:
        """仅保留最近 max_rounds 轮对话，并清洗为标准 user/assistant 消息。"""
        if not chat_history:
            return []

        max_messages = max(1, max_rounds) * 2
        recent = chat_history[-max_messages:]

        cleaned = []
        for msg in recent:
            role = (msg.get("role") or "").strip()
            content = str(msg.get("content") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def _build_messages(self, system_prompt: str, chat_history: List[Dict], user_message: str) -> List[Dict]:
        """构造发送给主模型的消息列表：system + 最近两轮历史 + 当前用户输入。"""
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._prepare_recent_history(chat_history, max_rounds=2))
        messages.append({"role": "user", "content": (user_message or "").strip()})
        return messages

    def _execute_tool_calls(self, tool_calls, memory_event_id: str = "") -> List[str]:
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
            elif function_name == "add_graph_memory":
                res = self.memory_manager.add_graph_memory(
                    arguments["source_entity"],
                    arguments["target_entity"],
                    arguments["relation"],
                    memory_evidence_ref=memory_event_id,
                )
            else:
                res = f"未知工具: {function_name}"
            
            results.append(res)
        return results

    def _persist_and_execute_tools_async(self, rewritten_query: str, final_response: str, tool_calls):
        """后台静默执行：先写长期记忆，再执行工具调用。"""
        try:
            with self._memory_lock:
                memory_result = self.memory_manager.save_chat_turn_to_memory_record(
                    rewritten_query=rewritten_query,
                    ai_response=final_response,
                )
            memory_event_id = memory_result.get("event_id", "")
            print(f"📝 [自动记忆] {memory_result.get('message', '')}")

            if tool_calls:
                with self._memory_lock:
                    self._execute_tool_calls(tool_calls, memory_event_id=memory_event_id)
        except Exception as e:
            print(f"⚠️ [后台记忆] 执行失败: {e}")

    @traceable(name="novel_agent_chat", run_type="chain")
    def chat(self, user_message: str, chat_history: List[Dict] = None):
        """
        一次完整的问答生命周期
        """
        chat_history = chat_history or []

        # =======================================================
        # 1. 极速路由判定 (Router)
        # =======================================================
        route_result = self.router.route_query(user_message, chat_history)

        # =======================================================
        # 2. 知识召回 (Retrieval) - 只读检索，不做写入
        # =======================================================
        try:
            retrieved_context = self.retriever.execute_retrieval(
                mode=route_result.mode,
                rewritten_query=route_result.rewritten_query,
                entities=route_result.entities,
            )
        except Exception as e:
            print(f"⚠️ [检索降级] execute_retrieval 失败: {e}")
            retrieved_context = ""

        # =======================================================
        # 3. 构建 System Prompt (静默注入当前 KV 记忆)
        # =======================================================
        current_kv = get_current_kv_profile()
        system_prompt = build_main_agent_system_prompt(
            current_kv=current_kv,
            retrieved_context=retrieved_context,
        )

        messages = self._build_messages(
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_message=user_message,
        )

        # =======================================================
        # 4. 召唤主控大模型，并赋予它 Tools 的能力！
        # =======================================================
        print("🧠 主控大模型正在思考与生成...")
        response = ModelFactory.chat_completion(
            messages=messages,
            model_tier=settings.main_agent_model_tier,
            tools=AGENT_MEMORY_TOOLS,
            tool_choice="auto", # 让大模型自己决定用不用工具
            max_tokens=26248,
            temperature=0.7,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []

        # 先确定主回复文本
        final_response = (response_message.content or "").strip()
        if not final_response and tool_calls:
            final_response = "好的，我已经记下你的设定位了！还有什么想了解的吗？"

        # 5. 后台静默更新记忆与图谱，不阻塞主回复
        self._memory_executor.submit(
            self._persist_and_execute_tools_async,
            route_result.rewritten_query,
            final_response,
            tool_calls,
        )

        return final_response

    def close(self):
        """释放外部资源连接。"""
        try:
            self.retriever.close()
        except Exception:
            pass
        try:
            self.memory_manager.close()
        except Exception:
            pass
        try:
            self._memory_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass


_prompt_toolkit_broken = False


def read_user_input(prompt_text: str) -> str:
    """优先使用 prompt_toolkit；异常时自动降级到内置 input，避免 CLI 崩溃。"""
    global _prompt_toolkit_broken
    if pt_prompt is not None and not _prompt_toolkit_broken:
        try:
            return pt_prompt(prompt_text)
        except Exception as e:
            _prompt_toolkit_broken = True
            print(f"\n⚠️ prompt_toolkit 不可用，已降级为 input()。原因: {e}")
    return input(prompt_text)


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    agent = NovelAgent()
    chat_history: List[Dict] = []
    print("\n🤖 NovelAgent 已启动，输入内容开始对话；输入 exit/quit/退出 结束。")

    try:
        while True:
            user_say = read_user_input("\n👱 用户: ").strip()
            if not user_say:
                continue
            if user_say.lower() in {"exit", "quit", "q"} or user_say in {"退出"}:
                print("\n👋 会话结束。")
                break

            if user_say == "删除记忆":
                print("\n请选择删除类型：")
                print("1. KV profile")
                print("2. 记忆数据库（删除 [当前用户] 出发的图谱边 + 重建记忆摘要集合）")
                print("12. 同时删除 1 和 2")
                choice = read_user_input("请输入 1 / 2 / 12: ").strip()

                if choice == "1":
                    result = agent.memory_manager.clear_kv_profile()
                    print(f"\n🧹 {result}")
                elif choice == "2":
                    result = agent.memory_manager.clear_user_graph_and_memory_db()
                    print(f"\n🧹 {result}")
                elif choice in {"12", "21"}:
                    result = agent.memory_manager.clear_all_memories()
                    print(f"\n🧹 {result}")
                else:
                    print("\n⚠️ 无效选项，已取消删除操作。")
                continue

            reply = agent.chat(user_say, chat_history=chat_history)
            print(f"\n🤖 助理:\n {reply}")

            # 会话历史持续累积，供后续轮次路由与生成使用
            chat_history.append({"role": "user", "content": user_say})
            chat_history.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n\n👋 会话被中断，正在退出。")
    finally:
        agent.close()