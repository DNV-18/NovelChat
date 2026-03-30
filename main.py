import streamlit as st
from typing import List, Dict

# 导入你刚刚写好的终极大脑
from src.agent.main_agent import NovelAgent
from src.config import settings

# ==========================================
# 1. 页面配置与样式初始化
# ==========================================
st.set_page_config(
    page_title="吞噬星空 - 超级知识体",
    page_icon="🌌",
    layout="centered"
)

st.title("🌌 吞噬星空 Agentic-RAG")
st.caption("集成了 GraphRAG 宏观视野与长期主动记忆的专属大模型")

# ==========================================
# 2. 全局状态与 Agent 单例挂载
# ==========================================
# 使用 cache_resource 确保 Agent（及其包含的数据库连接）在刷新网页时不会被重复创建
@st.cache_resource
def get_agent() -> NovelAgent:
    return NovelAgent()

agent = get_agent()

# 初始化 Streamlit session 状态中的聊天历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 3. 侧边栏：记忆管理面板 (可选高级功能)
# ==========================================
with st.sidebar:
    st.header("🧠 记忆控制台")
    st.markdown("在这里管理你的私有知识域。")

    if st.button("🧹 清空 KV Profile", type="secondary", use_container_width=True):
        with st.spinner("正在清空 KV Profile..."):
            res = agent.memory_manager.clear_kv_profile()
            st.success(res)

    if st.button("🧹 清空记忆数据库 (图谱边 + 摘要库)", type="secondary", use_container_width=True):
        with st.spinner("正在抹除当前用户的所有私有记忆..."):
            res = agent.memory_manager.clear_user_graph_and_memory_db()
            # 清空网页的聊天历史
            st.session_state.chat_history = []
            st.success(res)
            st.rerun() # 强制刷新页面

    if st.button("🧨 清空全部记忆 (KV + 事件 + 关系)", type="secondary", use_container_width=True):
        with st.spinner("正在抹除当前用户的所有私有记忆..."):
            res = agent.memory_manager.clear_all_memories()
            # 清空网页的聊天历史
            st.session_state.chat_history = []
            st.success(res)
            st.rerun() # 强制刷新页面

    st.divider()
    st.markdown("### 📊 系统状态")
    st.markdown(f"- **Router Tier**: {settings.query_router_model_tier}")
    st.markdown(f"- **Main Agent Tier**: {settings.main_agent_model_tier}")
    st.markdown(f"- **Main Model**: {settings.smart_llm_model if settings.main_agent_model_tier == 'smart' else settings.cheap_llm_model}")
    st.markdown(f"- **Retriever**: Milvus + Neo4j (RRF-k={settings.rrf_k}, 阈值={settings.rerank_threshold})")

# ==========================================
# 4. 渲染历史聊天记录
# ==========================================
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👱"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

# ==========================================
# 5. 处理用户新输入
# ==========================================
if user_input := st.chat_input("向我提问，或告诉我你的设定 (例如: 以后请叫我‘领主’)"):
    
    # a. 将用户输入显示在界面上
    with st.chat_message("user", avatar="👱"):
        st.markdown(user_input)

    # b. 调用 Agent 获取回答（带有加载动画）
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("思考与检索中... (正在判定路由与拉取数据)"):
            try:
                # 调用你写好的主控逻辑
                reply = agent.chat(user_input, chat_history=st.session_state.chat_history)
                st.markdown(reply)
                
                # c. 将本次对话追加到历史记录中
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"系统异常: {str(e)}")