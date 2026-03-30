import argparse

from src.memory.tools import MemoryManager


def main():
    parser = argparse.ArgumentParser(description="删除用户记忆工具")
    parser.add_argument(
        "--target",
        choices=["1", "2", "12", "21", "kv", "db", "all"],
        default="2",
        help="1/kv: 清空本地 KV Profile；2/db: 清理 [当前用户] 图谱边并重建记忆摘要库；12/all: 同时执行两者",
    )
    args = parser.parse_args()

    manager = MemoryManager()
    try:
        if args.target in {"1", "kv"}:
            print(manager.clear_kv_profile())
        elif args.target in {"12", "21", "all"}:
            print(manager.clear_all_memories())
        else:
            print(manager.clear_user_graph_and_memory_db())
    finally:
        manager.close()


if __name__ == "__main__":
    main()
