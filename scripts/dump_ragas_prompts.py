import json
from datetime import datetime
from pathlib import Path


def _try_adapt(metric_obj, lang: str = "zh") -> str:
    adapt_fn = getattr(metric_obj, "adapt", None)
    if not callable(adapt_fn):
        return "skip: adapt() not available"
    try:
        adapt_fn(lang, cache_config=True)
        return f"ok: adapt('{lang}')"
    except Exception as e:
        return f"skip: adapt failed ({e})"


def _extract_instructions(metric_obj):
    prompts = metric_obj.get_prompts()
    out = []
    if isinstance(prompts, dict):
        for name, prompt_obj in prompts.items():
            out.append(
                {
                    "prompt_name": name,
                    "instruction": getattr(prompt_obj, "instruction", ""),
                }
            )
    else:
        out.append(
            {
                "prompt_name": "unknown",
                "instruction": str(prompts),
            }
        )
    return out


def main():
    # 兼容你当前环境：该导入在当前 ragas 版本可拿到 metric 实例
    from ragas.metrics import faithfulness, context_precision

    records = []
    for metric_obj, metric_name in (
        (faithfulness, "faithfulness"),
        (context_precision, "context_precision"),
    ):
        adapt_status = _try_adapt(metric_obj, lang="zh")
        instructions = _extract_instructions(metric_obj)
        records.append(
            {
                "metric": metric_name,
                "adapt": adapt_status,
                "prompts": instructions,
            }
        )

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }

    out_file = Path(__file__).resolve().parent / "ragas_prompts_dump.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已写入: {out_file}")


if __name__ == "__main__":
    main()
