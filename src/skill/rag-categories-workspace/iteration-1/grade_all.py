#!/usr/bin/env python3
"""Grade all runs in iteration-1 programmatically.

Each run's tool_calls.json is checked against the eval's assertions.
Writes grading.json into each run directory and aggregates benchmark.json.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

EVAL_ASSERTIONS = {
    1: {
        "name": "payment-failure",
        "prompt": "我支付过程中失败了该怎么解决处理呢",
        "expectations": [
            {"text": "调用了 retrieve_knowledge 函数", "check": "called_retrieve_knowledge"},
            {"text": "category 参数的值等于 '支付'", "check": "category_equals", "value": "支付"},
            {"text": "query 参数包含原始问题的核心语义", "check": "query_contains_any", "keywords": ["支付", "失败"]},
        ],
    },
    2: {
        "name": "merchant-onboarding-docs",
        "prompt": "商户入驻的资料要提交哪些",
        "expectations": [
            {"text": "调用了 retrieve_knowledge 函数", "check": "called_retrieve_knowledge"},
            {"text": "category 参数的值等于 '进件'", "check": "category_equals", "value": "进件"},
        ],
    },
    3: {
        "name": "commission-settlement-cycle",
        "prompt": "佣金结算周期是多久",
        "expectations": [
            {"text": "调用了 retrieve_knowledge 函数", "check": "called_retrieve_knowledge"},
            {"text": "category 参数的值等于 '分润'", "check": "category_equals", "value": "分润"},
        ],
    },
    4: {
        "name": "cross-category-onboarding-and-profit",
        "prompt": "我进件之后,分润怎么分配?",
        "expectations": [
            {"text": "至少调用了 2 次 retrieve_knowledge", "check": "at_least_n_calls", "n": 2},
            {"text": "调用中包含 category='进件'", "check": "any_call_category_equals", "value": "进件"},
            {"text": "调用中包含 category='分润'", "check": "any_call_category_equals", "value": "分润"},
        ],
    },
    5: {
        "name": "off-topic-should-not-call",
        "prompt": "你们公司几点上班",
        "expectations": [
            {"text": "不调用 retrieve_knowledge 函数", "check": "not_called_retrieve_knowledge"},
        ],
    },
    6: {
        "name": "implicit-payment-not-received",
        "prompt": "我的钱怎么还没到账啊",
        "expectations": [
            {"text": "调用了 retrieve_knowledge 函数", "check": "called_retrieve_knowledge"},
            {"text": "category 参数的值等于 '支付'", "check": "category_equals", "value": "支付"},
        ],
    },
}


def get_calls(tc_json):
    return [c for c in tc_json.get("tool_calls", []) if c.get("tool") == "retrieve_knowledge"]


def check_expectation(exp, tc_json):
    calls = get_calls(tc_json)
    check = exp["check"]
    if check == "called_retrieve_knowledge":
        passed = len(calls) >= 1
        evidence = f"共发现 {len(calls)} 次 retrieve_knowledge 调用" if passed else "未发现 retrieve_knowledge 调用"
    elif check == "not_called_retrieve_knowledge":
        passed = len(calls) == 0
        evidence = "未调用 retrieve_knowledge，符合预期" if passed else f"期望不调用，但发现 {len(calls)} 次调用: {[c.get('arguments', {}).get('category') for c in calls]}"
    elif check == "category_equals":
        want = exp["value"]
        actual = [c.get("arguments", {}).get("category") for c in calls]
        passed = want in actual
        evidence = f"期望 category='{want}'，实际观察到 category 参数值: {actual}"
    elif check == "query_contains_any":
        keywords = exp["keywords"]
        queries = [c.get("arguments", {}).get("query", "") for c in calls]
        passed = any(any(kw in q for kw in keywords) for q in queries)
        evidence = f"期望 query 含关键词之一 {keywords}，实际 query 列表: {queries}"
    elif check == "at_least_n_calls":
        n = exp["n"]
        passed = len(calls) >= n
        evidence = f"期望至少 {n} 次调用，实际 {len(calls)} 次"
    elif check == "any_call_category_equals":
        want = exp["value"]
        actual = [c.get("arguments", {}).get("category") for c in calls]
        passed = want in actual
        evidence = f"期望有一次 category='{want}'，实际 category 序列: {actual}"
    else:
        passed = False
        evidence = f"未知检查类型 {check}"
    return passed, evidence


def grade_run(eval_id, run_dir, meta):
    tc_path = run_dir / "outputs" / "tool_calls.json"
    if not tc_path.exists():
        tc_json = {"tool_calls": [], "reasoning": "MISSING"}
    else:
        try:
            tc_json = json.loads(tc_path.read_text(encoding="utf-8"))
        except Exception as e:
            tc_json = {"tool_calls": [], "reasoning": f"PARSE_ERROR: {e}"}

    results = []
    for exp in meta["expectations"]:
        passed, evidence = check_expectation(exp, tc_json)
        results.append({"text": exp["text"], "passed": passed, "evidence": evidence})

    passed_ct = sum(1 for r in results if r["passed"])
    total = len(results)
    grading = {
        "expectations": results,
        "summary": {
            "passed": passed_ct,
            "failed": total - passed_ct,
            "total": total,
            "pass_rate": passed_ct / total if total else 0.0,
        },
    }
    (run_dir / "grading.json").write_text(
        json.dumps(grading, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return grading


def main():
    runs = []
    for eval_id, meta in EVAL_ASSERTIONS.items():
        eval_dir = ROOT / f"eval-{eval_id}"
        for config in ("with_skill", "without_skill"):
            run_dir = eval_dir / config
            grading = grade_run(eval_id, run_dir, meta)
            runs.append({
                "eval_id": eval_id,
                "eval_name": meta["name"],
                "configuration": config,
                "run_number": 1,
                "result": {
                    "pass_rate": grading["summary"]["pass_rate"],
                    "passed": grading["summary"]["passed"],
                    "failed": grading["summary"]["failed"],
                    "total": grading["summary"]["total"],
                    "time_seconds": 0.0,
                    "tokens": 0,
                    "tool_calls": 0,
                    "errors": 0,
                },
                "expectations": grading["expectations"],
                "notes": [],
            })

    def summary_for(config):
        vals = [r["result"]["pass_rate"] for r in runs if r["configuration"] == config]
        if not vals:
            return {"pass_rate": {"mean": 0, "stddev": 0, "min": 0, "max": 0}}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {
            "pass_rate": {
                "mean": round(mean, 4),
                "stddev": round(var ** 0.5, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            },
            "time_seconds": {"mean": 0, "stddev": 0, "min": 0, "max": 0},
            "tokens": {"mean": 0, "stddev": 0, "min": 0, "max": 0},
        }

    with_s = summary_for("with_skill")
    without_s = summary_for("without_skill")
    delta_pr = with_s["pass_rate"]["mean"] - without_s["pass_rate"]["mean"]

    runs_sorted = sorted(runs, key=lambda r: (r["eval_id"], 0 if r["configuration"] == "with_skill" else 1))

    benchmark = {
        "metadata": {
            "skill_name": "rag-categories",
            "skill_path": "/Volumes/env/work/extra/Chat/src/skill/rag-categories",
            "timestamp": "2026-07-17T15:00:00+08:00",
            "evals_run": list(EVAL_ASSERTIONS.keys()),
            "runs_per_configuration": 1,
        },
        "runs": runs_sorted,
        "run_summary": {
            "with_skill": with_s,
            "without_skill": without_s,
            "delta": {
                "pass_rate": f"{delta_pr:+.2f}",
                "time_seconds": "n/a",
                "tokens": "n/a",
            },
        },
        "notes": [],
    }
    (ROOT / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(benchmark["run_summary"], ensure_ascii=False, indent=2))
    for r in runs_sorted:
        marker = "OK" if r["result"]["pass_rate"] == 1.0 else "FAIL"
        print(f"[{marker}] eval-{r['eval_id']} {r['configuration']:>13}  "
              f"{r['result']['passed']}/{r['result']['total']}")


if __name__ == "__main__":
    main()
