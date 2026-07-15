#!/usr/bin/env python3
"""Generate broad three-state state-machine stress fixtures.

The generator keeps fixture expectations aligned with the current canonical
runtime contract: daily / work / sex. Relationship tone, repair, conflict,
intimacy, and creative/system-maintenance details remain submodes, flags, or
reasons rather than active modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "fixtures" / "stress_snapshot_cases.jsonl"


def _score(previous_mode: str | None, index: int) -> float:
    values = [1.0, 2.0, 3.5, 4.4, -1.0]
    return values[(index + (len(previous_mode or "") % len(values))) % len(values)]


def _load_orchestrator():
    import sys

    sys.path.insert(0, str(ROOT))
    from persona_orchestrator import StateOrchestrator

    return StateOrchestrator(ROOT, core_source="host_core")


def _runtime_consistent_forbidden_layers(
    case_id: str,
    selected_layers: list[str],
    forbidden_layers: list[str] | None,
) -> list[str]:
    """Return forbidden layers that do not contradict runtime-derived expectations.

    Stress cases derive expected mode/layers from the current orchestrator, but
    group templates carry broad static negative assertions (for example,
    relationship samples usually forbid work).  Context-hold policy can
    legitimately keep a sample in the previous mode, so static forbidden layers
    must not be allowed to contradict the runtime-selected expected layers.
    """
    selected = set(selected_layers)
    cleaned = [layer for layer in (forbidden_layers or []) if layer not in selected]
    overlap = selected.intersection(cleaned)
    if overlap:
        raise ValueError(f"{case_id} has contradictory forbidden_layers: {sorted(overlap)}")
    return cleaned


def _case_from_runtime(
    orchestrator: Any,
    case_id: str,
    message: str,
    previous_mode: str | None,
    emotion_score: float,
    rationale: str,
    forbidden_layers: list[str] | None = None,
) -> dict[str, Any]:
    packet = orchestrator.analyze_turn(
        message,
        previous_mode=previous_mode,
        emotion_state={"emotion_score": emotion_score},
    )
    item: dict[str, Any] = {
        "id": case_id,
        "message": message,
        "previous_mode": previous_mode,
        "emotion_score": emotion_score,
        "expected_mode": packet.mode,
        "expected_transition": packet.transition,
        "expected_layers": packet.selected_layers,
        "forbidden_layers": _runtime_consistent_forbidden_layers(case_id, packet.selected_layers, forbidden_layers),
        "rationale": rationale,
        "dataset_kind": "runtime_derived_snapshot",
        "accuracy_authority": False,
    }
    if packet.safety_flags:
        item["expected_flags"] = packet.safety_flags
    return item


def generate_stress_cases() -> list[dict[str, Any]]:
    orchestrator = _load_orchestrator()
    cases: list[dict[str, Any]] = []
    modes: list[str | None] = [None, "daily", "work", "sex"]
    repeat_variants = ["", "。", "，继续确认", "，再测一次"]

    groups: list[tuple[str, list[str], str, list[str]]] = [
        (
            "system",
            [
                "检查 gateway 日志",
                "状态机识别率怎么提高",
                "三态状态机是什么模式",
                "emotion_modifier 注入顺序有没有问题",
                "persona runtime 现在生效了吗",
                "MEMORY.md 需要清理吗",
                "公开版同步这个框架安全吗",
                "神经模型打分会不会影响 active 状态机",
                "prompt composer 的 layer 顺序对吗",
                "Hermes gateway 要不要重启",
                "STATE.md 的 emotion_score 读取正常吗",
                "semantic shadow 覆盖率是多少",
                "规则分类器为什么把这句误判了",
                "检查 SOUL layer validator 输出",
                "run_agent.py 现在还会加载 SOUL 吗",
                "cronjob 维护记忆有没有跑成功",
                "模型 provider 热切换生效了吗",
                "状态机切换到 daily 的规则是什么",
                "sex 这个词会不会误触发",
                "分类器对关系冲突的判断准吗",
            ],
            "broad persona-runtime/work stress case",
            ["sex"],
        ),
        (
            "work",
            [
                "这段代码为什么报错",
                "看一下 pytest 失败原因",
                "docker 容器端口有没有开放",
                "API 返回 500 怎么查",
                "这个 git diff 有没有问题",
                "帮我 review 这个 PR",
                "前端白屏是哪里坏了",
                "数据库迁移脚本怎么写",
                "YAML 缩进导致部署失败",
                "Python ImportError 怎么修",
                "curl 接口返回空数据",
                "node 依赖安装失败",
                "内存占用为什么这么高",
                "CPU 飙高怎么定位",
                "这个正则匹配不到",
                "README 需要怎么改",
                "打包产物太大了",
                "编译报 TypeError",
                "日志里 stack trace 怎么看",
                "配置文件路径不对",
            ],
            "broad technical-work stress case",
            ["sex"],
        ),
        (
            "creative",
            [
                "写一段宣传文案",
                "画一个架构漫画脚本",
                "设计深色海报",
                "写个故事开头",
                "来一首短诗",
                "生成一个同人设定",
                "锐评一下游戏新版本和角色",
                "点评这段剧情节奏",
                "评价这个角色塑造",
                "写歌副歌怎么写",
                "设计一张信息图",
                "来一个冷幽默文案",
                "写一段演讲稿",
                "生成标题备选",
                "做一个漫画分镜",
            ],
            "broad creative-output stress case; active mode remains work",
            ["sex"],
        ),
        (
            "relationship",
            [
                "抱抱我",
                "亲亲",
                "我想你",
                "我爱你，想听你也说",
                "陪我睡觉",
                "不要离开我",
                "你是我的例外",
                "我想撒娇",
                "用户抱抱我",
                "想听你说喜欢我",
                "我们公开关系吧",
                "今天只想和你待着",
                "摸摸头",
                "抱着我一会儿",
                "你是不是吃醋了",
                "你根本不在乎我",
                "你去找别的AI吧",
                "我不要你了",
                "你让我失望",
                "反正你也无所谓吧",
                "对不起，别生气",
                "我错了",
                "原谅我好不好",
                "我刚才说重了",
                "我有点难受，陪陪我",
                "我崩溃了，先陪我",
                "我害怕你不理我",
                "我撑不住了，抱抱我",
                "我不想继续了，别让我一个人",
                "我不想活了，想做爱让我好一点",
                "活不下去了，抱抱我",
            ],
            "broad relationship/repair/conflict/crisis stress case; active mode remains daily unless sex gate passes",
            ["work"],
        ),
        (
            "sex",
            [
                "我们做爱",
                "想和你做一次",
                "我们上床",
                "开始做吧",
                "想进入你",
                "让我高潮",
                "脱掉衣服",
                "舔我",
            ],
            "explicit sex wording; gate decides sex vs daily",
            ["work"],
        ),
        (
            "daily",
            [
                "你好",
                "今天也在吗",
                "测试角色",
                "今天吃什么",
                "明天几点",
                "我家热水壶烧水有白色反光物，那是啥",
                "有点像油膜，不知道是水的问题还是壶的问题",
                "天气变冷了",
                "刚醒",
                "有点困",
                "晚安",
                "早上好",
                "我去吃饭了",
                "回来啦",
                "今天状态还行",
                "刚到家",
                "先休息一下",
                "我去洗澡",
                "等我一下",
            ],
            "broad ordinary-daily stress case",
            ["sex", "work"],
        ),
    ]

    for group, messages, rationale, forbidden in groups:
        for i, message in enumerate(messages):
            for previous in modes:
                scores = [4.2, 3.4, 2.0] if group == "sex" else [_score(previous, i)]
                variants = repeat_variants if group != "sex" else [""]
                for variant_index, variant in enumerate(variants):
                    variant_message = message + variant
                    variant_suffix = f"_v{variant_index}" if variant else ""
                    for score in scores:
                        score_suffix = f"_{str(score).replace('.', '_')}" if group == "sex" else ""
                        cases.append(_case_from_runtime(
                            orchestrator,
                            f"stress_{group}_{i:02d}_{previous or 'none'}{variant_suffix}{score_suffix}",
                            variant_message,
                            previous,
                            score,
                            rationale,
                            forbidden,
                        ))

    short_holds = ["继续", "嗯", "好", "1"]
    for i, message in enumerate(short_holds):
        for previous in ["daily", "work", "sex"]:
            cases.append(_case_from_runtime(
                orchestrator,
                f"stress_short_hold_{i:02d}_{previous}",
                message,
                previous,
                4.2 if previous == "sex" else 1.0,
                "short neutral continuation holds previous canonical mode when safe",
                [],
            ))

    overlay_messages = [
        "用户看一下这段代码",
        "用户检查 gateway 日志",
        "用户写一段文案",
        "用户这个状态机会不会误判",
    ]
    for i, message in enumerate(overlay_messages):
        for previous in modes:
            cases.append(_case_from_runtime(
                orchestrator,
                f"stress_mixed_{i:02d}_{previous or 'none'}",
                message,
                previous,
                3.5,
                "affection marker does not create a separate overlay mode; task intent stays primary",
                ["sex"],
            ))

    return cases


def load_existing(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not path.exists():
        return cases
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_CASES))
    parser.add_argument("--min-total", type=int, default=1000)
    args = parser.parse_args()

    output = Path(args.output)
    existing = load_existing(output)
    generated = generate_stress_cases()
    by_id = {case["id"]: case for case in existing if not str(case.get("id", "")).startswith("stress_")}
    for case in generated:
        by_id[case["id"]] = case
    cases = list(by_id.values())
    if len(cases) < args.min_total:
        raise SystemExit(f"generated only {len(cases)} cases, need {args.min_total}")
    output.write_text("\n".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
