# -*- coding: utf-8 -*-
"""Fixed-seed AB/BA benchmark for built-in and custom Hex AIs.

Supports basic/random/hybrid/neural/strategic AIs, memory wrappers,
custom module:Class factories, JSON/CSV reports, and per-move latency data.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hex_ai import create_ai
from hex_match import play_game

BUILTIN_AIS = ("basic", "random", "neural", "hybrid", "hybrid_llm", "strategic")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT.parent / "evaluation" / "benchmark_results")
DEFAULT_GAMES = 500
DEFAULT_SEED = 20261024
RED_LABEL = "红"
BLUE_LABEL = "蓝"


def _safe_filename(name):
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(name)).strip(" .")
    return cleaned or "benchmark"


def load_ai_factory(spec):
    """返回无参 AI 工厂。支持内置名、module:Class 和 file.py:Class。"""
    spec = str(spec).strip()
    memory_modes = {
        "memory-shadow:": "shadow",
        "memory-assist:": "assist",
        "memory:": "shadow",
    }
    for prefix, mode in memory_modes.items():
        if spec.startswith(prefix):
            base_spec = spec[len(prefix):]
            if not base_spec:
                raise ValueError("memory wrapper requires a base AI")
            base_factory = load_ai_factory(base_spec)

            def memory_factory():
                from dataclasses import replace
                from ai.memory.memory_ai import (
                    MemoryAssistedAI,
                    MemoryRetrievalSettings,
                )
                settings = replace(
                    MemoryRetrievalSettings.from_env(),
                    mode=mode,
                )
                return MemoryAssistedAI(base_factory(), settings=settings)

            return memory_factory

    if spec == "hybrid":
        def hybrid_factory():
            from ai.hybrid_ai import HybridAI
            return HybridAI()
        return hybrid_factory

    if spec == "hybrid_llm":
        def hybrid_llm_factory():
            from ai.ai_factory import create_competition_ai
            ai, error = create_competition_ai("hybrid_llm")
            if error:
                raise RuntimeError(error)
            return ai
        return hybrid_llm_factory

    if spec == "strategic" or spec.startswith("strategic:"):
        variant = spec.split(":", 1)[1] if ":" in spec else "full"
        variants = {
            "basic": dict(use_distance=False, use_double_threat=False,
                          use_bridges=False),
            "distance": dict(use_distance=True, use_double_threat=False,
                             use_bridges=False),
            "threat": dict(use_distance=True, use_double_threat=True,
                           use_bridges=False),
            "full": dict(use_distance=True, use_double_threat=True,
                         use_bridges=True),
        }
        if variant not in variants:
            raise ValueError(f"?? StrategicAI ??: {variant!r}")

        def strategic_factory():
            from ai.strategic_ai import StrategicAI
            return StrategicAI(**variants[variant])
        return strategic_factory

    if spec == "neural" or spec.startswith("neural:"):
        model_path = None
        if spec.startswith("neural:"):
            model_path = spec.split(":", 1)[1]

        def neural_factory():
            from ai.neural_ai import NeuralMCTSAI
            return NeuralMCTSAI(model_path=model_path)
        return neural_factory

    if spec in ("basic", "random"):
        return lambda: create_ai(spec)

    if ":" not in spec:
        raise ValueError(
            f"未知 AI 配置 {spec!r}；使用 basic、random 或 module:Class")

    module_ref, symbol = spec.rsplit(":", 1)
    if not module_ref or not symbol:
        raise ValueError(f"非法 AI 配置: {spec!r}")

    module_path = Path(module_ref)
    if module_ref.lower().endswith(".py") or "/" in module_ref or "\\" in module_ref:
        if not module_path.is_absolute():
            module_path = (Path.cwd() / module_path).resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"AI 模块文件不存在: {module_path}")
        digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:10]
        module_name = f"_benchmark_ai_{module_path.stem}_{digest}"
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"无法加载 AI 模块: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    target = getattr(module, symbol, None)
    if target is None:
        raise ValueError(f"模块 {module_ref!r} 中不存在 {symbol!r}")
    if not callable(target):
        raise TypeError(f"AI 入口不可调用: {spec!r}")

    def factory():
        ai = target()
        if not callable(getattr(ai, "choose_move", None)):
            raise TypeError(f"AI 对象缺少 choose_move(board, player): {spec!r}")
        return ai

    return factory


def seed_ai(ai, seed):
    """优先调用 AI 自带的 set_seed/seed，否则由全局 random.seed 控制。"""
    for method_name in ("set_seed", "seed"):
        method = getattr(ai, method_name, None)
        if callable(method):
            method(seed)
            return True
    return False


def _ai_name(ai):
    return str(getattr(ai, "name", type(ai).__name__))


def _ai_metadata(ai):
    return {
        "name": _ai_name(ai),
        "version": str(getattr(ai, "version", "")),
        "model_path": str(getattr(ai, "model_path", "")),
        "device": str(getattr(ai, "device", "")),
        "time_limit_ms": getattr(ai, "time_limit_ms", None),
    }


def _wilson_interval(wins, games, z=1.96):
    if games <= 0:
        return [None, None]
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    margin = (z * ((p * (1 - p) / games
                    + z * z / (4 * games * games)) ** 0.5) / denominator)
    return [round(max(0.0, centre - margin) * 100, 2),
            round(min(1.0, centre + margin) * 100, 2)]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _rate(wins, games):
    return round(wins / games * 100, 2) if games else 0.0


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)


def summarize_games(games, ai_a_spec, ai_b_spec):
    """根据逐局结果计算报告统计。games 中可包含异常局。"""
    completed = [game for game in games if game["status"] == "completed"]
    errors = [game for game in games if game["status"] == "error"]
    completed_count = len(completed)

    a_wins = sum(1 for game in completed if game["winner_ai"] == "A")
    b_wins = sum(1 for game in completed if game["winner_ai"] == "B")
    red_wins = sum(1 for game in completed if game["winner_color"] == RED_LABEL)
    blue_wins = sum(1 for game in completed if game["winner_color"] == BLUE_LABEL)

    a_red = [game for game in completed if game["a_color"] == RED_LABEL]
    a_blue = [game for game in completed if game["a_color"] == BLUE_LABEL]
    a_red_wins = sum(1 for game in a_red if game["winner_ai"] == "A")
    a_blue_wins = sum(1 for game in a_blue if game["winner_ai"] == "A")

    a_move_time = sum(game["a_move_ms_sum"] for game in completed)
    b_move_time = sum(game["b_move_ms_sum"] for game in completed)
    a_move_count = sum(game["a_move_count"] for game in completed)
    b_move_count = sum(game["b_move_count"] for game in completed)
    a_max = max((game["a_max_move_ms"] for game in completed), default=0.0)
    b_max = max((game["b_max_move_ms"] for game in completed), default=0.0)
    a_times = [ms for game in completed for ms in game.get("a_move_times_ms", ())]
    b_times = [ms for game in completed for ms in game.get("b_move_times_ms", ())]
    a_timeout_count = sum(game.get("a_timeout_count", 0) for game in completed)
    b_timeout_count = sum(game.get("b_timeout_count", 0) for game in completed)
    total_move_count = a_move_count + b_move_count
    illegal_move_games = sum(
        1 for game in errors if game.get("error_type") == "IllegalMoveError"
    )
    draws = sum(1 for game in completed if game.get("winner_ai") not in ("A", "B"))
    ci_low, ci_high = _wilson_interval(a_wins, completed_count)

    return {
        "ai_a": ai_a_spec,
        "ai_b": ai_b_spec,
        "games_requested": len(games),
        "games_completed": completed_count,
        "errors": len(errors),
        "error_rate": _rate(len(errors), len(games)),
        "ai_a_wins": a_wins,
        "ai_b_wins": b_wins,
        "ai_a_win_rate": _rate(a_wins, completed_count),
        "ai_b_win_rate": _rate(b_wins, completed_count),
        "ai_a_win_rate_ci95_low": ci_low,
        "ai_a_win_rate_ci95_high": ci_high,
        "ai_a_red_games": len(a_red),
        "ai_a_red_wins": a_red_wins,
        "ai_a_red_win_rate": _rate(a_red_wins, len(a_red)),
        "ai_a_blue_games": len(a_blue),
        "ai_a_blue_wins": a_blue_wins,
        "ai_a_blue_win_rate": _rate(a_blue_wins, len(a_blue)),
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "red_win_rate": _rate(red_wins, completed_count),
        "average_moves": round(_mean([game["moves"] for game in completed]), 2),
        "average_game_ms": round(_mean([game["total_ms"] for game in completed]), 2),
        "ai_a_average_move_ms": round(a_move_time / a_move_count, 3) if a_move_count else 0.0,
        "ai_a_max_move_ms": round(a_max, 3),
        "ai_a_p50_move_ms": _percentile(a_times, 0.50),
        "ai_a_p95_move_ms": _percentile(a_times, 0.95),
        "ai_a_p99_move_ms": _percentile(a_times, 0.99),
        "ai_a_timeout_count": a_timeout_count,
        "ai_b_average_move_ms": round(b_move_time / b_move_count, 3) if b_move_count else 0.0,
        "ai_b_max_move_ms": round(b_max, 3),
        "ai_b_p50_move_ms": _percentile(b_times, 0.50),
        "ai_b_p95_move_ms": _percentile(b_times, 0.95),
        "ai_b_p99_move_ms": _percentile(b_times, 0.99),
        "ai_b_timeout_count": b_timeout_count,
        "move_timeout_count": a_timeout_count + b_timeout_count,
        "move_timeout_rate": _rate(a_timeout_count + b_timeout_count, total_move_count),
        "illegal_move_games": illegal_move_games,
        "illegal_move_rate": _rate(illegal_move_games, len(games)),
        "draws": draws,
        "error_types": dict(Counter(game["error_type"] for game in errors)),
    }


def _game_row(repeat, game_index, pair_index, orientation, seed,
              ai_a, ai_b, record=None, error=None, elapsed_ms=0.0):
    row = {
        "repeat": repeat,
        "game": game_index,
        "pair": pair_index,
        "orientation": orientation,
        "seed": seed,
        "ai_a": _ai_name(ai_a),
        "ai_b": _ai_name(ai_b),
        "status": "error" if error else "completed",
        "error_type": type(error).__name__ if error else "",
        "error_message": str(error) if error else "",
        "a_color": RED_LABEL if orientation == "AB" else BLUE_LABEL,
        "b_color": BLUE_LABEL if orientation == "AB" else RED_LABEL,
        "winner_ai": "",
        "winner_color": "",
        "moves": 0,
        "total_ms": round(elapsed_ms, 2),
        "a_move_ms_sum": 0.0,
        "b_move_ms_sum": 0.0,
        "a_move_count": 0,
        "b_move_count": 0,
        "a_max_move_ms": 0.0,
        "b_max_move_ms": 0.0,
        "a_move_times_ms": [],
        "b_move_times_ms": [],
        "a_timeout_count": 0,
        "b_timeout_count": 0,
        "timeout_count": 0,
        "move_time_limit_ms": record.get("move_time_limit_ms", 500)
        if record else 500,
    }
    if error:
        return row

    winner_color = record["winner"]
    row["winner_color"] = winner_color
    row["winner_ai"] = ("A" if (
        (orientation == "AB" and winner_color == RED_LABEL)
        or (orientation == "BA" and winner_color == BLUE_LABEL)
    ) else "B")
    row["moves"] = record["moves_count"]
    row["total_ms"] = record["total_ms"]

    a_color = row["a_color"]
    for move in record["moves"]:
        move_color = move["p"]
        owner = "a" if move_color == a_color else "b"
        row[f"{owner}_move_count"] += 1
        row[f"{owner}_move_ms_sum"] += move["ms"]
        row[f"{owner}_max_move_ms"] = max(
            row[f"{owner}_max_move_ms"], move["ms"])
        row[f"{owner}_move_times_ms"].append(move["ms"])
        if move.get("timeout"):
            row[f"{owner}_timeout_count"] += 1
            row["timeout_count"] += 1
    row["a_move_ms_sum"] = round(row["a_move_ms_sum"], 3)
    row["b_move_ms_sum"] = round(row["b_move_ms_sum"], 3)
    return row


def run_benchmark(ai_a_spec, ai_b_spec, games, repeats, seed,
                  output_dir=DEFAULT_OUTPUT_DIR, name=None, quiet=False,
                  move_time_limit_ms=500):
    """运行 AB/BA 配对评测并写入 JSON、汇总 CSV 和逐局 CSV。"""
    if games <= 0 or games % 2:
        raise ValueError("games 必须是大于 0 的偶数")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")

    factory_a = load_ai_factory(ai_a_spec)
    factory_b = load_ai_factory(ai_b_spec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(name or datetime.now().strftime("%Y%m%d_%H%M%S"))

    all_games = []
    repeat_summaries = []
    metadata_a = None
    metadata_b = None
    total_started = time.perf_counter()

    for repeat_index in range(repeats):
        repeat_no = repeat_index + 1
        repeat_seed = seed + repeat_index * 1_000_003
        load_started = time.perf_counter()
        ai_a = factory_a()
        ai_a_load_ms = (time.perf_counter() - load_started) * 1000
        load_started = time.perf_counter()
        ai_b = factory_b()
        ai_b_load_ms = (time.perf_counter() - load_started) * 1000
        if metadata_a is None:
            metadata_a = _ai_metadata(ai_a)
            metadata_b = _ai_metadata(ai_b)
        repeat_games = []
        games_per_orientation = games // 2

        for pair_index in range(1, games_per_orientation + 1):
            pair_seed = repeat_seed + pair_index
            for orientation in ("AB", "BA"):
                if orientation == "AB":
                    red_ai, blue_ai = ai_a, ai_b
                else:
                    red_ai, blue_ai = ai_b, ai_a

                random.seed(pair_seed)
                seed_ai(red_ai, pair_seed)
                seed_ai(blue_ai, pair_seed)
                started = time.perf_counter()
                try:
                    record = play_game(
                        red_ai, blue_ai,
                        move_time_limit_ms=move_time_limit_ms,
                    )
                    error = None
                except Exception as exc:
                    record = None
                    error = exc
                elapsed_ms = (time.perf_counter() - started) * 1000
                game_index = len(repeat_games) + 1
                row = _game_row(repeat_no, game_index, pair_index, orientation,
                                pair_seed, ai_a, ai_b, record, error, elapsed_ms)
                repeat_games.append(row)

                if not quiet:
                    print(f"\r第 {repeat_no}/{repeats} 轮  "
                          f"{len(repeat_games)}/{games} 局", end="", flush=True)

        all_games.extend(repeat_games)
        summary = summarize_games(repeat_games, ai_a_spec, ai_b_spec)
        summary.update({"repeat": repeat_no, "seed": repeat_seed,
                        "ai_a_load_ms": round(ai_a_load_ms, 3),
                        "ai_b_load_ms": round(ai_b_load_ms, 3)})
        repeat_summaries.append(summary)
        if not quiet:
            print()

    overall = summarize_games(all_games, ai_a_spec, ai_b_spec)
    overall["ai_a_average_load_ms"] = round(_mean(
        [summary["ai_a_load_ms"] for summary in repeat_summaries]), 3)
    overall["ai_b_average_load_ms"] = round(_mean(
        [summary["ai_b_load_ms"] for summary in repeat_summaries]), 3)
    overall["elapsed_seconds"] = round(time.perf_counter() - total_started, 3)
    document = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ai_a": ai_a_spec,
            "ai_b": ai_b_spec,
            "games_per_repeat": games,
            "repeats": repeats,
            "base_seed": seed,
            "move_time_limit_ms": float(move_time_limit_ms),
            "pairing": "AB/BA with colors swapped",
            "total_games": games * repeats,
            "ai_a_metadata": metadata_a,
            "ai_b_metadata": metadata_b,
        },
        "summary": overall,
        "repeats": repeat_summaries,
        "games": all_games,
    }

    json_path = output_dir / f"{stem}.json"
    summary_csv_path = output_dir / f"{stem}_summary.csv"
    games_csv_path = output_dir / f"{stem}_games.csv"

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)

    summary_rows = []
    for summary in repeat_summaries:
        summary_rows.append({"_scope": f"repeat_{summary['repeat']}",
                             "seed": summary["seed"],
                             **_flatten_summary(summary)})
    summary_rows.append({"_scope": "overall", "seed": seed,
                         **_flatten_summary(overall)})
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(games_csv_path, all_games)

    if not quiet:
        _print_summary(overall)
        print(f"JSON: {json_path}")
        print(f"汇总 CSV: {summary_csv_path}")
        print(f"逐局 CSV: {games_csv_path}")

    return document, {
        "json": json_path,
        "summary_csv": summary_csv_path,
        "games_csv": games_csv_path,
    }


def _flatten_summary(summary):
    flattened = {}
    for key, value in summary.items():
        if key == "error_types":
            flattened[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif key == "repeat":
            continue
        elif key == "seed":
            continue
        else:
            flattened[key] = value
    return flattened


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary):
    print("\n===== AI 对比评测 =====")
    print(f"AI A：{summary['ai_a']}    AI B：{summary['ai_b']}")
    print(f"完成：{summary['games_completed']} 局，异常：{summary['errors']} 局")
    print(f"AI A 胜率：{summary['ai_a_win_rate']:.2f}% "
          f"(95% CI {summary['ai_a_win_rate_ci95_low']}% ~ "
          f"{summary['ai_a_win_rate_ci95_high']}%)")
    print(f"AI A 执红胜率：{summary['ai_a_red_win_rate']:.2f}% "
          f"({summary['ai_a_red_wins']}/{summary['ai_a_red_games']})")
    print(f"AI A 执蓝胜率：{summary['ai_a_blue_win_rate']:.2f}% "
          f"({summary['ai_a_blue_wins']}/{summary['ai_a_blue_games']})")
    print(f"红方胜率：{summary['red_win_rate']:.2f}%")
    print(f"平均手数：{summary['average_moves']:.2f}")
    print(f"AI A \u5e73\u5747/P95/P99/\u6700\u5927\u8017\u65f6\uff1a"
          f"{summary['ai_a_average_move_ms']:.3f} / "
          f"{summary['ai_a_p95_move_ms']:.3f} / "
          f"{summary['ai_a_p99_move_ms']:.3f} / "
          f"{summary['ai_a_max_move_ms']:.3f} ms")
    print(f"AI B \u5e73\u5747/P95/P99/\u6700\u5927\u8017\u65f6\uff1a"
          f"{summary['ai_b_average_move_ms']:.3f} / "
          f"{summary['ai_b_p95_move_ms']:.3f} / "
          f"{summary['ai_b_p99_move_ms']:.3f} / "
          f"{summary['ai_b_max_move_ms']:.3f} ms")
    print(f"\u8d85\u65f6\u7740\u6cd5\uff1a{summary['move_timeout_count']} "
          f"({summary['move_timeout_rate']:.2f}%)")

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hex AI 固定种子、AB/BA 配对对比评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ai-a", default="basic",
                        help="AI A：basic、random、neural 或 module:Class")
    parser.add_argument("--ai-b", default="random",
                        help="AI B：basic、random、neural 或 module:Class")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="每轮总对局数，必须为偶数")
    parser.add_argument("--repeats", type=int, default=1,
                        help="重复实验轮数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="基础随机种子")
    parser.add_argument("--move-time-limit-ms", type=float, default=500,
                        help="\u7edf\u8ba1\u5355\u6b65\u8d85\u65f6\u7684\u6beb\u79d2\u4e0a\u9650")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="CSV/JSON 输出目录")
    parser.add_argument("--name", default=None,
                        help="输出文件名前缀；默认使用时间戳")
    parser.add_argument("--quiet", action="store_true", help="不显示进度和摘要")
    args = parser.parse_args(argv)

    try:
        document, paths = run_benchmark(
            args.ai_a, args.ai_b, args.games, args.repeats,
            args.seed, args.output_dir, args.name, args.quiet,
            args.move_time_limit_ms)
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    return 0 if document["summary"]["games_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
