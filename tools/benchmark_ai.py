# -*- coding: utf-8 -*-
"""Hex AI 对比评测工具。

支持：
- 内置 basic / random AI；
- 自定义 AI 类或工厂：module:Class、path/to/file.py:Class；
- 固定随机种子；
- AB/BA 配对并交换红蓝；
- 多轮重复实验；
- 汇总胜率、执色胜率、平均手数、平均/最大耗时和异常；
- 输出 JSON、汇总 CSV 和逐局 CSV。

示例：
  python tools/benchmark_ai.py --games 500 --repeats 3
  python tools/benchmark_ai.py --ai-a basic --ai-b random --games 1000
  python tools/benchmark_ai.py --ai-a basic --ai-b models.my_ai:ModelAI
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

BUILTIN_AIS = ("basic", "random")
DEFAULT_OUTPUT_DIR = "benchmark_results"
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
    if spec in BUILTIN_AIS:
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
        "ai_b_average_move_ms": round(b_move_time / b_move_count, 3) if b_move_count else 0.0,
        "ai_b_max_move_ms": round(b_max, 3),
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
    return row


def run_benchmark(ai_a_spec, ai_b_spec, games, repeats, seed,
                  output_dir=DEFAULT_OUTPUT_DIR, name=None, quiet=False):
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
    total_started = time.perf_counter()

    for repeat_index in range(repeats):
        repeat_no = repeat_index + 1
        repeat_seed = seed + repeat_index * 1_000_003
        ai_a = factory_a()
        ai_b = factory_b()
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
                    record = play_game(red_ai, blue_ai)
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
        summary.update({"repeat": repeat_no, "seed": repeat_seed})
        repeat_summaries.append(summary)
        if not quiet:
            print()

    overall = summarize_games(all_games, ai_a_spec, ai_b_spec)
    overall["elapsed_seconds"] = round(time.perf_counter() - total_started, 3)
    document = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ai_a": ai_a_spec,
            "ai_b": ai_b_spec,
            "games_per_repeat": games,
            "repeats": repeats,
            "base_seed": seed,
            "pairing": "AB/BA with colors swapped",
            "total_games": games * repeats,
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
    print(f"AI A 平均/最大耗时：{summary['ai_a_average_move_ms']:.3f} / "
          f"{summary['ai_a_max_move_ms']:.3f} ms")
    print(f"AI B 平均/最大耗时：{summary['ai_b_average_move_ms']:.3f} / "
          f"{summary['ai_b_max_move_ms']:.3f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hex AI 固定种子、AB/BA 配对对比评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ai-a", default="basic",
                        help="AI A：basic、random 或 module:Class")
    parser.add_argument("--ai-b", default="random",
                        help="AI B：basic、random 或 module:Class")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="每轮总对局数，必须为偶数")
    parser.add_argument("--repeats", type=int, default=1,
                        help="重复实验轮数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="基础随机种子")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="CSV/JSON 输出目录")
    parser.add_argument("--name", default=None,
                        help="输出文件名前缀；默认使用时间戳")
    parser.add_argument("--quiet", action="store_true", help="不显示进度和摘要")
    args = parser.parse_args(argv)

    try:
        document, paths = run_benchmark(
            args.ai_a, args.ai_b, args.games, args.repeats,
            args.seed, args.output_dir, args.name, args.quiet)
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    return 0 if document["summary"]["games_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
