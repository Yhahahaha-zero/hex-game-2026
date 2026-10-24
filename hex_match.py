# -*- coding: utf-8 -*-
"""
海克斯棋 AI 对 AI 自动对局工具：胜率统计、对局记录、失败棋局复现、运行耗时统计。

本模块只负责组织 AI 对局、统计和记录；落子、换手、判胜与历史记录统一由
HexEngine 负责，不在自动对局或复盘流程中复制规则逻辑。

用法：
  py hex_match.py                          # 默认 100 局，基础AI vs 基础AI
  py hex_match.py --games 500
  py hex_match.py --ai-red random --ai-blue basic
  py hex_match.py --list                   # 列出已保存的对局记录
  py hex_match.py --replay 7               # 逐步复现第 7 局（回车走一步）
  py hex_match.py --replay 7 --speed 0.3   # 每 0.3 秒自动走一步
  py hex_match.py --replay-lost red        # 依次复现红方失败的全部棋局

对局记录保存在 hex_records.json（UTF-8）。
"""
import argparse
import json
import sys
import time
from datetime import datetime

from hex_board import RED, BLUE, COLOR_NAMES, parse_cell
from hex_ai import create_ai
from hex_engine import HexEngine, STATE_PLAYING, STATE_GAME_OVER

RECORDS_FILE = "hex_records.json"


def play_game(red_ai, blue_ai):
    """AI 对 AI 一局，返回对局记录 dict（含每手坐标、原因、耗时）。

    所有落子、换手和判胜均通过 HexEngine，本函数不直接修改棋盘。
    """
    game = HexEngine("红方", "蓝方")
    t0 = time.perf_counter()

    while game.state == STATE_PLAYING:
        color = game.current_color
        ai = red_ai if color == RED else blue_ai
        started = time.perf_counter()
        (q, r), reason = ai.choose_move(game.board, color)
        elapsed_ms = (time.perf_counter() - started) * 1000
        game.play_move(q, r, ms=elapsed_ms, reason=reason)

    if game.winner_color not in (RED, BLUE):
        raise RuntimeError("对局结束但没有产生胜方")

    moves = [
        {
            "p": COLOR_NAMES[move["color"]],
            "cell": move["cell"],
            "reason": move["reason"],
            "ms": move["ms"],
        }
        for move in game.move_history
    ]
    return {
        "winner": COLOR_NAMES[game.winner_color],
        "moves": moves,
        "moves_count": len(moves),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def run_match(n, red_ai, blue_ai, quiet=False):
    """进行 n 局对弈，打印统计并保存记录，返回记录文档 dict。"""
    if n <= 0:
        raise ValueError("对局数量必须大于 0")

    t0 = time.perf_counter()
    red_wins = blue_wins = 0
    total_ms_sum = max_ms = 0.0
    records = []
    for i in range(1, n + 1):
        rec = play_game(red_ai, blue_ai)
        rec["id"] = i
        rec["red_ai"] = red_ai.name
        rec["blue_ai"] = blue_ai.name
        if rec["winner"] == COLOR_NAMES[RED]:
            red_wins += 1
        else:
            blue_wins += 1
        records.append(rec)
        for move in rec["moves"]:
            total_ms_sum += move["ms"]
            max_ms = max(max_ms, move["ms"])
        if not quiet and i % 10 == 0:
            print(f"\r已对弈 {i}/{n} 局…", end="", flush=True)

    elapsed = time.perf_counter() - t0
    total_moves = sum(record["moves_count"] for record in records)

    doc = {
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "red_ai": red_ai.name,
        "blue_ai": blue_ai.name,
        "game_count": n,
        "summary": {"red_wins": red_wins, "blue_wins": blue_wins},
        "games": records,
    }
    save_records(doc)

    print("\n===== 对局统计 =====")
    print(f"红方 AI：{red_ai.name}（每局先手）    蓝方 AI：{blue_ai.name}")
    print(f"总局数：{n}（全部决出胜负，无和棋）")
    print(f"红方胜：{red_wins} 局（{red_wins / n * 100:.1f}%）    "
          f"蓝方胜：{blue_wins} 局（{blue_wins / n * 100:.1f}%）")
    print(f"平均每局手数：{total_moves / n:.1f}")
    print(f"运行总耗时：{elapsed:.2f} 秒（平均 {elapsed / n * 1000:.0f} 毫秒/局）")
    print(f"AI 单步思考：平均 {total_ms_sum / total_moves:.2f} 毫秒，最大 {max_ms:.2f} 毫秒")
    print(f"对局记录已保存：{RECORDS_FILE}")
    return doc


def save_records(doc, path=RECORDS_FILE):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=1)


def load_records(path=RECORDS_FILE):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def replay_game(rec, speed=None):
    """逐步复现一局。speed=None 时回车走一步；否则每隔 speed 秒自动走一步。"""
    game = HexEngine("红方", "蓝方")
    board = game.board
    print(f"—— 第 {rec['id']} 局  {rec.get('red_ai', '?')} vs {rec.get('blue_ai', '?')}  "
          f"胜方：{rec['winner']} ——")
    for i, move in enumerate(rec["moves"], 1):
        cell = parse_cell(move["cell"])
        if cell is None:
            raise ValueError(f"复盘记录包含非法坐标: {move['cell']!r}")
        q, r = cell
        recorded_color = RED if move["p"] == COLOR_NAMES[RED] else BLUE
        if game.state != STATE_PLAYING:
            raise ValueError("复盘记录在对局结束后仍然包含落子")
        if game.current_color != recorded_color:
            raise ValueError("复盘记录的颜色顺序与规则不一致")
        game.play_move(q, r, ms=move.get("ms", 0.0), reason=move.get("reason", ""))
        print(f"第 {i:>3} 手  {move['p']}方 {move['cell']}（{move['reason']}）")
        print(board.render_text())
        if speed is None:
            input("回车继续…")
        else:
            time.sleep(speed)

    if game.state != STATE_GAME_OVER:
        raise ValueError("复盘记录提前结束，未产生胜方")
    if COLOR_NAMES[game.winner_color] != rec["winner"]:
        raise ValueError("复盘结果与记录中的胜方不一致")
    print(f"本局结束：{rec['winner']}方获胜，共 {rec['moves_count']} 手")


def list_records(path=RECORDS_FILE):
    doc = load_records(path)
    summary = doc["summary"]
    game_count = doc.get("game_count", len(doc["games"]))
    print(f"{doc['saved']} 保存，共 {game_count} 局"
          f"（{doc['red_ai']} vs {doc['blue_ai']}）")
    print(f"  红胜 {summary['red_wins']}，蓝胜 {summary['blue_wins']}")
    for rec in doc["games"]:
        print(f"  第 {rec['id']:>3} 局  {rec['winner']}胜  {rec['moves_count']:>3} 手  "
              f"{rec['total_ms']:>8.1f} ms")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="海克斯棋 AI 对 AI 自动对局：统计、记录、复盘、耗时",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--games", type=int, default=100, help="对局数量")
    parser.add_argument("--ai-red", choices=("random", "basic"), default="basic", help="红方 AI")
    parser.add_argument("--ai-blue", choices=("random", "basic"), default="basic", help="蓝方 AI")
    parser.add_argument("--list", action="store_true", help="列出已保存的对局记录")
    parser.add_argument("--replay", type=int, metavar="ID", help="复现指定编号的对局")
    parser.add_argument("--replay-lost", choices=("red", "blue"),
                        help="依次复现该颜色一方失败的全部棋局")
    parser.add_argument("--speed", type=float, metavar="秒",
                        help="复现时每步自动间隔（秒），省略则回车逐步")
    args = parser.parse_args(argv)

    if args.list:
        list_records()
    elif args.replay:
        doc = load_records()
        rec = next((game for game in doc["games"] if game["id"] == args.replay), None)
        if rec is None:
            print(f"没有编号为 {args.replay} 的对局记录")
            return 1
        replay_game(rec, args.speed)
    elif args.replay_lost:
        doc = load_records()
        lost = [game for game in doc["games"]
                if game["winner"] != (COLOR_NAMES[RED] if args.replay_lost == "red"
                                      else COLOR_NAMES[BLUE])]
        if not lost:
            print(f"记录中 {args.replay_lost} 方没有败局")
            return 0
        print(f"{args.replay_lost} 方共败 {len(lost)} 局，开始依次复现：")
        for rec in lost:
            replay_game(rec, args.speed)
    else:
        run_match(args.games, create_ai(args.ai_red), create_ai(args.ai_blue))
    return 0


if __name__ == "__main__":
    sys.exit(main())
