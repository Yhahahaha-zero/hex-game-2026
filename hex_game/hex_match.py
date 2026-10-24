# -*- coding: utf-8 -*-
"""
海克斯棋 AI 对 AI 自动对局工具：胜率统计、对局记录、失败棋局复现、运行耗时统计。

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

from hex_board import HexBoard, RED, BLUE, cell_name, parse_cell
from hex_ai import create_ai

RECORDS_FILE = "hex_records.json"


def play_game(red_ai, blue_ai):
    """AI 对 AI 一局，返回对局记录 dict（含每手坐标、原因、耗时）。"""
    board = HexBoard()
    turn = RED
    moves = []
    t0 = time.perf_counter()
    while True:
        ai = red_ai if turn == RED else blue_ai
        s = time.perf_counter()
        (q, r), reason = ai.choose_move(board, turn)
        ms = (time.perf_counter() - s) * 1000
        board.place(q, r, turn)
        moves.append({"p": "红" if turn == RED else "蓝",
                      "cell": cell_name(q, r),
                      "reason": reason,
                      "ms": round(ms, 2)})
        if board.has_won(turn):
            return {"winner": "红" if turn == RED else "蓝",
                    "moves": moves,
                    "moves_count": len(moves),
                    "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        turn = BLUE if turn == RED else RED


def run_match(n, red_ai, blue_ai, quiet=False):
    """进行 n 局对弈，打印统计并保存记录，返回记录文档 dict。"""
    t0 = time.perf_counter()
    red_wins = blue_wins = 0
    total_ms_sum = max_ms = 0.0
    records = []
    for i in range(1, n + 1):
        rec = play_game(red_ai, blue_ai)
        rec["id"] = i
        rec["red_ai"] = red_ai.name
        rec["blue_ai"] = blue_ai.name
        if rec["winner"] == "红":
            red_wins += 1
        else:
            blue_wins += 1
        records.append(rec)
        for m in rec["moves"]:
            total_ms_sum += m["ms"]
            max_ms = max(max_ms, m["ms"])
        if not quiet and i % 10 == 0:
            print(f"\r已对弈 {i}/{n} 局…", end="", flush=True)
    elapsed = time.perf_counter() - t0
    total_moves = sum(r["moves_count"] for r in records)

    doc = {"saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "red_ai": red_ai.name, "blue_ai": blue_ai.name, "games": n,
           "summary": {"red_wins": red_wins, "blue_wins": blue_wins},
           "games": records}
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def load_records(path=RECORDS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def replay_game(rec, speed=None):
    """逐步复现一局。speed=None 时回车走一步；否则每隔 speed 秒自动走一步。"""
    board = HexBoard()
    print(f"—— 第 {rec['id']} 局  {rec.get('red_ai', '?')} vs {rec.get('blue_ai', '?')}  "
          f"胜方：{rec['winner']} ——")
    for i, m in enumerate(rec["moves"], 1):
        q, r = parse_cell(m["cell"])
        board.place(q, r, RED if m["p"] == "红" else BLUE)
        print(f"第 {i:>3} 手  {m['p']}方 {m['cell']}（{m['reason']}）")
        print(board.render_text())
        if speed is None:
            input("回车继续…")
        else:
            time.sleep(speed)
    print(f"本局结束：{rec['winner']}方获胜，共 {rec['moves_count']} 手")


def list_records(path=RECORDS_FILE):
    doc = load_records(path)
    s = doc["summary"]
    print(f"{doc['saved']} 保存，共 {doc['games']} 局"
          f"（{doc['red_ai']} vs {doc['blue_ai']}）")
    print(f"  红胜 {s['red_wins']}，蓝胜 {s['blue_wins']}")
    for rec in doc["games"]:
        print(f"  第 {rec['id']:>3} 局  {rec['winner']}胜  {rec['moves_count']:>3} 手  "
              f"{rec['total_ms']:>8.1f} ms")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(
        description="海克斯棋 AI 对 AI 自动对局：统计、记录、复盘、耗时",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--games", type=int, default=100, help="对局数量")
    p.add_argument("--ai-red", choices=("random", "basic"), default="basic", help="红方 AI")
    p.add_argument("--ai-blue", choices=("random", "basic"), default="basic", help="蓝方 AI")
    p.add_argument("--list", action="store_true", help="列出已保存的对局记录")
    p.add_argument("--replay", type=int, metavar="ID", help="复现指定编号的对局")
    p.add_argument("--replay-lost", choices=("red", "blue"),
                   help="依次复现该颜色一方失败的全部棋局")
    p.add_argument("--speed", type=float, metavar="秒",
                   help="复现时每步自动间隔（秒），省略则回车逐步")
    args = p.parse_args(argv)

    if args.list:
        list_records()
    elif args.replay:
        doc = load_records()
        rec = next((g for g in doc["games"] if g["id"] == args.replay), None)
        if rec is None:
            print(f"没有编号为 {args.replay} 的对局记录")
            return 1
        replay_game(rec, args.speed)
    elif args.replay_lost:
        doc = load_records()
        lost = [g for g in doc["games"]
                if g["winner"] != ("红" if args.replay_lost == "red" else "蓝")]
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
