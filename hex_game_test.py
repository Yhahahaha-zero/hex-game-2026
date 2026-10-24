# -*- coding: utf-8 -*-
"""海克斯棋自测：坐标换算、邻居表、连通性判定（BFS/DFS/并查集）、
随机对局、AI 策略、图形界面冒烟、自动对局与复盘。"""
import contextlib
import io
import os
import random
import time
import tempfile

os.environ["SDL_VIDEODRIVER"] = "dummy"   # 无窗口模式

import hex_board as hb
import hex_ai as hx
import hex_game as hg
import hex_match as hm

# 1) 坐标往返：每个单元格中心必须映回自身
for q in range(11):
    for r in range(11):
        x, y = hg.cell_center(q, r)
        assert hg.pixel_to_hex(x, y) == (q, r), (q, r, hg.pixel_to_hex(x, y))
print("[1] 坐标往返 OK")

# 2) 邻居数与坐标文本解析
b = hb.HexBoard()
assert len(list(b.neighbors(5, 5))) == 6
assert len(list(b.neighbors(0, 0))) == 2
assert hb.cell_name(2, 4) == "5C"                # 先纵后横
assert hb.parse_cell("5C") == (2, 4)
assert hb.parse_cell("C5") is None               # strict default
assert hb.parse_cell("C5", strict=False) == (2, 4)  # explicit compatibility
for invalid in ("5c", "05C", " 5C", "5C ", "0A", "12A"):
    assert hb.parse_cell(invalid) is None, invalid
print("[2] 邻居表与坐标解析 OK")

# 3) 随机对局 300 局：每局必有胜负（无和棋），且获胜通路端点正确
results = {hb.RED: 0, hb.BLUE: 0}
for _ in range(300):
    b = hb.HexBoard()
    turn = hb.RED
    moves = 0
    while True:
        q, r = random.choice(b.legal_moves())
        b.place(q, r, turn)
        moves += 1
        if b.has_won(turn):
            results[turn] += 1
            path = b.winning_path(turn)
            if turn == hb.RED:
                assert any(rr == 0 for _, rr in path), "红方胜路未触及上边界"
                assert any(rr == 10 for _, rr in path), "红方胜路未触及下边界"
            else:
                assert any(qq == 0 for qq, _ in path), "蓝方胜路未触及左边界"
                assert any(qq == 10 for qq, _ in path), "蓝方胜路未触及右边界"
            break
        assert moves < 121, "棋盘下满仍未分胜负（不应发生）"
        turn = 3 - turn
print(f"[3] 随机对局 300 局均分胜负 OK（红胜 {results[hb.RED]}，蓝胜 {results[hb.BLUE]}）")

# 4) 三种连通性判定互相一致；would_win 与“试下+BFS”对照
random.seed(20261024)
for _ in range(50):
    b = hb.HexBoard()
    cells = [(q, r) for q in range(11) for r in range(11)]
    random.shuffle(cells)
    for q, r in cells[:60]:
        b.place(q, r, random.choice((hb.RED, hb.BLUE)))
    for p in (hb.RED, hb.BLUE):
        bfs = b.winning_path(p, "bfs") is not None
        dfs = b.winning_path(p, "dfs") is not None
        assert bfs == dfs == b.has_won(p), "BFS/DFS/并查集 结论不一致"
        for q, r in b.legal_moves(p):
            b.place(q, r, p)
            naive = b.winning_path(p) is not None
            b.undo()
            assert b.would_win(q, r, p) == naive, f"would_win 误判 {(q, r)}"
print("[4] BFS/DFS/并查集一致性与 would_win 对照 OK")

# 5) AI：直接获胜、紧急防守、合法着法
b = hb.HexBoard()
for r in range(10):                             # 红方占满 A 列前 10 格
    b.place(0, r, hb.RED)
assert hx.winning_moves(b, hb.RED) == [(0, 10)]  # 唯一一步取胜点
ai = hx.BasicAI()
m, reason = ai.choose_move(b, hb.RED)
assert m == (0, 10) and reason == "直接获胜"
m, reason = ai.choose_move(b, hb.BLUE)
assert m == (0, 10) and reason == "紧急防守"

random.seed(42)
for _ in range(50):                             # 随机棋盘上 AI 总返回合法空位
    b = hb.HexBoard()
    cells = [(q, r) for q in range(11) for r in range(11)]
    random.shuffle(cells)
    for q, r in cells[:40]:
        b.place(q, r, random.choice((hb.RED, hb.BLUE)))
    for p in (hb.RED, hb.BLUE):
        m, _ = ai.choose_move(b, p)
        assert b.is_empty(*m), "AI 落子非法"
print("[5] AI 直接获胜 / 紧急防守 / 合法性 OK")

# 6) 图形界面冒烟：开始画面、双人一局、人机回合守卫、AI 走子
g = hg.HexGame()
assert g.state == hg.S_START
g.draw()
g.start_mode(1)                                 # 双人对战
assert g.state == hg.S_PLAY
while g.state == hg.S_PLAY:                     # 随机走完一局
    q, r = random.choice(g.board.legal_moves())
    x, y = hg.cell_center(q, r)
    g.on_click((int(x), int(y)))
    g.draw()
assert g.state == hg.S_WIN
g.on_space()                                    # 第二局（交换红蓝）
assert g.state == hg.S_PLAY and g.game_no == 2

g2 = hg.HexGame()
g2.start_mode(2)                                # 人机：玩家执红
q, r = random.choice(g2.board.legal_moves())
x, y = hg.cell_center(q, r)
g2.on_click((int(x), int(y)))
assert g2.board.history and g2.turn == hg.BLUE  # 轮到 AI
g2.on_click((int(x), int(y)))                   # AI 回合点人无效
assert len(g2.board.history) == 1
started_ai = time.monotonic()
g2.update_ai()                                  # 启动后台 AI
assert time.monotonic() - started_ai < 0.5, "AI 调用阻塞了主线程"
deadline = time.monotonic() + 5.0
while g2.pending_move is None and time.monotonic() < deadline:
    time.sleep(0.01)
    g2.update_ai()
assert g2.pending_move is not None
g2.ai_act_at = 0
g2.update_ai()                                  # AI 落子
assert len(g2.board.history) == 2
g2.draw()
print("[6] 图形界面冒烟（双人/人机/AI 走子）OK")

# 7) 自动对局 10 局 + 临时记录保存/读取 + 复盘冒烟
random.seed(7)
with tempfile.TemporaryDirectory() as temp_dir:
    record_path = os.path.join(temp_dir, "hex_records.json")
    doc = hm.run_match(10, hx.create_ai("random"), hx.create_ai("basic"),
                       quiet=True, path=record_path)
    s = doc["summary"]
    assert s["red_wins"] + s["blue_wins"] == 10
    assert len(doc["games"][0]["moves"]) == doc["games"][0]["moves_count"]
    rec = hm.load_records(record_path)["games"][0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hm.replay_game(rec, speed=0.0)
    out = buf.getvalue()
    assert "本局结束" in out and rec["winner"] in out
print("[7] 自动对局 / 临时记录 / 复盘 OK "
      f"（随机红 {s['red_wins']} : {s['blue_wins']} 基础蓝）")

pygame = hg.pygame
pygame.quit()
print("全部自测通过")
