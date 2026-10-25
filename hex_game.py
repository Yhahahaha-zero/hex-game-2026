# -*- coding: utf-8 -*-
"""海克斯棋 pygame 界面：输入、协调、计时和复盘。"""
import os

import pygame

from ai.worker import AIWorker
from game_mode import GameMode
from game_records import (RecordValidationError, ReplaySession,
                          latest_record, load_record, save_match_record)
from game_timer import DualGameTimer
from report_export import capture_screen, export_report_bundle
from hex_board import BLUE, IllegalMoveError, RED, cell_name
from hex_engine import (HexEngine, STATE_GAME_OVER, STATE_MATCH_OVER,
                        STATE_PLAYING)
from ui import theme
from ui.board_renderer import (BOARD_W, WINDOW_H, BoardRenderer,
                               cell_center, hex_corners, pixel_to_hex)
from ui.history_panel import HistoryPanel, PANEL_W
from ui.screen_renderer import OverlayRenderer, make_confetti
from ui.sound import SoundManager


S_START, S_PLAY, S_WIN, S_MATCH_END, S_REPLAY = range(5)
WINDOW_W = BOARD_W + PANEL_W
AI_DELAY_MS = 350
WATCH_AUTO_MS = 1600
CNAMES = {RED: "红方", BLUE: "蓝方"}
FONT_DIR = r"C:\Windows\Fonts"
FONT_NAMES = {
    False: ("msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
    True: ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"),
}


def get_font(size, bold=False):
    """查找支持中文的系统字体。"""
    if os.name == "nt":
        for name in FONT_NAMES[bold]:
            path = os.path.join(FONT_DIR, name)
            if os.path.isfile(path):
                try:
                    return pygame.font.Font(path, size)
                except pygame.error:
                    continue
    else:
        for name in ("microsoftyahei", "msyh", "simhei", "dengxian", "simsun", None):
            if name is None:
                break
            try:
                path = pygame.font.match_font(name, bold=bold)
            except Exception:
                continue
            if path:
                return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class HexGame:
    S_START = S_START
    S_PLAY = S_PLAY
    S_WIN = S_WIN
    S_MATCH_END = S_MATCH_END
    S_REPLAY = S_REPLAY

    """界面控制器：规则由 HexEngine 管理，AI 和绘制由独立模块完成。"""

    def __init__(self):
        pygame.init()
        self.sound = SoundManager()
        self.window = pygame.display.set_mode(self._initial_window_size(), pygame.RESIZABLE)
        self.screen = pygame.Surface((WINDOW_W, WINDOW_H)).convert()
        pygame.display.set_caption("海克斯棋 Hex —— 红连上下，蓝连左右")
        self.clock = pygame.time.Clock()
        self.font_xs = get_font(15)
        self.font_s = get_font(18)
        self.font_m = get_font(22)
        self.font_l = get_font(40, bold=True)
        self.font_xl = get_font(54, bold=True)
        self.colorblind = False
        self.renderer = BoardRenderer(self.font_s)
        self.history_panel = HistoryPanel()
        self.history_panel.set_fonts(self.font_m, self.font_xs)
        self.history_panel.set_palette(self.renderer.CC)
        self.overlay = OverlayRenderer(self)
        self.timer = DualGameTimer()
        self.mode = None
        self.human = None
        self.ai = None
        self.ai_status_message = ""
        self.ai_request_id = 0
        self.ai_worker = None
        self._retired_workers = []
        self.last_search_stats = ()
        self.last_ai_explanation = None
        self.mode_buttons = []
        self.panel_message = ""
        self.replay = None
        self.replay_return_state = S_PLAY
        self.last_record_path = latest_record()
        self.last_report_paths = None
        self.engine = HexEngine("玩家1", "玩家2")
        self.reset_match(show_intro=True)

    # ---------- 引擎与计时同步 ----------
    @staticmethod
    def _player_labels():
        return {"玩家1": "玩家1", "玩家2": "玩家2"}

    @staticmethod
    def _initial_window_size():
        info = pygame.display.Info()
        available_w = max(560, (info.current_w or WINDOW_W) - 40)
        available_h = max(420, (info.current_h or WINDOW_H) - 80)
        scale = min(1.0, available_w / WINDOW_W, available_h / WINDOW_H)
        scale = max(0.35, scale)
        return int(WINDOW_W * scale), int(WINDOW_H * scale)

    def _resize_window(self, width, height):
        width = max(480, int(width))
        height = max(360, int(height))
        scale = min(width / WINDOW_W, height / WINDOW_H)
        self.window = pygame.display.set_mode(
            (int(WINDOW_W * scale), int(WINDOW_H * scale)), pygame.RESIZABLE
        )

    def _to_canvas(self, pos):
        width, height = self.window.get_size()
        return (
            int(pos[0] * WINDOW_W / max(1, width)),
            int(pos[1] * WINDOW_H / max(1, height)),
        )

    def _present(self):
        scaled = pygame.transform.smoothscale(self.screen, self.window.get_size())
        self.window.blit(scaled, (0, 0))
        pygame.display.flip()

    def _sync_engine(self):
        self.board = self.engine.board
        self.game_no = self.engine.game_no
        self.red_player = 1 if self.engine.red_side == "玩家1" else 2
        scores = self.engine.match_score
        self.match_scores = [scores["玩家1"], scores["玩家2"]]
        self.turn = self.engine.current_color or self.engine.winner_color
        self.winner = self.engine.winner_color
        self.win_path = self.engine.win_path
        moves = self.engine.game_moves()
        if moves:
            move = moves[-1]
            self.last_move = (move["q"], move["r"], move["color"], move["reason"])
        else:
            self.last_move = None

    def _cancel_ai_worker(self, timeout=0.6):
        for retired in self._retired_workers:
            retired.cancel()
        worker = self.ai_worker
        if worker is None:
            self._prune_retired_workers()
            return True
        worker.cancel()
        stopped = worker.join(timeout)
        if not stopped:
            self._retired_workers.append(worker)
        self.ai_worker = None
        self._prune_retired_workers()
        return stopped

    def _prune_retired_workers(self):
        self._retired_workers = [w for w in self._retired_workers if w.is_alive()]

    def _reset_transient_state(self):
        self._cancel_ai_worker()
        self.hover = None
        self.win_at = None
        self.last_move_at = None
        self.ripples = []
        self.confetti = None
        self.paused = False
        self.pending_move = None
        self.pending_reason = None
        self.ai_act_at = 0
        self.ai_request_id += 1
        self.ai_worker = None
        self.last_search_stats = ()
        self.last_ai_explanation = None

    def reset_match(self, show_intro=False):
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self.state = S_START if show_intro else S_PLAY
        self.intro_at = pygame.time.get_ticks()
        if self.state == S_PLAY:
            self.timer.start(self.engine.current_color)

    def new_game(self):
        self.engine.start_match()
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self.timer.start(self.engine.current_color)
        self.state = S_PLAY

    def start_mode(self, mode):
        self.mode = GameMode(mode)
        self.human = self.mode.human_player
        self.ai = None
        self.ai_status_message = ""
        self.panel_message = ""
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self.timer.start(self.engine.current_color)
        self.state = S_PLAY
        self.sound.play("click")

    def player_of(self, color):
        return 1 if self.engine.side_of(color) == "玩家1" else 2

    def player_label(self, pid):
        if self.mode == GameMode.HUMAN_VS_HUMAN or self.human == pid:
            return f"玩家{pid}"
        if self.mode == GameMode.AI_VS_AI:
            return f"AI{pid}"
        return "AI"

    def is_human_turn(self):
        if self.engine.state != STATE_PLAYING:
            return False
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        if self.mode == GameMode.AI_VS_AI:
            return False
        return self.player_of(self.engine.current_color) == self.human

    def advance_match(self):
        if self.engine.state != STATE_GAME_OVER:
            return
        self.engine.next_game()
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        if self.engine.state == STATE_MATCH_OVER:
            self.state = S_MATCH_END
        else:
            self.timer.start(self.engine.current_color)
            self.state = S_PLAY

    # ---------- 记录和复盘 ----------
    def save_record(self):
        try:
            self.last_record_path = save_match_record(
                self.engine,
                self.mode,
                self._player_labels(),
                self.timer,
            )
            self.panel_message = f"记录已保存：{self.last_record_path.name}"
        except Exception as exc:
            self.panel_message = f"保存失败：{exc}"

    def start_replay(self):
        path = latest_record() or self.last_record_path
        if path is None:
            self.panel_message = "暂无 JSON 对局记录"
            return
        try:
            document = load_record(path)
            self.replay = ReplaySession(document)
            self._cancel_ai_worker()
            self.replay_return_state = self.state
            self.timer.pause()
            self.state = S_REPLAY
            self.panel_message = self.replay.record_status
            self.last_record_path = path
        except RecordValidationError as exc:
            self.panel_message = f"记录异常：{exc}"
        except Exception as exc:
            self.panel_message = f"复盘失败：{exc}"

    def exit_replay(self):
        previous = self.replay_return_state
        self.replay = None
        self.state = previous
        if previous == S_PLAY:
            self.timer.resume()

    def replay_next(self):
        if self.replay is not None:
            self.replay.next()

    def replay_previous(self):
        if self.replay is not None:
            self.replay.previous()

    # ---------- 输入 ----------
    def export_report(self):
        search_stats = self.last_search_stats
        if self.ai_worker is not None:
            search_stats = tuple(self.ai_worker.progress.get("candidates", ()))
        try:
            self.last_report_paths = export_report_bundle(
                self.engine,
                self.mode,
                self._player_labels(),
                timer=self.timer,
                screen=self.screen,
                search_stats=search_stats,
            )
            self.panel_message = f"报告已导出：{self.last_report_paths['directory'].name}"
        except Exception as exc:
            self.panel_message = f"报告导出失败：{exc}"

    def save_screenshot(self):
        try:
            path = capture_screen(self.screen)
            self.panel_message = f"截图已保存：{path.name}"
        except Exception as exc:
            self.panel_message = f"截图失败：{exc}"

    def pick(self, pos):
        q, r = pixel_to_hex(*pos)
        return (q, r) if self.board.in_range(q, r) else None

    def _handle_panel_click(self, pos):
        if self.state == S_START:
            return False
        if self.history_panel.save_button.collidepoint(pos):
            self.save_record()
            return True
        if self.history_panel.export_button.collidepoint(pos):
            self.export_report()
            return True
        if self.history_panel.replay_button.collidepoint(pos):
            if self.state == S_REPLAY:
                self.exit_replay()
            else:
                self.start_replay()
            return True
        return False

    def on_click(self, pos):
        if self._handle_panel_click(pos):
            return
        if self.state == S_START:
            for rect, mode in self.mode_buttons:
                if rect.collidepoint(pos):
                    self.start_mode(mode)
                    return
        elif self.state == S_REPLAY:
            if pos[0] < BOARD_W:
                self.replay_next()
        elif self.state == S_WIN:
            self.advance_match()
        elif self.state == S_MATCH_END:
            self.reset_match()
        elif self.state == S_PLAY:
            if not self.is_human_turn():
                return
            cell = self.pick(pos)
            if cell is None or not self.board.is_empty(*cell):
                return
            self.apply_move(*cell)

    def on_space(self):
        if self.state == S_START:
            self.start_mode(GameMode.HUMAN_VS_HUMAN)
        elif self.state == S_REPLAY:
            self.replay_next()
        elif self.state == S_WIN:
            self.advance_match()
        elif self.state == S_MATCH_END:
            self.reset_match()
        elif self.state == S_PLAY and self.mode == GameMode.AI_VS_AI:
            self.paused = not self.paused
            if self.paused:
                self.timer.pause()
            else:
                self.timer.resume()

    def apply_move(self, q, r, reason=None):
        if self.engine.state != STATE_PLAYING:
            return False
        color = self.engine.current_color
        try:
            won = self.engine.play_move(q, r, reason=reason)
        except IllegalMoveError:
            return False
        now = pygame.time.get_ticks()
        self.last_move = (q, r, color, reason)
        self.last_move_at = now
        self.ripples.append((q, r, color, now))
        self.ripples = self.ripples[-12:]
        self.sound.play("tap")
        self._sync_engine()
        if won or self.engine.state == STATE_GAME_OVER:
            self.timer.stop()
            self.state = S_WIN
            self.win_at = now
            self.confetti = make_confetti(self.renderer)
            self.sound.play("win")
            self.pending_move = None
            self.pending_reason = None
            self.ai_worker = None
        else:
            self.timer.switch_to(self.engine.current_color)
            self.state = S_PLAY
        return won

    # ---------- AI ----------
    def _consume_ai_result(self, result):
        if result is None:
            return
        if (result.request_id != self.ai_request_id or self.state != S_PLAY
                or self.engine.current_color != result.color
                or result.cancelled):
            return
        if result.ai is not None:
            self.ai = result.ai
        self.last_search_stats = tuple(result.search_stats or ())
        self.last_ai_explanation = result.explanation
        if result.init_error:
            print(f"[AI] {result.init_error}")
            self.ai_status_message = "模型不可用，已回退基础 AI"
        if result.inference_error:
            print(f"[AI] 推理失败: {result.inference_error}")
            self.ai_status_message = "模型推理失败，已回退基础 AI"
        if result.fallback_error:
            print(f"[AI] 基础 AI 回退失败: {result.fallback_error}")
            self.ai_status_message = "AI 执行失败"
            return
        if result.move is None:
            return
        self.pending_move = result.move
        self.pending_reason = result.reason
        self.ai_act_at = pygame.time.get_ticks() + AI_DELAY_MS

    def _ai_progress_text(self):
        if self.ai_worker is None:
            return ""
        progress = self.ai_worker.progress
        phase = progress.get("phase")
        if phase == "loading":
            return "AI 正在加载模型…"
        if phase == "searching":
            best = progress.get("best_move")
            best_text = cell_name(*best) if best else "-"
            return (
                f"搜索 {progress.get('completed', 0)}/"
                f"{progress.get('total', 0)}\n"
                f"最佳 {best_text} · {progress.get('elapsed_ms', 0):.0f} ms"
            )
        if phase == "thinking":
            return "AI 正在思考…"
        return ""

    def update_ai(self):
        if self.state != S_PLAY or self.is_human_turn():
            return
        self._prune_retired_workers()
        if self._retired_workers:
            return
        now = pygame.time.get_ticks()
        if self.pending_move is not None:
            if now >= self.ai_act_at:
                q, r = self.pending_move
                reason, self.pending_move = self.pending_reason, None
                self.apply_move(q, r, reason)
            return
        if self.ai_worker is not None:
            if self.ai_worker.is_alive():
                return
            worker, self.ai_worker = self.ai_worker, None
            if worker.result is not None:
                self._consume_ai_result(worker.result)
                return
        color = self.engine.current_color
        self.ai_request_id += 1
        self.ai_worker = AIWorker(self.board, color, self.ai_request_id, ai=self.ai)
        self.ai_worker.start()

    def quit(self):
        self._cancel_ai_worker(timeout=1.0)
        for worker in self._retired_workers:
            worker.cancel()
            worker.join(0.2)
        self._retired_workers = [
            worker for worker in self._retired_workers if worker.is_alive()
        ]
        self.sound.quit()
        pygame.quit()
        raise SystemExit

    # ---------- 主循环 ----------
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    ctrl = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
                    if event.key == pygame.K_ESCAPE:
                        if self.state == S_REPLAY:
                            self.exit_replay()
                        else:
                            self.quit()
                    elif ctrl and event.key == pygame.K_s:
                        self.save_record()
                    elif ctrl and event.key == pygame.K_e:
                        self.export_report()
                    elif ctrl and event.key == pygame.K_p:
                        self.save_screenshot()
                    elif event.key == pygame.K_F5:
                        self.start_replay()
                    elif event.key == pygame.K_c and not ctrl:
                        self.colorblind = not self.colorblind
                        self.renderer.set_colorblind(self.colorblind)
                        self.history_panel.set_palette(self.renderer.CC)
                        self.sound.play("tick")
                        self.panel_message = (
                            "色盲友好配色已开启"
                            if self.colorblind else
                            "已恢复普通配色"
                        )
                    elif self.state == S_REPLAY:
                        if event.key == pygame.K_LEFT:
                            self.replay_previous()
                        elif event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                            self.replay_next()
                    elif event.key == pygame.K_SPACE:
                        self.on_space()
                    elif self.state == S_START and pygame.K_1 <= event.key <= pygame.K_4:
                        self.start_mode(event.key - pygame.K_1 + 1)
                elif event.type == pygame.VIDEORESIZE:
                    self._resize_window(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(self._to_canvas(event.pos))
            self.hover = self.pick(self._to_canvas(pygame.mouse.get_pos()))
            if self.state == S_PLAY and not self.paused:
                self.update_ai()
            elif (self.state == S_WIN and self.mode == GameMode.AI_VS_AI
                  and not self.paused and self.win_at
                  and pygame.time.get_ticks() - self.win_at > WATCH_AUTO_MS):
                self.advance_match()
            self.draw()
            self._present()
            self.clock.tick(60)

    # ---------- 绘制 ----------
    def blit(self, text, pos, font, color, anchor="topleft"):
        surface = font.render(text, True, color)
        rect = surface.get_rect(**{anchor: pos})
        self.screen.blit(surface, rect)
        return rect

    def draw(self):
        self.screen.fill(theme.INK_TOP)
        replaying = self.state == S_REPLAY and self.replay is not None
        board = self.replay.board if replaying else self.board
        win_path = None if replaying else self.win_path
        last_move = self.replay.last_move if replaying else self.last_move
        now = pygame.time.get_ticks()
        can_hover = self.state == S_PLAY and self.is_human_turn()
        preview_color = self.engine.current_color if can_hover else None
        self.ripples = [
            event for event in self.ripples if now - event[3] <= 700
        ]
        self.renderer.draw(
            self.screen,
            board,
            hover=self.hover,
            preview_color=preview_color,
            win_path=win_path,
            last_move=last_move,
            can_hover=can_hover,
            now_ms=now,
            last_move_at=None if replaying else self.last_move_at,
            win_started_at=None if replaying else self.win_at,
            ripples=() if replaying else tuple(self.ripples),
        )
        if self.state != S_START:
            moves = self.replay.current_moves() if replaying else self.engine.game_moves(self.game_no)
            game_no = self.replay.game_no if replaying else self.game_no
            status = self._panel_status()
            if replaying:
                search_stats = ()
            elif self.ai_worker is not None:
                search_stats = tuple(self.ai_worker.progress.get("candidates", ()))
            else:
                search_stats = self.last_search_stats
            self.history_panel.draw(
                self.screen,
                moves,
                game_no,
                timers=self.timer.formatted(),
                ai_status=status,
                replay_mode=replaying,
                search_stats=search_stats,
                ai_explanation=self.last_ai_explanation,
            )
        self.draw_hud()
        if self.state == S_START:
            self.draw_start_screen()
        elif self.state == S_WIN:
            self.draw_win_banner()
        elif self.state == S_MATCH_END:
            self.draw_match_end()
        elif self.state == S_REPLAY:
            self.draw_replay_banner()

    def _panel_status(self):
        if self.state == S_REPLAY and self.replay is not None:
            return (
                f"复盘 {self.replay.step}/{self.replay.total_moves}\n"
                f"{self.replay.record_status} / {self.replay.game_no} 局\n"
                "← → 切换 / Esc 返回"
            )
        progress = self._ai_progress_text()
        parts = [part for part in (self.ai_status_message, progress, self.panel_message) if part]
        return "\n".join(parts)

    def draw_hud(self):
        self.overlay.draw_hud()

    def draw_start_screen(self):
        self.overlay.draw_start_screen()

    def draw_win_banner(self):
        self.overlay.draw_win_banner()

    def draw_match_end(self):
        self.overlay.draw_match_end()

    def draw_replay_banner(self):
        self.overlay.draw_replay_banner()


def main():
    HexGame().run()


if __name__ == "__main__":
    main()
