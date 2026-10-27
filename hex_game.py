# -*- coding: utf-8 -*-
"""海克斯棋 pygame 界面：输入、协调、计时和复盘。"""
import os

import pygame

from ai.worker import AIWorker
from ai.analysis_worker import AnalysisWorker
from ai.board_observation import build_engine_observation
from ai.llm_advisor import LLMAdvisor
from ai.llm_client import LLMDisabledError, create_llm_client
from ai.llm_config import load_llm_config
from ai.memory.collector import create_memory_collector_from_env
from ai.memory.trace import (
    ai_actor,
    ai_decision_trace,
    human_actor,
    human_decision_trace,
    snapshot_board,
)
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
from ui.analysis_panel import AnalysisPanel
from ui.vision_panel import VisionPanel
from vision.worker import VisionImportWorker
from vision.config import VisionConfigError, load_vision_config
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

    def __init__(self, analysis_advisor_factory=None, memory_collector=None):
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
        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.set_fonts(self.font_m, self.font_s, self.font_xs)
        self.vision_panel = VisionPanel()
        self.vision_panel.set_fonts(self.font_m, self.font_s, self.font_xs)
        self.vision_worker = None
        self._retired_vision_workers = []
        self.vision_request_id = 0
        self.vision_status = "idle"
        self.vision_error = ""
        self.vision_available = self._read_vision_available()
        self._analysis_advisor_factory = analysis_advisor_factory
        self._analysis_advisor = None
        self.analysis_worker = None
        self._retired_analysis_workers = []
        self.analysis_request_id = 0
        self.analysis_status = "idle"
        self.analysis_error = ""
        self.last_analysis = None
        self.analysis_model = ""
        self.analysis_elapsed_ms = None
        self.analysis_position_id = ""
        self.analysis_configured = self._read_analysis_configured()
        self.memory_collector = (
            memory_collector
            if memory_collector is not None
            else create_memory_collector_from_env()
        )
        self._pending_memory_trace = None
        self.timer = DualGameTimer()
        self.timer_history = []
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

    def _read_analysis_configured(self):
        if self._analysis_advisor_factory is not None:
            return True
        try:
            return bool(load_llm_config().enabled)
        except Exception:
            return False

    def _load_analysis_advisor(self):
        if self._analysis_advisor is not None:
            return self._analysis_advisor
        if self._analysis_advisor_factory is not None:
            advisor = self._analysis_advisor_factory()
        else:
            config = load_llm_config()
            if not config.enabled:
                raise LLMDisabledError(
                    "大模型尚未启用，请在 .env.local 中设置 LLM_ENABLED=1"
                )
            advisor = LLMAdvisor(
                create_llm_client(config),
                timeout=config.analysis_timeout_seconds,
                enable_thinking=config.enable_thinking,
            )
        if advisor is None:
            raise RuntimeError("分析助手初始化失败")
        self._analysis_advisor = advisor
        return advisor

    def _cancel_analysis_worker(self, timeout=0.1):
        for retired in self._retired_analysis_workers:
            retired.cancel()
        worker = self.analysis_worker
        if worker is None:
            self._prune_retired_analysis_workers()
            return True
        worker.cancel()
        stopped = worker.join(timeout)
        if not stopped:
            self._retired_analysis_workers.append(worker)
        self.analysis_worker = None
        self._prune_retired_analysis_workers()
        return stopped

    def _prune_retired_analysis_workers(self):
        self._retired_analysis_workers = [
            worker for worker in self._retired_analysis_workers
            if worker.is_alive()
        ]

    def _invalidate_analysis(self, mark_stale=False, hide=False):
        was_loading = self.analysis_worker is not None
        had_result = self.last_analysis is not None
        self._cancel_analysis_worker()
        self.analysis_request_id += 1
        self.last_analysis = None
        self.analysis_model = ""
        self.analysis_elapsed_ms = None
        self.analysis_position_id = ""
        if mark_stale and (was_loading or had_result):
            self.analysis_status = "stale"
            self.analysis_error = "局面已变化，请重新分析。"
            self.analysis_panel.show()
        else:
            self.analysis_status = "idle"
            self.analysis_error = ""
            if hide:
                self.analysis_panel.hide()

    def _current_analysis_position_id(self):
        try:
            observation = build_engine_observation(
                self.engine, history_limit=0, candidate_limit=0
            )
            return observation["position_id"]
        except Exception:
            return ""

    def _show_analysis_message(self, status, message):
        self.analysis_status = str(status)
        self.analysis_error = str(message)
        self.analysis_panel.show()

    def start_analysis(self):
        self.close_vision_panel()
        if self.analysis_worker is not None and self.analysis_worker.is_alive():
            self.analysis_panel.show()
            return False
        if self.state not in (S_PLAY, S_WIN, S_MATCH_END):
            message = (
                "复盘模式暂不支持大模型分析。"
                if self.state == S_REPLAY else
                "开始一局游戏后才能进行 AI 分析。"
            )
            self._show_analysis_message("error", message)
            return False
        try:
            observation = build_engine_observation(self.engine)
        except Exception as exc:
            self._show_analysis_message("error", f"生成棋局观察数据失败：{exc}")
            return False

        self._cancel_analysis_worker()
        self.analysis_request_id += 1
        self.analysis_status = "loading"
        self.analysis_error = ""
        self.last_analysis = None
        self.analysis_model = ""
        self.analysis_elapsed_ms = None
        self.analysis_position_id = observation["position_id"]
        self.analysis_panel.show()
        worker = AnalysisWorker(
            observation,
            self.analysis_request_id,
            advisor_loader=self._load_analysis_advisor,
        )
        self.analysis_worker = worker
        worker.start()
        self.sound.play("click")
        return True

    def close_analysis(self):
        if self.analysis_worker is not None:
            self._cancel_analysis_worker()
            self.analysis_request_id += 1
            self.analysis_status = "idle"
            self.analysis_error = ""
        self.analysis_panel.hide()

    def toggle_analysis(self):
        if self.analysis_panel.visible:
            self.close_analysis()
            return False
        return self.start_analysis()

    def update_analysis(self):
        self._prune_retired_analysis_workers()
        worker = self.analysis_worker
        if worker is None or worker.is_alive():
            return False
        self.analysis_worker = None
        result = worker.result
        if result is None or result.request_id != self.analysis_request_id:
            return False
        if result.status == "cancelled":
            if self.analysis_status != "stale":
                self.analysis_status = "idle"
                self.analysis_error = ""
            return False
        if result.status == "error":
            self.analysis_status = (
                "disabled" if result.error_type == "LLMDisabledError" else "error"
            )
            self.analysis_error = result.error or "分析失败"
            if result.error_type == "LLMDisabledError":
                self.analysis_configured = False
            self.analysis_panel.show()
            return False
        if result.status != "complete" or result.analysis is None:
            self._show_analysis_message("error", "分析结果无效")
            return False
        if result.position_id != self._current_analysis_position_id():
            self.last_analysis = None
            self._show_analysis_message("stale", "局面已变化，分析结果已过期。")
            return False

        self.last_analysis = result.analysis.as_dict()
        self.analysis_model = result.analysis.model_name
        self.analysis_elapsed_ms = result.analysis.elapsed_ms
        self.analysis_position_id = result.position_id
        self.analysis_status = "complete"
        self.analysis_error = ""
        self.analysis_configured = True
        self.analysis_panel.show()
        return True

    def _read_vision_available(self):
        try:
            if not load_vision_config().enabled:
                return False
        except VisionConfigError:
            return False
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False

    def _cancel_vision_worker(self, timeout=0.1):
        for worker in self._retired_vision_workers:
            worker.cancel()
        worker = self.vision_worker
        if worker is None:
            self._prune_retired_vision_workers()
            return True
        worker.cancel()
        stopped = worker.join(timeout)
        if not stopped:
            self._retired_vision_workers.append(worker)
        self.vision_worker = None
        self._prune_retired_vision_workers()
        return stopped

    def _prune_retired_vision_workers(self):
        self._retired_vision_workers = [
            worker for worker in self._retired_vision_workers
            if worker.is_alive()
        ]

    def close_vision_panel(self):
        if self.vision_worker is not None:
            self._cancel_vision_worker()
            self.vision_request_id += 1
        self.vision_status = "idle"
        self.vision_error = ""
        self.vision_panel.hide()

    def start_vision_import(self, image_path):
        if not self.vision_available:
            self.vision_status = "error"
            self.vision_error = "视觉依赖未安装，请安装 requirements-vision.txt"
            self.vision_panel.show(status="error", error=self.vision_error)
            return False
        if self.vision_worker is not None and self.vision_worker.is_alive():
            self.vision_panel.show(status="processing")
            return False
        self.close_analysis()
        self._cancel_vision_worker()
        self.vision_request_id += 1
        self.vision_status = "processing"
        self.vision_error = ""
        self.vision_panel.show(status="processing")
        try:
            vision_config = load_vision_config()
        except VisionConfigError as exc:
            self.vision_status = "error"
            self.vision_error = f"视觉配置无效: {exc}"
            self.vision_panel.show(status="error", error=self.vision_error)
            return False
        use_vlm = False
        model_name = "qwen3-vl-plus"
        try:
            llm_config = load_llm_config()
            use_vlm = bool(llm_config.enabled and vision_config.vlm_fallback)
            model_name = llm_config.vision_model
        except Exception:
            pass
        worker = VisionImportWorker(
            image_path,
            self.vision_request_id,
            use_vlm=use_vlm,
            model_name=model_name,
            min_confidence=vision_config.min_confidence,
            max_image_mb=vision_config.max_image_mb,
            cache_enabled=vision_config.cache_enabled,
        )
        self.vision_worker = worker
        worker.start()
        self.sound.play("click")
        return True

    def open_vision_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="选择海克斯棋棋盘截图",
                filetypes=(
                    ("图片", "*.png *.jpg *.jpeg *.webp"),
                    ("所有文件", "*.*"),
                ),
            )
            root.destroy()
        except Exception as exc:
            self.vision_status = "error"
            self.vision_error = f"无法打开文件选择器: {exc}"
            self.vision_panel.show(status="error", error=self.vision_error)
            return False
        if not path:
            return False
        return self.start_vision_import(path)

    def update_vision_import(self):
        self._prune_retired_vision_workers()
        worker = self.vision_worker
        if worker is None or worker.is_alive():
            return False
        self.vision_worker = None
        result = worker.result
        if result is None or result.request_id != self.vision_request_id:
            return False
        if result.status == "cancelled":
            self.vision_status = "idle"
            return False
        if result.status == "error":
            self.vision_status = "error"
            self.vision_error = result.error or "视觉导入失败"
            self.vision_panel.show(status="error", error=self.vision_error)
            return False
        self.vision_status = "complete"
        self.vision_error = ""
        self.vision_panel.show(status="complete", session=result.session)
        return True

    def save_vision_snapshot(self):
        session = self.vision_panel.session
        state = getattr(session, "state", None)
        if state is None or not state.valid or state.uncertain_cells:
            self.panel_message = "识别结果仍需人工检查，不能保存"
            return False
        try:
            try:
                output_dir = load_vision_config().output_dir or None
            except VisionConfigError:
                output_dir = None
            path = session.save_snapshot(output_dir)
            self.panel_message = f"视觉快照已保存：{path.name}"
        except Exception as exc:
            self.panel_message = f"视觉快照保存失败：{exc}"
            return False
        self.close_vision_panel()
        return True

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

    def _cancel_ai_request(self):
        """Cancel and invalidate the active AI request after undo."""
        self._cancel_ai_worker()
        self.ai_request_id += 1
        self.pending_move = None
        self.pending_reason = None
        self.ai_act_at = 0
        self.last_search_stats = ()
        self.last_ai_explanation = None

    def _reset_transient_state(self):
        self._invalidate_analysis(hide=True)
        self.close_vision_panel()
        self._cancel_ai_worker()
        self.timer_history = []
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
        self._pending_memory_trace = None

    def reset_match(self, show_intro=False):
        self._finalize_memory_before_replacement()
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self.state = S_START if show_intro else S_PLAY
        self.intro_at = pygame.time.get_ticks()
        if self.state == S_PLAY:
            self._start_memory_match()
            self.timer.start(self.engine.current_color)

    def new_game(self):
        self._finalize_memory_before_replacement()
        self.engine.start_match()
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self._start_memory_match()
        self.timer.start(self.engine.current_color)
        self.state = S_PLAY

    def start_mode(self, mode):
        self._finalize_memory_before_replacement()
        self.mode = GameMode(mode)
        self.human = self.mode.human_player
        self.ai = None
        self.ai_status_message = ""
        self.panel_message = ""
        self.engine = HexEngine("玩家1", "玩家2")
        self._reset_transient_state()
        self._sync_engine()
        self.timer.reset()
        self._start_memory_match()
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

    def is_human_color(self, color):
        """判断指定颜色当前是否由人类玩家执棋。"""
        if color not in (RED, BLUE):
            return False
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        if self.mode == GameMode.AI_VS_AI:
            return False
        return self.player_of(color) == self.human
    def is_human_turn(self):
        if self.engine.state != STATE_PLAYING:
            return False
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        if self.mode == GameMode.AI_VS_AI:
            return False
        return self.player_of(self.engine.current_color) == self.human

    def _memory_source_type(self):
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return "human_vs_human"
        if self.mode == GameMode.AI_VS_AI:
            return "ai_vs_ai"
        return "human_vs_ai"

    def _memory_actor_types(self):
        return {
            RED: "human" if self.is_human_color(RED) else "ai",
            BLUE: "human" if self.is_human_color(BLUE) else "ai",
        }

    def _memory_actor_metadata(self):
        actors = {}
        for color in (RED, BLUE):
            if self.is_human_color(color):
                actors[color] = human_actor(self.player_of(color))
            elif self.ai is not None:
                actors[color] = ai_actor(self.ai)
        return actors

    def _memory_call(self, method, *args, **kwargs):
        try:
            return getattr(self.memory_collector, method)(*args, **kwargs)
        except Exception as exc:
            print(f"[MEMORY] {type(exc).__name__}: {exc}")
            return None

    def _memory_context(self):
        try:
            return dict(getattr(self.memory_collector, "current_context", {}) or {})
        except Exception:
            return {}

    def _start_memory_match(self):
        if self.mode is None:
            return None
        sides = self.engine.sides_for_game(1)
        return self._memory_call(
            "start_match",
            mode_key=int(self.mode),
            mode_name=self.mode.label,
            red_team=sides[RED],
            blue_team=sides[BLUE],
            source_type=self._memory_source_type(),
            actor_types=self._memory_actor_types(),
            actor_metadata=self._memory_actor_metadata(),
        )

    def _start_memory_game(self):
        if self.mode is None:
            return None
        sides = self.engine.sides_for_game(self.engine.game_no)
        return self._memory_call(
            "begin_game",
            self.engine.game_no,
            sides=sides,
            actor_types=self._memory_actor_types(),
            actor_metadata=self._memory_actor_metadata(),
        )

    def _finalize_memory_before_replacement(self):
        if self.mode is None:
            return
        if self.engine.state == STATE_MATCH_OVER:
            self._memory_call("accept_match")
        else:
            self._memory_call("abort_match", "match_restarted_before_finish")

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
            self._start_memory_game()
            self.timer.start(self.engine.current_color)
            self.state = S_PLAY

    # ---------- records and replay ----------

    def save_record(self):
        try:
            self.last_record_path = save_match_record(
                self.engine,
                self.mode,
                self._player_labels(),
                self.timer,
            )
            if self.engine.state == STATE_MATCH_OVER:
                document = load_record(self.last_record_path)
                self._memory_call(
                    "accept_match",
                    record_id=document.get("record_id"),
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
            self._invalidate_analysis(hide=True)
            self.close_vision_panel()
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
        if self.history_panel.undo_button.collidepoint(pos):
            self.undo_move()
            return True
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
        if self.history_panel.analysis_button.collidepoint(pos):
            self.start_analysis()
            return True
        if self.history_panel.vision_button.collidepoint(pos):
            self.open_vision_dialog()
            return True
        return False

    def on_click(self, pos):
        if self.vision_panel.visible:
            action = self.vision_panel.handle_click(pos)
            if action == "reimport":
                self.open_vision_dialog()
            elif action == "save":
                self.save_vision_snapshot()
            elif action == "close":
                self.close_vision_panel()
            return

        if self.analysis_panel.visible:
            action = self.analysis_panel.handle_click(pos)
            if action == "retry":
                self.start_analysis()
            elif action == "close":
                self.close_analysis()
            return
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

    def undo_move(self):
        """Undo one move, or AI+human in human-vs-AI modes."""
        if self.state not in (S_PLAY, S_WIN, S_MATCH_END):
            self.panel_message = "当前状态不能悔棋"
            return False
        if not self.engine.can_undo():
            self.panel_message = "当前没有可以悔棋的棋子"
            return False

        self._cancel_ai_request()
        count = 1
        if self.mode != GameMode.HUMAN_VS_HUMAN:
            moves = self.engine.game_moves()
            if moves and not self.is_human_color(moves[-1]["color"]):
                count = min(2, len(moves))

        count = min(count, len(self.timer_history))
        if count <= 0:
            self.panel_message = "计时历史不可用，无法悔棋"
            return False

        restore_index = len(self.timer_history) - count
        timer_snapshot = self.timer_history[restore_index]
        undone = self.engine.undo_move(count)
        if undone <= 0:
            self.panel_message = "悔棋失败"
            return False
        self._memory_call("undo_moves", undone)
        self._pending_memory_trace = None
        self._invalidate_analysis(mark_stale=True)

        del self.timer_history[-undone:]
        self.timer.restore(timer_snapshot)
        now = pygame.time.get_ticks()
        self._sync_engine()
        self.state = S_PLAY
        self.win_at = None
        self.confetti = None
        self.last_move_at = now
        self.ripples = []
        self.paused = False
        self.timer.stop()
        self.timer.start(self.engine.current_color)
        self.panel_message = f"已悔棋 {undone} 步"
        self.sound.play("click")
        return True

    def apply_move(self, q, r, reason=None):
        if self.engine.state != STATE_PLAYING:
            return False
        color = self.engine.current_color
        board_before = snapshot_board(self.board)
        trace = self._pending_memory_trace
        if (trace is None or trace.color != color or trace.move != (q, r)):
            if self.is_human_color(color):
                trace = human_decision_trace(
                    (q, r), color, reason, self.player_of(color)
                )
            else:
                trace = None
        timer_snapshot = self.timer.savepoint()
        try:
            won = self.engine.play_move(q, r, reason=reason)
        except IllegalMoveError:
            self._pending_memory_trace = None
            return False
        self._pending_memory_trace = None
        if trace is not None:
            self._memory_call("record_move", board_before, trace)
        self.timer_history.append(timer_snapshot)
        self._invalidate_analysis(mark_stale=True)
        now = pygame.time.get_ticks()
        self.last_move = (q, r, color, reason)
        self.last_move_at = now
        self.ripples.append((q, r, color, now))
        self.ripples = self.ripples[-12:]
        self.sound.play("tap")
        self._sync_engine()
        if won or self.engine.state == STATE_GAME_OVER:
            result_reason = (
                self.engine.results[-1].get("reason", "connect")
                if self.engine.results else "connect"
            )
            self._memory_call(
                "mark_game_over",
                self.engine.winner_color,
                result_reason,
            )
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
            print(f"[AI] {result.inference_error}")
            self.ai_status_message = "模型推理失败，已回退基础 AI"
        if result.fallback_error:
            print(f"[AI] {result.fallback_error}")
            self.ai_status_message = "AI 执行失败"
            self._pending_memory_trace = None
            return
        if result.move is None:
            self._pending_memory_trace = None
            return
        self.pending_move = result.move
        self.pending_reason = result.reason
        try:
            self._pending_memory_trace = ai_decision_trace(result)
        except Exception as exc:
            self._pending_memory_trace = None
            print(f"[MEMORY] 决策轨迹创建失败: {exc}")
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
        self.ai_worker = AIWorker(
            self.board,
            color,
            self.ai_request_id,
            ai=self.ai,
            memory_context=self._memory_context(),
        )
        self.ai_worker.start()

    def quit(self):
        self._finalize_memory_before_replacement()
        self._memory_call("flush", timeout=3.0)
        self._memory_call("close")
        self._cancel_vision_worker(timeout=1.0)
        for worker in self._retired_vision_workers:
            worker.cancel()
            worker.join(0.2)
        self._retired_vision_workers = [
            worker for worker in self._retired_vision_workers
            if worker.is_alive()
        ]
        self._cancel_analysis_worker(timeout=1.0)
        for worker in self._retired_analysis_workers:
            worker.cancel()
            worker.join(0.2)
        self._retired_analysis_workers = [
            worker for worker in self._retired_analysis_workers
            if worker.is_alive()
        ]
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

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    ctrl = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
                    if event.key == pygame.K_F7:
                        self.open_vision_dialog()
                        continue
                    if event.key == pygame.K_F6:
                        self.toggle_analysis()
                        continue
                    if self.vision_panel.visible:
                        if event.key == pygame.K_ESCAPE:
                            self.close_vision_panel()
                        continue
                    if self.analysis_panel.visible:
                        if event.key == pygame.K_ESCAPE:
                            self.close_analysis()
                        continue
                    if event.key == pygame.K_ESCAPE:
                        if self.state == S_REPLAY:
                            self.exit_replay()
                        else:
                            self.quit()
                    elif ctrl and event.key == pygame.K_z:
                        self.undo_move()
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
            self.update_analysis()
            self.update_vision_import()
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
            moves = (
                self.replay.current_moves()
                if replaying
                else self.engine.game_moves(self.game_no)
            )
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
                undo_enabled=(
                    not replaying
                    and self.state in (S_PLAY, S_WIN, S_MATCH_END)
                    and self.engine.can_undo()
                ),
                analysis_active=self.analysis_status == "loading",
                analysis_available=self.analysis_configured,
                vision_available=self.vision_available,
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
        if self.analysis_panel.visible:
            self.analysis_panel.draw(
                self.screen,
                status=self.analysis_status,
                analysis=self.last_analysis,
                error=self.analysis_error,
                model_name=self.analysis_model,
                elapsed_ms=self.analysis_elapsed_ms,
                position_id=self.analysis_position_id,
            )
        if self.vision_panel.visible:
            self.vision_panel.draw(self.screen)

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
