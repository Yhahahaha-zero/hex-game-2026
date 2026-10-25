# -*- coding: utf-8 -*-
"""V5 policy/value 网络与有限时间 MCTS AI。"""
import random
import time
from pathlib import Path

from ai.evaluation import candidate_distance_score, shortest_path_cells
from ai.tactics import bridge_move_bonus, creates_double_threat
from hex_ai import BasicAI, winning_moves
from hex_board import BLUE, RED

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "hex_model_v5.pt"
MODEL_VERSION = "5.0.0"
ENCODING_VERSION = "3x11x11-v1"


def choose_tactical_move(board, player):
    """优先返回一步获胜或一步防守着法；没有战术着法时返回 None。"""
    wins = winning_moves(board, player)
    if wins:
        return random.choice(wins), "直接获胜"
    opponent = BLUE if player == RED else RED
    threats = winning_moves(board, opponent)
    if threats:
        return random.choice(threats), "紧急防守"
    return None


class NeuralMCTSAI:
    """神经网络先验 + PUCT MCTS，返回统一 AI 接口格式。"""

    name = "HexMCTS-v5"
    version = MODEL_VERSION
    time_limit_ms = 200

    def __init__(self, model_path=None, device=None, simulations=100,
                 c_puct=1.5, time_limit_ms=200, seed=None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"找不到神经网络模型：{self.model_path}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch 未安装，请先安装 requirements-ai.txt 中的依赖"
            ) from exc

        from ai.model_v5 import ENCODING_VERSION as MODEL_ENCODING_VERSION
        from ai.model_v5 import HexNetV5, MODEL_VERSION as MODEL_FILE_VERSION
        from ai.mcts import MCTS

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.simulations = simulations
        self.c_puct = c_puct
        self.time_limit_ms = time_limit_ms
        self.load_ms = 0.0
        self.progress_callback = None

        started = time.perf_counter()
        self.model = HexNetV5().to(self.device)
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_version = str(checkpoint.get("model_version", ""))
            checkpoint_encoding = str(checkpoint.get("encoding_version", ""))
        else:
            state_dict = checkpoint
            checkpoint_version = ""
            checkpoint_encoding = ""

        if checkpoint_version and checkpoint_version != MODEL_FILE_VERSION:
            raise RuntimeError(
                f"模型版本不匹配：文件为 {checkpoint_version}，代码要求 {MODEL_FILE_VERSION}"
            )
        if checkpoint_encoding and checkpoint_encoding != MODEL_ENCODING_VERSION:
            raise RuntimeError(
                f"编码版本不匹配：文件为 {checkpoint_encoding}，代码要求 {MODEL_ENCODING_VERSION}"
            )

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(f"模型结构或权重不匹配：{exc}") from exc

        self.model.eval()
        if self.device == "cpu" and torch.get_num_threads() > 8:
            torch.set_num_threads(8)
        with torch.no_grad():
            self.model(torch.zeros(1, 3, 11, 11, device=self.device))
        self.load_ms = (time.perf_counter() - started) * 1000
        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            simulations=simulations,
            c_puct=c_puct,
            time_limit_ms=time_limit_ms,
        )
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed):
        """固定 Python、NumPy、PyTorch 和 CUDA 的随机种子。"""
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        try:
            self.torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass

    def reset(self):
        """MCTS 每步创建新树，不保留跨局状态。"""

    def set_progress_callback(self, callback):
        """??? AIWorker ????????????"""
        self.progress_callback = callback

    def _report_progress(self, progress):
        if self.progress_callback is not None:
            self.progress_callback(dict(progress))

    @staticmethod
    def _heuristic_prior_scores(board, player):
        opponent = BLUE if player == RED else RED
        candidates = set(shortest_path_cells(board, player))
        candidates.update(shortest_path_cells(board, opponent))
        candidates.update(board.legal_moves(player))
        bridges = bridge_move_bonus(board, player)
        return {
            move: candidate_distance_score(board, move, player)
            + 4.0 * bridges.get(move, 0)
            for move in candidates
            if board.is_empty(*move)
        }

    def choose_move(self, board, player):
        if player not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {player!r}")

        tactical = choose_tactical_move(board, player)
        if tactical is not None:
            self._report_progress({"phase": "tactical", "best_move": tactical[0]})
            return tactical

        self._report_progress({"phase": "thinking"})
        move, _, _ = self.mcts.search(board, player, on_progress=self._report_progress)
        if not board.in_range(*move) or not board.is_empty(*move):
            raise RuntimeError(f"MCTS 返回非法着法: {move!r}")
        return move, f"MCTS {self.mcts.last_simulations}次模拟/{self.mcts.last_elapsed_ms:.0f}ms"


def create_ai_with_fallback(model_path=None, **kwargs):
    """为页面创建 AI；模型不可用时返回 BasicAI 和错误说明。"""
    try:
        return NeuralMCTSAI(model_path=model_path, **kwargs), None
    except Exception as exc:
        return BasicAI(), f"模型不可用，已回退基础 AI：{exc}"
