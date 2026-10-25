# -*- coding: utf-8 -*-
"""V5 policy/value 网络与有限时间 MCTS AI。"""
import hashlib
import json
import random
import time
from pathlib import Path

from ai.evaluation import score_moves, shortest_path_cells
from ai.explanation import AIExplanation
from ai.tactics import (block_multiple_threats, boundary_bridge_bonus,
                        bridge_score_map, find_double_threat_moves,
                        opponent_threats)
from hex_ai import BasicAI, winning_moves
from hex_board import BLUE, RED

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "hex_model_v5.pt"
SELECTED_MODEL_MANIFEST = MODEL_DIR / "selected_model.json"
MODEL_VERSION = "5.0.0"
ENCODING_VERSION = "3x11x11-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_model(manifest_path=None):
    """Load and validate the single deployment-model manifest."""
    manifest_path = Path(manifest_path or SELECTED_MODEL_MANIFEST)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"?????????: {manifest_path}")
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError(f"???? selected_model schema: {document.get('schema_version')!r}")
    model_file = document.get("model_file")
    if not isinstance(model_file, str) or not model_file:
        raise ValueError("selected_model ?? model_file")
    relative = Path(model_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"selected_model ??????: {model_file!r}")
    model_path = (manifest_path.parent / relative).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"???????: {model_path}")
    expected_hash = str(document.get("sha256", "")).lower()
    if not expected_hash or file_sha256(model_path) != expected_hash:
        raise RuntimeError(f"???? SHA256 ????: {model_path}")
    if document.get("model_version") not in (None, "", MODEL_VERSION):
        raise ValueError(f"selected_model ???????: {document.get('model_version')!r}")
    if document.get("encoding_version") not in (None, "", ENCODING_VERSION):
        raise ValueError(
            f"selected_model ???????: {document.get('encoding_version')!r}"
        )
    return model_path, document


def resolve_model_path(model_path=None):
    """Resolve explicit path or the deployment model selected in JSON."""
    if model_path is not None:
        return Path(model_path).expanduser().resolve(), {}
    if SELECTED_MODEL_MANIFEST.is_file():
        return load_selected_model()
    return DEFAULT_MODEL_PATH.resolve(), {}


def choose_tactical_move(board, player):
    """Return an immediate win or the best forced defence when available."""
    wins = winning_moves(board, player)
    if wins:
        return sorted(wins)[0], "直接获胜"

    threats = opponent_threats(board, player)
    if len(threats) == 1:
        return threats[0], "紧急防守"
    if len(threats) >= 2:
        move = block_multiple_threats(board, player)
        if move is not None:
            return move, "多威胁最佳防守"
    return None


class NeuralMCTSAI:
    """神经网络先验 + PUCT MCTS，返回统一 AI 接口格式。"""

    name = "HexMCTS-v5"
    version = MODEL_VERSION
    time_limit_ms = 200

    def __init__(self, model_path=None, device=None, simulations=100,
                 c_puct=1.5, time_limit_ms=200, seed=None):
        self.model_path, self.model_selection = resolve_model_path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"??????????{self.model_path}")

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
        self._cancel_event = None
        self.last_explanation = None

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

    def set_cancel_event(self, cancel_event):
        self._cancel_event = cancel_event

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
        candidates = [move for move in candidates if board.is_empty(*move)]
        bridges = bridge_score_map(board, player)
        distance_scores = score_moves(board, player, candidates)
        return {
            move: distance_scores.get(move, 0.0)
            + bridges.get(move, 0.0)
            + boundary_bridge_bonus(board, move[0], move[1], player)
            for move in candidates
        }

    def _publish_explanation(self, move, tactic, started, heuristic_scores=None,
                             children=None, metadata=None):
        if children:
            candidate_rows = tuple({
                "move": child.move,
                "visits": child.visits,
                "prior": round(child.prior, 6),
                "value": round(child.value, 6),
                "score": child.visits,
                "reason": tactic,
            } for child in children[:5])
            score_type = "访问次数 visits"
        elif heuristic_scores:
            ordered = sorted(
                heuristic_scores,
                key=lambda cell: (-heuristic_scores[cell], cell[0], cell[1]),
            )
            candidate_rows = tuple({
                "move": cell,
                "score": round(float(heuristic_scores[cell]), 4),
                "reason": tactic,
            } for cell in ordered[:5])
            score_type = "距离/桥启发式分"
        else:
            candidate_rows = ({"move": move, "score": None, "reason": tactic},)
            score_type = "战术优先级"
        self.last_explanation = AIExplanation(
            selected_move=move,
            tactic=tactic,
            candidates=candidate_rows,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            search_type="puct-mcts",
            score_type=score_type,
            metadata=dict(metadata or {}),
        )

    def choose_move(self, board, player):
        started = time.perf_counter()
        if player not in (RED, BLUE):
            raise ValueError(f"非法颜色编号: {player!r}")

        tactical = choose_tactical_move(board, player)
        if tactical is not None:
            self._report_progress({"phase": "tactical", "best_move": tactical[0]})
            self._publish_explanation(tactical[0], tactical[1], started)
            return tactical

        heuristic_scores = self._heuristic_prior_scores(board, player)
        ordered = sorted(
            heuristic_scores,
            key=lambda move: (-heuristic_scores[move], move[0], move[1]),
        )
        double_moves = set(
            find_double_threat_moves(board, player, ordered[:30])
        )
        for move in ordered[:30]:
            if move in double_moves:
                tactic = "创造双威胁"
                self._publish_explanation(
                    move, tactic, started, heuristic_scores=heuristic_scores
                )
                return move, tactic

        self._report_progress({"phase": "thinking"})
        result = self.mcts.search_result(
            board,
            player,
            on_progress=self._report_progress,
            heuristic_scores=heuristic_scores,
            cancel_event=self._cancel_event,
        )
        move = result.move
        if not board.in_range(*move) or not board.is_empty(*move):
            raise RuntimeError(f"MCTS 返回非法着法: {move!r}")
        reason = (
            f"MCTS {self.mcts.last_simulations}次模拟/"
            f"{self.mcts.last_elapsed_ms:.0f}ms"
        )
        self._publish_explanation(
            move,
            reason,
            started,
            heuristic_scores=heuristic_scores,
            children=result.children,
            metadata={"simulations": result.simulations},
        )
        return move, reason


def create_ai_with_fallback(model_path=None, **kwargs):
    """为页面创建 AI；模型不可用时返回 BasicAI 和错误说明。"""
    try:
        return NeuralMCTSAI(model_path=model_path, **kwargs), None
    except Exception as exc:
        return BasicAI(), f"模型不可用，已回退基础 AI：{exc}"
