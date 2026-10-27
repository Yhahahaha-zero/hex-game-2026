# -*- coding: utf-8 -*-
"""Evaluate HybridAI + LLM candidate coordination on fixed positions.

Default mode is offline Mock. Pass --live only when a real API experiment is
intended; the live mode supports the position cache and does not print secrets.
"""
import argparse
from datetime import datetime
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.ai_coordinator import AICoordinator, CandidateAnalysisCache
from ai.analysis_result import AnalysisResult, CandidateAnalysis
from ai.hybrid_ai import HybridAI
from hex_board import BLUE, RED, HexBoard


DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "evaluation" / "llm_candidate_results"
DEFAULT_SEED = 20261025


class MockCandidateAdvisor:
    model = "mock-llm"

    def analyze(self, observation):
        candidates = [tuple(item["move"]) for item in observation.get("candidates", ())]
        recommended = candidates[1] if len(candidates) > 1 else (candidates[0] if candidates else None)
        analyses = []
        for index, move in enumerate(candidates[:8]):
            analyses.append(CandidateAnalysis(
                move=move,
                preference="prefer" if move == recommended else "neutral",
                pros="离线 Mock 候选偏好",
                cons="不代表真实模型判断",
            ))
        return AnalysisResult(
            schema_version=1,
            prompt_version="offline-evaluation-v1",
            position_id=observation["position_id"],
            summary="离线固定局面评测",
            priority="balanced",
            game_state="playing",
            is_terminal=False,
            candidates=tuple(analyses),
            recommended_move=recommended,
            subjective_confidence=0.5,
            warnings=("离线 Mock 结果，仅用于验证流程",),
            model_name=self.model,
            elapsed_ms=0.0,
        )


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def generate_positions(count, seed):
    rng = random.Random(seed)
    positions = []
    attempts = 0
    while len(positions) < count and attempts < max(50, count * 20):
        attempts += 1
        board = HexBoard()
        target = rng.randint(8, 30)
        for ply in range(target):
            player = RED if ply % 2 == 0 else BLUE
            moves = [
                move for move in board.legal_moves()
                if not board.would_win(move[0], move[1], player)
            ]
            if not moves:
                break
            q, r = rng.choice(moves)
            board.place(q, r, player)
        if board.has_won(RED) or board.has_won(BLUE):
            continue
        next_player = RED if len(board.history) % 2 == 0 else BLUE
        positions.append((board, next_player))
    return positions


def _live_advisor(config):
    from ai.llm_advisor import LLMAdvisor
    from ai.llm_client import create_llm_client
    return LLMAdvisor(
        create_llm_client(config),
        timeout=config.analysis_timeout_seconds,
        max_tokens=config.max_output_tokens,
        enable_thinking=config.enable_thinking,
    )


def run_evaluation(*, positions=20, seed=DEFAULT_SEED, mode="analysis_only",
                   alpha=0.0, candidates=8, live=False, output_dir=None,
                   name=None):
    if live:
        from ai.llm_config import load_llm_config
        config = load_llm_config()
        if not config.enabled or not config.key_configured:
            raise RuntimeError("实时评测需要 LLM_ENABLED=1 和有效的 LLM_API_KEY")
        advisor = _live_advisor(config)
        model_name = config.analysis_model
        cache = CandidateAnalysisCache() if config.cache_enabled else None
    else:
        config = None
        advisor = MockCandidateAdvisor()
        model_name = advisor.model
        cache = None

    positions_list = generate_positions(int(positions), int(seed))
    results = []
    for index, (board, player) in enumerate(positions_list):
        coordinator = AICoordinator(
            HybridAI(time_limit=0.08, seed=int(seed) + index),
            advisor,
            mode=mode,
            alpha=alpha,
            candidate_limit=candidates,
            cache=cache,
            model_name=model_name,
        )
        try:
            move, reason = coordinator.decide(board, player)
            result = coordinator.last_result
            row = {
                "index": index,
                "move": list(move),
                "reason": reason,
                **result.as_dict(),
            }
        except Exception as exc:
            row = {
                "index": index,
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            }
        results.append(row)

    completed = [row for row in results if "error" not in row]
    changed = sum(row["base_move"] != row["final_move"] for row in completed)
    fallbacks = sum(row["decision_source"] == "fallback" for row in completed)
    cache_hits = sum(bool(row["cache_hit"]) for row in completed)
    durations = [float(row["elapsed_ms"]) for row in completed]
    summary = {
        "positions_requested": int(positions),
        "positions_completed": len(completed),
        "errors": len(results) - len(completed),
        "mode": mode,
        "alpha": float(alpha),
        "candidate_limit": int(candidates),
        "live": bool(live),
        "model_name": model_name,
        "changed_decisions": changed,
        "changed_rate": round(changed / len(completed), 4) if completed else 0.0,
        "fallbacks": fallbacks,
        "fallback_rate": round(fallbacks / len(completed), 4) if completed else 0.0,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / len(completed), 4) if completed else 0.0,
        "elapsed_ms_p50": _percentile(durations, 0.50),
        "elapsed_ms_p95": _percentile(durations, 0.95),
        "elapsed_ms_max": round(max(durations), 3) if durations else 0.0,
    }
    document = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "summary": summary,
        "results": results,
    }
    root = Path(output_dir or DEFAULT_OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    stem = name or f"llm_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = root / f"{stem}.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document, path


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定局面 HybridAI+LLM 候选融合评测")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=("analysis_only", "rerank"),
                        default="analysis_only")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--live", action="store_true",
                        help="调用真实模型；默认离线 Mock，不产生费用")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    try:
        document, path = run_evaluation(
            positions=args.positions,
            seed=args.seed,
            mode=args.mode,
            alpha=args.alpha,
            candidates=args.candidates,
            live=args.live,
            output_dir=args.output_dir,
            name=args.name,
        )
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    summary = document["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())