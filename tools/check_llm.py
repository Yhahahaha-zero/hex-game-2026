# -*- coding: utf-8 -*-
"""Check LLM configuration and optionally make one minimal live request."""
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.llm_client import LLMClientError, create_llm_client
from ai.llm_config import LLMConfigError, load_llm_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查千问 LLM 配置与连通性")
    parser.add_argument(
        "--live",
        action="store_true",
        help="执行一次最小真实 API 请求（可能产生少量费用）",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="使用 qwen3-vl-plus 测试一张本地图片",
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    try:
        config = load_llm_config()
    except LLMConfigError as exc:
        print(f"[ERROR] 配置无效：{exc}")
        return 2

    summary = config.describe()
    print(
        "config="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    )
    if not args.live:
        print("mode=offline（未发送真实请求）")
        return 0
    if not config.enabled:
        print("[ERROR] LLM_ENABLED=0；请只在本次测试进程中临时设置为 1")
        return 2

    try:
        client = create_llm_client(config)
        if args.image is not None:
            model = config.vision_model
            response = client.analyze_image(
                "这是接口连通性测试。请只回复 OK，不要描述图片。",
                args.image,
                model=model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
        else:
            model = config.analysis_model
            response = client.analyze_text(
                "这是接口连通性测试。请只回复 OK。",
                model=model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
    except (LLMClientError, OSError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1

    result = response.safe_summary()
    result["status"] = "success"
    print("result=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())