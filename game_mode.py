# -*- coding: utf-8 -*-
"""统一游戏模式定义。"""
from enum import IntEnum


class GameMode(IntEnum):
    HUMAN_VS_HUMAN = 1
    HUMAN_RED_VS_AI = 2
    HUMAN_BLUE_VS_AI = 3
    AI_VS_AI = 4

    @property
    def label(self):
        return {
            GameMode.HUMAN_VS_HUMAN: "双人对战",
            GameMode.HUMAN_RED_VS_AI: "人机对战 · 玩家执红",
            GameMode.HUMAN_BLUE_VS_AI: "人机对战 · 玩家执蓝",
            GameMode.AI_VS_AI: "AI 对 AI 观战",
        }[self]

    @property
    def description(self):
        return {
            GameMode.HUMAN_VS_HUMAN: "两名玩家轮流点击落子",
            GameMode.HUMAN_RED_VS_AI: "玩家先手",
            GameMode.HUMAN_BLUE_VS_AI: "AI 先手",
            GameMode.AI_VS_AI: "空格暂停 / 继续",
        }[self]

    @property
    def human_player(self):
        if self == GameMode.HUMAN_RED_VS_AI:
            return 1
        if self == GameMode.HUMAN_BLUE_VS_AI:
            return 2
        return None

    @property
    def uses_ai(self):
        return self != GameMode.HUMAN_VS_HUMAN
