# -*- coding: utf-8 -*-
"""程序合成音效；没有音频设备时自动静默降级。"""
import math
import random
from array import array

import pygame


class SoundManager:
    """生成并播放短促的界面与落子音效，不依赖外部素材。"""

    def __init__(self):
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
        except Exception:
            return
        arpeggio = [
            event
            for index, frequency in enumerate((523.25, 659.25, 783.99, 1046.5))
            for event in (
                (index * 0.12, frequency, 0.5, 7.0, 0.0),
                (index * 0.12, frequency * 2, 0.14, 9.0, 0.0),
            )
        ]
        try:
            self.sounds = {
                "tap": self._synth([(0.0, (560, 260), 0.6, 26.0, 0.35)], 0.13, 0.9),
                "tick": self._synth([(0.0, 1150, 0.5, 110.0, 0.0)], 0.035, 0.22),
                "click": self._synth([(0.0, (760, 430), 0.55, 40.0, 0.15)], 0.08, 0.5),
                "win": self._synth(arpeggio, 0.85, 0.6),
            }
        except Exception:
            self.sounds = {}

    @staticmethod
    def _synth(events, duration, volume):
        sample_rate = 22050
        count = max(1, int(duration * sample_rate))
        buffer = [0.0] * count
        for start, frequency, amplitude, decay, noise in events:
            phase = 0.0
            start_index = int(start * sample_rate)
            for index in range(start_index, count):
                elapsed = (index - start_index) / sample_rate
                if isinstance(frequency, tuple):
                    progress = min(1.0, elapsed / max(duration - start, 1e-6))
                    current_frequency = frequency[0] + (frequency[1] - frequency[0]) * progress
                else:
                    current_frequency = frequency
                phase += 2 * math.pi * current_frequency / sample_rate
                value = math.sin(phase) * amplitude * math.exp(-decay * elapsed)
                if noise:
                    value += (random.random() * 2 - 1) * noise * math.exp(-260 * elapsed)
                buffer[index] += value
        peak = max(1e-9, max(abs(value) for value in buffer))
        samples = array(
            "h",
            (int(value / peak * 32767 * volume) for value in buffer),
        )
        return pygame.mixer.Sound(buffer=samples.tobytes())

    def play(self, name):
        sound = self.sounds.get(name)
        if sound is None:
            return
        try:
            sound.play()
        except pygame.error:
            pass

    def quit(self):
        self.sounds = {}
