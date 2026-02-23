"""
U-Flash Adaptive FEC Controller
Dynamically selects FEC mode based on real-time channel quality estimation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class FECMode(Enum):
    NONE = 'none'
    CONV_ONLY = 'conv_only'
    CONV_INTERLEAVER = 'conv_interleaver'
    CONV_INTERLEAVER_RS = 'conv_interleaver_rs'
    HEAVY = 'heavy'  # K=9 + deep interleaver + RS


class FECModeConfig:
    """Configuration for each FEC mode."""

    CONFIGS = {
        FECMode.NONE: {
            'name': 'No FEC',
            'conv_enabled': False,
            'interleaver_enabled': False,
            'rs_enabled': False,
            'code_rate': 1.0,
            'effective_data_rate_ratio': 1.0,
            'min_snr_db': 20,
            'description': 'No error correction. Maximum data rate.',
        },
        FECMode.CONV_ONLY: {
            'name': 'Convolutional Only',
            'conv_enabled': True,
            'conv_K': 7,
            'interleaver_enabled': False,
            'rs_enabled': False,
            'code_rate': 0.5,
            'effective_data_rate_ratio': 0.5,
            'min_snr_db': 12,
            'description': 'K=7 convolutional code. Good for random errors.',
        },
        FECMode.CONV_INTERLEAVER: {
            'name': 'Conv + Interleaver',
            'conv_enabled': True,
            'conv_K': 7,
            'interleaver_enabled': True,
            'interleaver_depth': 20,
            'rs_enabled': False,
            'code_rate': 0.5,
            'effective_data_rate_ratio': 0.5,
            'min_snr_db': 8,
            'description': 'Conv code with block interleaving. Handles burst errors.',
        },
        FECMode.CONV_INTERLEAVER_RS: {
            'name': 'Conv + Interleaver + RS',
            'conv_enabled': True,
            'conv_K': 7,
            'interleaver_enabled': True,
            'interleaver_depth': 20,
            'rs_enabled': True,
            'code_rate': 0.44,
            'effective_data_rate_ratio': 0.44,
            'min_snr_db': 5,
            'description': 'Full FEC stack. Maximum reliability.',
        },
        FECMode.HEAVY: {
            'name': 'Heavy FEC',
            'conv_enabled': True,
            'conv_K': 7,
            'interleaver_enabled': True,
            'interleaver_depth': 40,
            'rs_enabled': True,
            'code_rate': 0.33,
            'effective_data_rate_ratio': 0.33,
            'min_snr_db': 0,
            'description': 'Maximum protection. Deep interleaving + RS. Low data rate.',
        },
    }


class AdaptiveFECController:
    """
    Monitors channel quality and selects optimal FEC mode.
    Balances throughput vs. reliability based on real-time conditions.
    """

    def __init__(self, base_data_rate: float = 180.0):
        self.base_data_rate = base_data_rate  # bps
        self.current_mode = FECMode.CONV_INTERLEAVER
        self.snr_history: list = []
        self.ber_history: list = []
        self.mode_history: list = []
        self.adaptation_interval = 5.0  # seconds
        self.hysteresis_db = 2.0  # Prevent oscillation
        self.max_history = 100

    def estimate_channel_quality(self, snr_db: float, ber: float,
                                  burst_rate: float = 0.0) -> Dict:
        """
        Estimate channel quality from metrics.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            ber: Current bit error rate (before FEC)
            burst_rate: Burst error rate (bursts per second)
        """
        self.snr_history.append(snr_db)
        self.ber_history.append(ber)
        if len(self.snr_history) > self.max_history:
            self.snr_history = self.snr_history[-self.max_history:]
            self.ber_history = self.ber_history[-self.max_history:]

        # Compute averaged metrics
        avg_snr = np.mean(self.snr_history[-10:])
        avg_ber = np.mean(self.ber_history[-10:])
        snr_variance = np.var(self.snr_history[-10:]) if len(self.snr_history) >= 2 else 0

        # Quality label
        if avg_snr > 15:
            quality = 'excellent'
        elif avg_snr > 10:
            quality = 'good'
        elif avg_snr > 5:
            quality = 'fair'
        else:
            quality = 'poor'

        return {
            'quality': quality,
            'avg_snr_db': round(float(avg_snr), 2),
            'avg_ber': float(avg_ber),
            'snr_variance': round(float(snr_variance), 3),
            'burst_rate': float(burst_rate),
            'current_mode': self.current_mode.value,
        }

    # Mode ordering for comparison (lower = less protection)
    MODE_ORDER = {
        FECMode.NONE: 0,
        FECMode.CONV_ONLY: 1,
        FECMode.CONV_INTERLEAVER: 2,
        FECMode.CONV_INTERLEAVER_RS: 3,
        FECMode.HEAVY: 4,
    }

    def select_mode(self, quality: Dict) -> FECMode:
        """Select optimal FEC mode based on channel quality assessment."""
        avg_snr = quality['avg_snr_db']
        q = quality['quality']

        # Determine target mode based on quality
        if q == 'excellent':
            target_mode = FECMode.CONV_ONLY
        elif q == 'good':
            target_mode = FECMode.CONV_INTERLEAVER
        elif q == 'fair':
            target_mode = FECMode.CONV_INTERLEAVER_RS
        else:  # poor
            target_mode = FECMode.HEAVY

        current_order = self.MODE_ORDER[self.current_mode]
        target_order = self.MODE_ORDER[target_mode]

        # Switching to MORE protection: do it immediately (always safe)
        if target_order > current_order:
            new_mode = target_mode
        # Switching to LESS protection: apply hysteresis
        elif target_order < current_order:
            # Only relax if SNR exceeds threshold by hysteresis margin
            target_config = FECModeConfig.CONFIGS[target_mode]
            if avg_snr > target_config['min_snr_db'] + self.hysteresis_db:
                new_mode = target_mode
            else:
                new_mode = self.current_mode
        else:
            new_mode = self.current_mode

        # High burst rate always needs interleaver
        if quality['burst_rate'] > 1.0 and new_mode == FECMode.CONV_ONLY:
            new_mode = FECMode.CONV_INTERLEAVER

        self.current_mode = new_mode
        self.mode_history.append(new_mode.value)
        return new_mode

    def get_mode_config(self, mode: Optional[FECMode] = None) -> Dict:
        """Get configuration for specified or current FEC mode."""
        mode = mode or self.current_mode
        config = FECModeConfig.CONFIGS[mode].copy()
        config['effective_data_rate'] = round(
            self.base_data_rate * config['effective_data_rate_ratio'], 1
        )
        return config

    def adapt(self, snr_db: float, ber: float, burst_rate: float = 0.0) -> Dict:
        """
        Full adaptation cycle: estimate quality + select mode.
        Returns new mode configuration.
        """
        quality = self.estimate_channel_quality(snr_db, ber, burst_rate)
        new_mode = self.select_mode(quality)
        config = self.get_mode_config(new_mode)

        return {
            'channel_quality': quality,
            'selected_mode': new_mode.value,
            'mode_config': config,
            'mode_changed': len(self.mode_history) >= 2 and
                           self.mode_history[-1] != self.mode_history[-2],
        }

    def get_all_modes_summary(self) -> list:
        """Return summary of all available FEC modes."""
        modes = []
        for mode in FECMode:
            config = self.get_mode_config(mode)
            modes.append({
                'mode': mode.value,
                'name': config['name'],
                'code_rate': config['code_rate'],
                'effective_data_rate': config['effective_data_rate'],
                'description': config['description'],
                'is_current': mode == self.current_mode,
            })
        return modes
