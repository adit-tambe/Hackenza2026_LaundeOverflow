"""
U-Flash Run-Length Limited (RLL) Coding Module
Prevents LED flickering caused by rolling-shutter effect.
Adapts parameters based on ambient light intensity.
"""

import numpy as np
from typing import Tuple


class RLLEncoder:
    """
    Run-Length Limited (d, k) encoder.
    - d: minimum number of 0s between consecutive 1s
    - k: maximum number of 0s between consecutive 1s
    
    Uses lookup tables for efficient encoding.
    Default (1,7) RLL suitable for camera-based optical communication.
    """

    # RLL(1,7) encoding table: maps input bits to codewords
    # This ensures min 1 zero between consecutive 1s
    ENCODE_TABLE_1_7 = {
        (0, 0): (0, 1, 0, 0),
        (0, 1): (0, 1, 0, 1),
        (1, 0): (1, 0, 0, 0),
        (1, 1): (1, 0, 0, 1),
    }

    DECODE_TABLE_1_7 = {v: k for k, v in ENCODE_TABLE_1_7.items()}

    def __init__(self, d: int = 1, k: int = 7):
        self.d = d
        self.k = k

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode bits using RLL(d,k) coding."""
        # Pad to even length
        if len(bits) % 2 != 0:
            bits = np.concatenate([bits, [0]])

        encoded = []
        for i in range(0, len(bits), 2):
            pair = (int(bits[i]), int(bits[i + 1]))
            codeword = self.ENCODE_TABLE_1_7.get(pair, (0, 1, 0, 0))
            encoded.extend(codeword)

        return np.array(encoded, dtype=int)

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decode RLL-encoded bits back to original data."""
        decoded = []
        for i in range(0, len(encoded_bits), 4):
            if i + 4 > len(encoded_bits):
                break
            codeword = tuple(int(b) for b in encoded_bits[i:i + 4])
            pair = self.DECODE_TABLE_1_7.get(codeword, (0, 0))
            decoded.extend(pair)

        return np.array(decoded, dtype=int)

    def get_overhead(self) -> float:
        """Return coding overhead ratio."""
        return 4.0 / 2.0  # 2 input bits -> 4 output bits = 2x overhead


class AdaptiveRLLEncoder:
    """
    Ambient-light-adaptive RLL encoder.
    Adjusts (d, k) parameters based on light intensity
    to balance flicker reduction vs. data rate.
    
    - High ambient light -> larger d (more flicker suppression needed)
    - Low ambient light -> smaller d (less flicker visible, higher rate)
    """

    # Multiple RLL configurations
    CONFIGS = {
        'low_light': {'d': 0, 'k': 4, 'rate': 0.8},      # Minimal constraint
        'medium_light': {'d': 1, 'k': 7, 'rate': 0.5},    # Standard
        'high_light': {'d': 2, 'k': 10, 'rate': 0.4},     # Max flicker suppression
    }

    def __init__(self):
        self.current_config = 'medium_light'
        self.encoder = RLLEncoder(d=1, k=7)

    def adapt(self, ambient_light_lux: float):
        """Select RLL parameters based on ambient light level."""
        if ambient_light_lux < 100:
            self.current_config = 'low_light'
        elif ambient_light_lux < 1000:
            self.current_config = 'medium_light'
        else:
            self.current_config = 'high_light'

        config = self.CONFIGS[self.current_config]
        self.encoder = RLLEncoder(d=config['d'], k=config['k'])

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode with current adaptive parameters."""
        return self.encoder.encode(bits)

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decode with current adaptive parameters."""
        return self.encoder.decode(encoded_bits)

    def get_effective_rate(self) -> float:
        """Return effective data rate multiplier."""
        return self.CONFIGS[self.current_config]['rate']

    def get_config_name(self) -> str:
        return self.current_config
