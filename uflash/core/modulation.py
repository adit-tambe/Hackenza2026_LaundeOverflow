"""
U-Flash Modulation Module
Implements On-Off Keying (OOK) and Pulse Position Modulation (PPM)
with dispersion-aware slot adjustment for underwater optical communication.
"""

import numpy as np
from typing import Tuple, List


class OOKModulator:
    """On-Off Keying modulation: bit 1 = LED ON, bit 0 = LED OFF."""

    def __init__(self, samples_per_bit: int = 10):
        self.samples_per_bit = samples_per_bit

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bit array to OOK waveform."""
        waveform = np.repeat(bits.astype(float), self.samples_per_bit)
        return waveform

    def demodulate(self, waveform: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert OOK waveform back to bits using threshold detection."""
        n_bits = len(waveform) // self.samples_per_bit
        bits = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            segment = waveform[i * self.samples_per_bit:(i + 1) * self.samples_per_bit]
            bits[i] = 1 if np.mean(segment) > threshold else 0
        return bits

    def soft_demodulate(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return both hard decisions and confidence scores."""
        n_bits = len(waveform) // self.samples_per_bit
        bits = np.zeros(n_bits, dtype=int)
        confidence = np.zeros(n_bits, dtype=float)
        for i in range(n_bits):
            segment = waveform[i * self.samples_per_bit:(i + 1) * self.samples_per_bit]
            mean_val = np.mean(segment)
            bits[i] = 1 if mean_val > 0.5 else 0
            confidence[i] = abs(mean_val - 0.5) * 2  # 0 = no confidence, 1 = full
        return bits, confidence


class PPMModulator:
    """
    Pulse Position Modulation (M-PPM).
    Maps log2(M) bits to one of M time slots within a symbol period.
    Includes dispersion-aware slot boundary adjustment.
    """

    def __init__(self, M: int = 4, samples_per_slot: int = 10):
        self.M = M
        self.bits_per_symbol = int(np.log2(M))
        self.samples_per_slot = samples_per_slot
        self.samples_per_symbol = M * samples_per_slot
        # Dispersion compensation offsets (µs) per color channel
        # Blue arrives first, green second, red last in water
        self.dispersion_offsets = {'blue': 0.0, 'green': 0.3, 'red': 0.8}

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bit array to PPM waveform."""
        # Pad bits to multiple of bits_per_symbol
        pad_len = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

        n_symbols = len(bits) // self.bits_per_symbol
        waveform = np.zeros(n_symbols * self.samples_per_symbol)

        for i in range(n_symbols):
            symbol_bits = bits[i * self.bits_per_symbol:(i + 1) * self.bits_per_symbol]
            slot_index = int(''.join(str(b) for b in symbol_bits), 2)
            start = i * self.samples_per_symbol + slot_index * self.samples_per_slot
            end = start + self.samples_per_slot
            waveform[start:end] = 1.0

        return waveform

    def demodulate(self, waveform: np.ndarray) -> np.ndarray:
        """Convert PPM waveform back to bits."""
        n_symbols = len(waveform) // self.samples_per_symbol
        bits = []

        for i in range(n_symbols):
            symbol = waveform[i * self.samples_per_symbol:(i + 1) * self.samples_per_symbol]
            # Find slot with maximum energy
            energies = np.array([
                np.sum(symbol[s * self.samples_per_slot:(s + 1) * self.samples_per_slot])
                for s in range(self.M)
            ])
            slot_index = np.argmax(energies)
            symbol_bits = format(slot_index, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in symbol_bits])

        return np.array(bits, dtype=int)

    def soft_demodulate(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return hard decisions and per-bit confidence scores."""
        n_symbols = len(waveform) // self.samples_per_symbol
        bits = []
        confidences = []

        for i in range(n_symbols):
            symbol = waveform[i * self.samples_per_symbol:(i + 1) * self.samples_per_symbol]
            energies = np.array([
                np.sum(symbol[s * self.samples_per_slot:(s + 1) * self.samples_per_slot])
                for s in range(self.M)
            ])
            # Softmax-like confidence
            max_energy = np.max(energies)
            total_energy = np.sum(energies) + 1e-10
            confidence = max_energy / total_energy

            slot_index = np.argmax(energies)
            symbol_bits = format(slot_index, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in symbol_bits])
            confidences.extend([confidence] * self.bits_per_symbol)

        return np.array(bits, dtype=int), np.array(confidences, dtype=float)

    def dispersion_compensate(self, rgb_waveforms: dict, distance_m: float) -> np.ndarray:
        """
        Combine RGB channel waveforms with dispersion compensation.
        Adjusts slot boundaries based on predicted arrival times per channel.
        """
        # Dispersion increases with distance
        scale = distance_m / 5.0  # Normalize to 5m reference
        compensated = np.zeros_like(rgb_waveforms.get('blue', rgb_waveforms.get('green')))

        weights = {'blue': 0.5, 'green': 0.35, 'red': 0.15}  # Blue strongest underwater

        for channel, waveform in rgb_waveforms.items():
            offset_samples = int(self.dispersion_offsets.get(channel, 0) * scale * self.samples_per_slot)
            shifted = np.roll(waveform, -offset_samples)
            compensated += weights.get(channel, 0.33) * shifted

        return compensated


class AdaptiveThresholdDemodulator:
    """
    Adaptive threshold demodulation that dynamically adjusts
    the decision threshold based on local signal statistics.
    Handles ambient light variations and RoI changes.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def demodulate(self, waveform: np.ndarray, samples_per_bit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate with sliding-window adaptive threshold."""
        n_bits = len(waveform) // samples_per_bit
        bits = np.zeros(n_bits, dtype=int)
        confidence = np.zeros(n_bits, dtype=float)

        for i in range(n_bits):
            # Local window for threshold estimation
            center = i * samples_per_bit + samples_per_bit // 2
            win_start = max(0, center - self.window_size * samples_per_bit // 2)
            win_end = min(len(waveform), center + self.window_size * samples_per_bit // 2)
            local_window = waveform[win_start:win_end]

            # Adaptive threshold: midpoint between local min and max
            local_min = np.percentile(local_window, 10)
            local_max = np.percentile(local_window, 90)
            threshold = (local_min + local_max) / 2

            segment = waveform[i * samples_per_bit:(i + 1) * samples_per_bit]
            mean_val = np.mean(segment)
            bits[i] = 1 if mean_val > threshold else 0

            # Confidence based on distance from threshold
            range_val = max(local_max - local_min, 1e-10)
            confidence[i] = min(abs(mean_val - threshold) / (range_val / 2), 1.0)

        return bits, confidence
