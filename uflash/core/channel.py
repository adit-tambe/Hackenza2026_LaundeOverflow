"""
U-Flash Underwater Channel Simulator
Gilbert-Elliott burst error model with physically-accurate
scattering, absorption, and ambient light modeling.
"""

import numpy as np
from typing import Tuple, Dict


class UnderwaterChannel:
    """
    Simulates underwater optical channel with:
    - Beer-Lambert absorption
    - Mie/Rayleigh scattering (positive gain for camera-based systems)
    - Gilbert-Elliott burst error model
    - Ambient light interference
    - Depth-dependent performance
    """

    # Water type absorption coefficients (per meter)
    WATER_TYPES = {
        'clear': {'absorption': 0.05, 'scattering': 0.02, 'turbidity_ntu': 0.5},
        'coastal': {'absorption': 0.12, 'scattering': 0.10, 'turbidity_ntu': 5.0},
        'harbor': {'absorption': 0.30, 'scattering': 0.25, 'turbidity_ntu': 20.0},
        'turbid': {'absorption': 0.50, 'scattering': 0.40, 'turbidity_ntu': 50.0},
    }

    # Wavelength-dependent absorption (per meter)
    WAVELENGTH_ABSORPTION = {
        'blue': 0.015,   # 470nm - best underwater
        'green': 0.040,  # 520nm
        'red': 0.350,    # 630nm - worst underwater
    }

    def __init__(self, water_type: str = 'coastal', distance_m: float = 5.0,
                 depth_m: float = 3.0, ambient_lux: float = 500.0):
        self.water_type = water_type
        self.distance_m = distance_m
        self.depth_m = depth_m
        self.ambient_lux = ambient_lux

        # Gilbert-Elliott model parameters
        self.p_gb = 0.02   # Prob: Good -> Bad (burst start)
        self.p_bg = 0.3    # Prob: Bad -> Good (burst end)
        self.error_rate_good = 0.001  # BER in good state
        self.error_rate_bad = 0.3     # BER in bad state
        self.state = 'good'  # Current channel state

        self._update_parameters()

    def _update_parameters(self):
        """Update channel parameters based on conditions."""
        wp = self.WATER_TYPES.get(self.water_type, self.WATER_TYPES['coastal'])

        # Attenuation increases with distance (Beer-Lambert law)
        total_attenuation = (wp['absorption'] + wp['scattering']) * self.distance_m
        self.signal_power = np.exp(-total_attenuation)

        # Scattering gain: RoI enlargement factor (unique to camera-based systems)
        # Scattering is BENEFICIAL up to a point
        self.scattering_gain = 1.0 + wp['scattering'] * self.distance_m * 0.5
        if wp['scattering'] * self.distance_m > 3.0:
            # Beyond saturation point, scattering becomes harmful
            excess = wp['scattering'] * self.distance_m - 3.0
            self.scattering_gain *= np.exp(-excess * 0.3)

        # RoI size in pixels (normalized)
        self.roi_size = 50 + self.scattering_gain * self.distance_m * 20
        self.roi_size = min(self.roi_size, 500)  # Camera limit

        # Ambient light interference (decreases with depth)
        surface_lux = self.ambient_lux
        self.effective_ambient = surface_lux * np.exp(-0.1 * self.depth_m)

        # SNR calculation
        noise_power = self.effective_ambient / 10000 + 0.01
        self.snr_linear = (self.signal_power * self.scattering_gain) / noise_power
        self.snr_db = 10 * np.log10(max(self.snr_linear, 1e-10))

        # Update Gilbert-Elliott parameters based on conditions
        turbulence_factor = 1.0 + wp['turbidity_ntu'] / 50.0
        self.p_gb = min(0.02 * turbulence_factor, 0.15)
        self.error_rate_good = min(0.001 * (1 + self.distance_m / 10), 0.05)

        # Compute effective BER
        p_good = self.p_bg / (self.p_gb + self.p_bg)
        p_bad = self.p_gb / (self.p_gb + self.p_bg)
        self.average_ber = p_good * self.error_rate_good + p_bad * self.error_rate_bad

    def set_parameters(self, **kwargs):
        """Update channel parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._update_parameters()

    def transmit(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Transmit signal through underwater channel.
        Returns received signal and channel metrics.
        """
        received = signal.copy().astype(float)

        # Apply attenuation
        received *= self.signal_power * self.scattering_gain

        # Add ambient light noise
        noise_std = np.sqrt(self.effective_ambient / 10000 + 0.01)
        noise = np.random.normal(0, noise_std, len(received))
        received += noise

        # Apply Gilbert-Elliott burst errors
        error_mask = self._gilbert_elliott_errors(len(signal))
        bit_errors = np.sum(error_mask)

        # Apply errors to signal (flip bits)
        for i in range(len(received)):
            if error_mask[i]:
                received[i] = 1.0 - received[i] if received[i] > 0.5 else received[i] + 0.5

        # Clip to valid range
        received = np.clip(received, 0, 1)

        # Generate RGB channel signals (for dispersion-aware decoding)
        rgb_signals = self._apply_chromatic_dispersion(received)

        metrics = {
            'snr_db': float(self.snr_db),
            'ber_raw': float(bit_errors / max(len(signal), 1)),
            'bit_errors': int(bit_errors),
            'signal_power': float(self.signal_power),
            'scattering_gain': float(self.scattering_gain),
            'roi_size': float(self.roi_size),
            'effective_ambient': float(self.effective_ambient),
            'channel_state_pct_bad': float(self.p_gb / (self.p_gb + self.p_bg)),
            'avg_burst_length': float(1.0 / self.p_bg) if self.p_bg > 0 else 0,
            'water_type': self.water_type,
            'distance_m': self.distance_m,
            'depth_m': self.depth_m,
        }

        return received, metrics

    def transmit_bits(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Transmit bit array through channel.
        Returns received bits, confidence scores, and metrics.
        """
        # Apply Gilbert-Elliott errors
        error_mask = self._gilbert_elliott_errors(len(bits))
        received_bits = bits.copy()
        received_bits[error_mask] = 1 - received_bits[error_mask]

        # Generate confidence scores (lower near errors)
        confidence = np.ones(len(bits), dtype=float)
        for i in range(len(bits)):
            # Base confidence from SNR
            base_conf = min(self.snr_linear / 20.0, 0.99)

            # Reduce confidence in bad channel states
            if error_mask[i]:
                confidence[i] = np.random.uniform(0.1, 0.4)
            else:
                confidence[i] = np.random.uniform(base_conf * 0.8, min(base_conf * 1.1, 0.99))

        metrics = {
            'snr_db': float(self.snr_db),
            'ber_raw': float(np.sum(error_mask) / max(len(bits), 1)),
            'bit_errors': int(np.sum(error_mask)),
            'total_bits': int(len(bits)),
            'scattering_gain': float(self.scattering_gain),
            'roi_size': float(self.roi_size),
            'avg_confidence': float(np.mean(confidence)),
            'water_type': self.water_type,
            'distance_m': self.distance_m,
            'depth_m': self.depth_m,
        }

        return received_bits, confidence, metrics

    def _gilbert_elliott_errors(self, length: int) -> np.ndarray:
        """Generate burst errors using Gilbert-Elliott model."""
        errors = np.zeros(length, dtype=bool)
        state = self.state

        for i in range(length):
            if state == 'good':
                errors[i] = np.random.random() < self.error_rate_good
                if np.random.random() < self.p_gb:
                    state = 'bad'
            else:  # bad state
                errors[i] = np.random.random() < self.error_rate_bad
                if np.random.random() < self.p_bg:
                    state = 'good'

        self.state = state
        return errors

    def _apply_chromatic_dispersion(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate chromatic dispersion: different wavelengths
        arrive at different times in water.
        """
        dispersion_per_meter = {'blue': 0, 'green': 1, 'red': 3}  # samples delay
        rgb = {}

        for color, delay_per_m in dispersion_per_meter.items():
            delay = int(delay_per_m * self.distance_m / 5)
            atten = self.WAVELENGTH_ABSORPTION[color] * self.distance_m
            channel_signal = signal * np.exp(-atten)
            rgb[color] = np.roll(channel_signal, delay)

        return rgb

    def get_quality_label(self) -> str:
        """Return human-readable channel quality."""
        if self.snr_db > 15:
            return 'Excellent'
        elif self.snr_db > 10:
            return 'Good'
        elif self.snr_db > 5:
            return 'Fair'
        else:
            return 'Poor'

    def get_state_dict(self) -> Dict:
        """Return full channel state for dashboard display."""
        return {
            'water_type': self.water_type,
            'distance_m': self.distance_m,
            'depth_m': self.depth_m,
            'ambient_lux': self.ambient_lux,
            'snr_db': round(self.snr_db, 2),
            'quality': self.get_quality_label(),
            'signal_power': round(self.signal_power, 4),
            'scattering_gain': round(self.scattering_gain, 2),
            'roi_size': round(self.roi_size, 1),
            'effective_ambient': round(self.effective_ambient, 2),
            'average_ber': f'{self.average_ber:.2e}',
        }
