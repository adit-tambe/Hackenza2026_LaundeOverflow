"""
U-Flash End-to-End Simulation Engine
Runs the complete transmission pipeline:
Text -> Encode -> Modulate -> Channel -> Demodulate -> Decode -> Text
With all modules integrated.
"""

import numpy as np
import time
from typing import Dict, Optional

from core.modulation import PPMModulator, AdaptiveThresholdDemodulator
from core.rll import AdaptiveRLLEncoder
from core.convolutional import ConvolutionalEncoder, ViterbiDecoder
from core.interleaver import BlockInterleaver
from core.reed_solomon import ReedSolomonCodec
from core.channel import UnderwaterChannel
from core.framing import Frame, FrameManager, CRC16
from core.adaptive_fec import AdaptiveFECController, FECMode
from core.water_quality import WaterQualityEstimator
from core.motion_compensation import MotionCompensator
from core.channel_prediction import ChannelPredictor


class SimulationEngine:
    """
    Complete U-Flash simulation engine integrating all modules.
    """

    def __init__(self):
        # Core modules
        self.ppm = PPMModulator(M=4, samples_per_slot=10)
        self.demodulator = AdaptiveThresholdDemodulator(window_size=30)
        self.rll = AdaptiveRLLEncoder()
        self.conv_encoder = ConvolutionalEncoder(K=7)
        self.viterbi = ViterbiDecoder(K=7, burst_weight=1.5)
        self.interleaver = BlockInterleaver(depth=20, width=6)
        self.rs = ReedSolomonCodec(n=255, k=223)
        self.channel = UnderwaterChannel()
        self.frame_mgr = FrameManager(device_id=1)

        # Advanced modules
        self.adaptive_fec = AdaptiveFECController(base_data_rate=180)
        self.water_quality = WaterQualityEstimator()
        self.motion_comp = MotionCompensator(sample_rate=100)
        self.channel_pred = ChannelPredictor(lookback_window=20, prediction_horizon=5)

        # State
        self.last_result = None
        self.transmission_count = 0

    def text_to_bits(self, text: str) -> np.ndarray:
        """Convert text string to bit array."""
        bits = []
        for char in text.encode('utf-8'):
            for j in range(7, -1, -1):
                bits.append((char >> j) & 1)
        return np.array(bits, dtype=int)

    def bits_to_text(self, bits: np.ndarray) -> str:
        """Convert bit array back to text string."""
        n_bytes = len(bits) // 8
        chars = []
        for i in range(n_bytes):
            byte = 0
            for j in range(8):
                if i * 8 + j < len(bits):
                    byte |= int(bits[i * 8 + j]) << (7 - j)
            chars.append(byte)
        try:
            return bytes(chars).decode('utf-8', errors='replace')
        except Exception:
            return ''.join(chr(c) if 32 <= c < 127 else '?' for c in chars)

    def run_transmission(self, message: str, channel_params: Optional[Dict] = None,
                        motion_level: str = 'handheld') -> Dict:
        """
        Run full end-to-end transmission simulation.
        
        Args:
            message: Text message to transmit
            channel_params: Optional channel parameter overrides
            motion_level: IMU motion simulation level
        """
        start_time = time.time()
        stages = {}
        original_bits_len = 0

        # Guard: empty message
        if not message:
            elapsed = time.time() - start_time
            self.transmission_count += 1
            return {
                'success': True, 'original_message': '', 'decoded_message': '',
                'ber': '0.00e+00', 'ber_float': 0.0, 'bit_errors': 0,
                'total_bits': 0, 'elapsed_ms': round(elapsed * 1000, 2),
                'transmission_id': self.transmission_count,
                'fec_mode': self.adaptive_fec.current_mode.value,
                'channel_quality': 'good',
                'waveforms': {'tx': [], 'rx': []}, 'stages': {},
            }

        # 1. Text to Bits
        data_bits = self.text_to_bits(message)
        original_bits_len = len(data_bits)
        stages['input'] = {
            'message': message,
            'n_bits': len(data_bits),
            'n_bytes': len(message.encode('utf-8')),
        }

        # 2. Configure Channel
        if channel_params:
            self.channel.set_parameters(**channel_params)

        # 3. Adaptive FEC Decision
        # Use previous channel info or defaults
        fec_result = self.adaptive_fec.adapt(
            snr_db=self.channel.snr_db,
            ber=self.channel.average_ber,
            burst_rate=self.channel.p_gb * 180  # Approximate burst rate
        )
        current_mode = FECMode(fec_result['selected_mode'])
        mode_config = fec_result['mode_config']

        stages['fec_adaptation'] = fec_result

        # 4. RLL Encoding
        self.rll.adapt(self.channel.ambient_lux)
        rll_encoded = self.rll.encode(data_bits)
        stages['rll'] = {
            'input_bits': len(data_bits),
            'output_bits': len(rll_encoded),
            'config': self.rll.get_config_name(),
            'overhead': f'{len(rll_encoded)/len(data_bits):.2f}x',
        }

        # 5. Convolutional Encoding (if enabled)
        if mode_config.get('conv_enabled', True):
            conv_encoded = self.conv_encoder.encode(rll_encoded)
            stages['convolutional'] = {
                'input_bits': len(rll_encoded),
                'output_bits': len(conv_encoded),
                'rate': '1/2',
                'constraint_length': 7,
            }
        else:
            conv_encoded = rll_encoded
            stages['convolutional'] = {'skipped': True}

        # 6. Interleaving (if enabled)
        if mode_config.get('interleaver_enabled', True):
            interleaved = self.interleaver.interleave(conv_encoded)
            stages['interleaver'] = {
                'input_bits': len(conv_encoded),
                'output_bits': len(interleaved),
                'depth': self.interleaver.depth,
                'width': self.interleaver.width,
                'burst_protection': f'{self.interleaver.get_burst_protection()} bits',
            }
        else:
            interleaved = conv_encoded
            stages['interleaver'] = {'skipped': True}

        # 7. PPM Modulation (for waveform visualization only)
        waveform_tx = self.ppm.modulate(interleaved)
        stages['modulation'] = {
            'input_bits': len(interleaved),
            'waveform_samples': len(waveform_tx),
            'type': f'{self.ppm.M}-PPM',
            'samples_per_slot': self.ppm.samples_per_slot,
        }

        # Store TX waveform for visualization (downsample for JSON)
        max_display_points = 500
        if len(waveform_tx) > max_display_points:
            indices = np.linspace(0, len(waveform_tx) - 1, max_display_points, dtype=int)
            tx_display = waveform_tx[indices].tolist()
        else:
            tx_display = waveform_tx.tolist()

        # 8. Channel Transmission (bit-level for actual data)
        received_bits, bit_confidence, channel_metrics = self.channel.transmit_bits(interleaved)
        stages['channel'] = channel_metrics

        # Generate RX waveform for visualization
        waveform_rx_viz = self.ppm.modulate(received_bits)
        noise = np.random.normal(0, 0.05, len(waveform_rx_viz))
        waveform_rx_viz = np.clip(waveform_rx_viz + noise, 0, 1)
        if len(waveform_rx_viz) > max_display_points:
            indices_rx = np.linspace(0, len(waveform_rx_viz) - 1, max_display_points, dtype=int)
            rx_display = waveform_rx_viz[indices_rx].tolist()
        else:
            rx_display = waveform_rx_viz.tolist()

        # 9. IMU Motion Simulation + Compensation
        duration_s = max(0.5, len(interleaved) / 180.0)
        imu_data = self.motion_comp.simulate_imu_data(duration_s, motion_level)
        motion_result = self.motion_comp.process_imu(imu_data)
        stages['motion'] = motion_result['summary']

        # 10. Apply motion compensation to confidence
        motion_conf = np.array(motion_result['confidence_weights'])
        motion_conf_resampled = np.interp(
            np.linspace(0, 1, len(bit_confidence)),
            np.linspace(0, 1, len(motion_conf)),
            motion_conf
        )
        combined_confidence = bit_confidence * motion_conf_resampled

        demod_bits = received_bits
        demod_confidence = combined_confidence

        stages['demodulation'] = {
            'output_bits': len(demod_bits),
            'avg_confidence': round(float(np.mean(bit_confidence)), 3),
            'avg_motion_confidence': round(float(np.mean(motion_conf_resampled)), 3),
            'avg_combined_confidence': round(float(np.mean(combined_confidence)), 3),
        }


        # 11. Deinterleaving
        if mode_config.get('interleaver_enabled', True):
            deinterleaved = self.interleaver.deinterleave(demod_bits)
            deint_confidence = self.interleaver.deinterleave_float(combined_confidence)
        else:
            deinterleaved = demod_bits
            deint_confidence = combined_confidence

        # 12. Viterbi Decoding
        if mode_config.get('conv_enabled', True):
            decoded_rll = self.viterbi.decode(deinterleaved, deint_confidence)
        else:
            decoded_rll = deinterleaved

        # 13. RLL Decoding
        decoded_bits = self.rll.decode(decoded_rll)

        # Truncate to original length
        decoded_bits = decoded_bits[:original_bits_len]
        if len(decoded_bits) < original_bits_len:
            pad = original_bits_len - len(decoded_bits)
            decoded_bits = np.concatenate([decoded_bits, np.zeros(pad, dtype=int)])

        # 14. Compute BER
        ber = float(np.sum(data_bits != decoded_bits)) / max(len(data_bits), 1)
        bit_errors = int(np.sum(data_bits != decoded_bits))

        stages['decoding'] = {
            'decoded_bits': len(decoded_bits),
            'bit_errors': bit_errors,
            'ber': f'{ber:.2e}',
            'ber_float': ber,
        }

        # 15. Reconstruct text
        decoded_message = self.bits_to_text(decoded_bits)
        stages['output'] = {
            'decoded_message': decoded_message,
            'match': decoded_message == message,
            'char_errors': sum(1 for a, b in zip(message, decoded_message) if a != b),
        }

        # 16. Water Quality Estimation
        wq = self.water_quality.get_full_assessment(
            roi_size=channel_metrics.get('roi_size', 100),
            distance_m=self.channel.distance_m,
            rgb_intensities={
                'blue': 0.5 * self.channel.signal_power,
                'green': 0.35 * self.channel.signal_power,
                'red': 0.15 * self.channel.signal_power,
            }
        )
        stages['water_quality'] = wq

        # 17. Channel Prediction
        self.channel_pred.add_measurement(
            snr_db=channel_metrics.get('snr_db', 10),
            ber=ber,
            roi_size=channel_metrics.get('roi_size', 100),
            angular_velocity=motion_result['summary'].get('avg_angular_velocity', 5)
        )
        prediction = self.channel_pred.get_summary()
        stages['channel_prediction'] = prediction

        # Timing
        elapsed = time.time() - start_time
        self.transmission_count += 1

        # Final result
        result = {
            'success': decoded_message == message,
            'original_message': message,
            'decoded_message': decoded_message,
            'ber': f'{ber:.2e}',
            'ber_float': ber,
            'bit_errors': bit_errors,
            'total_bits': original_bits_len,
            'elapsed_ms': round(elapsed * 1000, 2),
            'transmission_id': self.transmission_count,
            'fec_mode': fec_result['selected_mode'],
            'channel_quality': fec_result['channel_quality']['quality'],
            'waveforms': {
                'tx': tx_display,
                'rx': rx_display,
            },
            'stages': stages,
        }

        self.last_result = result
        return result

    def run_ber_sweep(self, message: str = "Hello", distances: list = None) -> Dict:
        """Run BER measurement across multiple distances."""
        if distances is None:
            distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        results = []
        for d in distances:
            bers = []
            for trial in range(5):
                r = self.run_transmission(message, {'distance_m': d})
                bers.append(r['ber_float'])
            avg_ber = np.mean(bers)
            results.append({
                'distance_m': d,
                'avg_ber': float(avg_ber),
                'ber_log': float(np.log10(max(avg_ber, 1e-7))),
            })

        return {'sweep_results': results, 'message': message}

    def get_system_status(self) -> Dict:
        """Get current system status for dashboard."""
        return {
            'channel': self.channel.get_state_dict(),
            'fec_mode': self.adaptive_fec.current_mode.value,
            'fec_modes_available': self.adaptive_fec.get_all_modes_summary(),
            'rll_config': self.rll.get_config_name(),
            'motion_level': 'handheld',
            'transmissions': self.transmission_count,
            'channel_prediction': self.channel_pred.get_summary(),
        }
