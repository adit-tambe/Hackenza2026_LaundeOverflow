"""
U-Flash IMU-Based Motion Compensation Module
Simulates smartphone IMU sensors and implements motion artifact
detection with decoder confidence weighting.
"""

import numpy as np
from typing import Dict, Tuple, List


class MadgwickFilter:
    """
    Simplified Madgwick orientation filter for sensor fusion.
    Fuses accelerometer and gyroscope data to estimate device orientation.
    """

    def __init__(self, sample_rate: float = 100.0, beta: float = 0.1):
        self.sample_rate = sample_rate
        self.beta = beta
        # Quaternion: [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro: np.ndarray, accel: np.ndarray):
        """Update orientation estimate with new sensor data."""
        q = self.q.copy()
        dt = 1.0 / self.sample_rate

        # Normalize accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-10:
            return
        accel = accel / accel_norm

        # Gradient descent step
        f = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accel[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accel[1],
            2 * (0.5 - q[1]**2 - q[2]**2) - accel[2]
        ])

        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])

        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-10:
            step /= step_norm

        # Quaternion rate from gyroscope
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])

        # Apply correction
        q += (q_dot - self.beta * step) * dt
        q /= np.linalg.norm(q)
        self.q = q

    def get_euler_angles(self) -> Dict[str, float]:
        """Convert quaternion to Euler angles (degrees)."""
        q = self.q
        # Roll (x)
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
        # Pitch (y)
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        pitch = np.degrees(np.arcsin(np.clip(sinp, -1, 1)))
        # Yaw (z)
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        return {'roll': round(roll, 2), 'pitch': round(pitch, 2), 'yaw': round(yaw, 2)}


class MotionCompensator:
    """
    Processes IMU data to detect motion artifacts and
    adjust decoder confidence accordingly.
    """

    def __init__(self, sample_rate: float = 100.0):
        self.filter = MadgwickFilter(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.angular_velocity_threshold = 10.0  # deg/s
        self.high_motion_threshold = 30.0       # deg/s
        self.history: List[Dict] = []
        self.max_history = 500

    def simulate_imu_data(self, duration_s: float, motion_level: str = 'handheld') -> Dict:
        """
        Generate simulated IMU data for testing.
        
        Motion levels:
        - 'stationary': mounted on tripod
        - 'handheld': natural hand tremor  
        - 'walking': walking with device
        - 'swimming': swimming with device
        """
        n_samples = int(duration_s * self.sample_rate)

        motion_params = {
            'stationary': {'gyro_std': 0.5, 'accel_std': 0.05, 'drift': 0.01},
            'handheld': {'gyro_std': 3.0, 'accel_std': 0.3, 'drift': 0.1},
            'walking': {'gyro_std': 8.0, 'accel_std': 1.5, 'drift': 0.5},
            'swimming': {'gyro_std': 15.0, 'accel_std': 3.0, 'drift': 1.0},
        }

        params = motion_params.get(motion_level, motion_params['handheld'])

        # Generate gyroscope data (deg/s)
        t = np.linspace(0, duration_s, n_samples)
        gyro = np.zeros((n_samples, 3))
        for axis in range(3):
            gyro[:, axis] = (
                np.random.normal(0, params['gyro_std'], n_samples) +
                params['drift'] * np.sin(2 * np.pi * 0.5 * t + np.random.random() * np.pi)
            )

        # Add occasional sudden movements
        n_jolts = int(duration_s * 0.5)  # ~0.5 jolts per second
        for _ in range(n_jolts):
            jolt_pos = np.random.randint(0, n_samples)
            jolt_len = np.random.randint(5, 20)
            jolt_end = min(jolt_pos + jolt_len, n_samples)
            gyro[jolt_pos:jolt_end] *= np.random.uniform(2, 5)

        # Generate accelerometer data (m/s^2)
        accel = np.zeros((n_samples, 3))
        accel[:, 2] = 9.81  # Gravity
        for axis in range(3):
            accel[:, axis] += np.random.normal(0, params['accel_std'], n_samples)

        return {
            'gyro': gyro,
            'accel': accel,
            'timestamps': t,
            'motion_level': motion_level,
        }

    def process_imu(self, imu_data: Dict) -> Dict:
        """
        Process IMU data and compute motion metrics.
        Returns per-frame motion status and confidence weights.
        """
        gyro = imu_data['gyro']
        accel = imu_data['accel']
        n_samples = len(gyro)

        # Compute angular velocity magnitude
        angular_vel = np.sqrt(np.sum(gyro**2, axis=1))

        # Run orientation filter
        orientations = []
        for i in range(n_samples):
            self.filter.update(np.radians(gyro[i]), accel[i])
            orientations.append(self.filter.get_euler_angles())

        # Detect motion artifacts
        is_artifact = angular_vel > self.angular_velocity_threshold
        is_high_motion = angular_vel > self.high_motion_threshold

        # Compute confidence weights for decoder
        # Lower confidence during motion artifacts
        confidence_weights = np.ones(n_samples)
        for i in range(n_samples):
            if is_high_motion[i]:
                confidence_weights[i] = 0.2  # Very low confidence
            elif is_artifact[i]:
                confidence_weights[i] = 0.5  # Moderate confidence reduction
            else:
                confidence_weights[i] = 1.0  # Full confidence

        # Smooth confidence weights
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        confidence_weights = np.convolve(confidence_weights, kernel, mode='same')

        # Compute summary statistics
        artifact_pct = np.mean(is_artifact) * 100
        high_motion_pct = np.mean(is_high_motion) * 100
        avg_angular_vel = np.mean(angular_vel)

        result = {
            'angular_velocity': angular_vel.tolist(),
            'confidence_weights': confidence_weights.tolist(),
            'is_artifact': is_artifact.tolist(),
            'orientations': orientations,
            'summary': {
                'artifact_percentage': round(artifact_pct, 2),
                'high_motion_percentage': round(high_motion_pct, 2),
                'avg_angular_velocity': round(float(avg_angular_vel), 2),
                'max_angular_velocity': round(float(np.max(angular_vel)), 2),
                'avg_confidence': round(float(np.mean(confidence_weights)), 3),
                'motion_level': imu_data.get('motion_level', 'unknown'),
            }
        }

        return result

    def apply_to_decoder(self, bit_confidence: np.ndarray,
                         motion_confidence: np.ndarray,
                         frame_rate: float = 30.0) -> np.ndarray:
        """
        Combine bit-level confidence from demodulator with
        motion-level confidence from IMU.
        
        Args:
            bit_confidence: Per-bit confidence from demodulator [0,1]
            motion_confidence: Per-frame confidence from IMU [0,1]
            frame_rate: Camera frame rate
        """
        # Map motion confidence (at IMU rate) to bit rate
        bits_per_frame = max(1, len(bit_confidence) // max(1, int(len(motion_confidence) / (self.sample_rate / frame_rate))))

        combined = bit_confidence.copy()
        for i in range(len(bit_confidence)):
            # Find corresponding motion confidence
            motion_idx = min(int(i / bits_per_frame * (self.sample_rate / frame_rate)),
                           len(motion_confidence) - 1)
            combined[i] *= motion_confidence[motion_idx]

        return combined
