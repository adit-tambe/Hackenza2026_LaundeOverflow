"""
U-Flash ML Channel Prediction Module
Uses a simple neural network to predict channel quality
5 seconds ahead for proactive FEC adaptation.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class ChannelPredictor:
    """
    Lightweight channel quality predictor using a simple feedforward network.
    Uses NumPy only (no PyTorch dependency for simplicity).
    
    Predicts channel quality (excellent/good/fair/poor) from time-series
    of SNR, BER, and RoI size measurements.
    """

    QUALITY_LABELS = ['excellent', 'good', 'fair', 'poor']

    def __init__(self, lookback_window: int = 20, prediction_horizon: int = 5):
        """
        Args:
            lookback_window: Number of past measurements to use as input
            prediction_horizon: How many steps ahead to predict
        """
        self.lookback = lookback_window
        self.horizon = prediction_horizon
        self.n_features = 4  # SNR, BER, RoI_size, angular_velocity
        self.n_classes = 4
        self.input_size = self.lookback * self.n_features

        # Simple 2-layer network (random init, will be "trained" on synthetic data)
        np.random.seed(42)
        self.hidden_size = 32
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.n_classes) * 0.1
        self.b2 = np.zeros(self.n_classes)

        # History buffer
        self.history = deque(maxlen=200)
        self._is_trained = False

        # Pre-train on synthetic data
        self._pretrain()

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        h = self._relu(x @ self.W1 + self.b1)
        out = self._softmax(h @ self.W2 + self.b2)
        return out

    def _pretrain(self):
        """Pre-train on synthetic channel data."""
        n_samples = 1000
        X_train = []
        y_train = []

        for _ in range(n_samples):
            # Generate a channel quality trajectory
            quality = np.random.choice(4)
            snr_base = [18, 12, 7, 3][quality]
            ber_base = [1e-4, 1e-3, 5e-3, 2e-2][quality]

            sequence = np.zeros((self.lookback + self.horizon, self.n_features))
            for t in range(self.lookback + self.horizon):
                sequence[t, 0] = snr_base + np.random.normal(0, 2)  # SNR
                sequence[t, 1] = ber_base * (1 + np.random.normal(0, 0.3))  # BER
                sequence[t, 2] = 100 + np.random.normal(0, 20)  # RoI
                sequence[t, 3] = np.random.exponential(5)  # Angular velocity

            # Add quality transitions
            if np.random.random() > 0.7:
                transition_point = np.random.randint(self.lookback - 5, self.lookback)
                new_quality = min(3, quality + 1)
                new_snr = [18, 12, 7, 3][new_quality]
                new_ber = [1e-4, 1e-3, 5e-3, 2e-2][new_quality]
                for t in range(transition_point, self.lookback + self.horizon):
                    sequence[t, 0] = new_snr + np.random.normal(0, 2)
                    sequence[t, 1] = new_ber * (1 + np.random.normal(0, 0.3))
                quality = new_quality

            x = sequence[:self.lookback].flatten()
            # Target: quality at horizon steps ahead
            future_snr = np.mean(sequence[self.lookback:, 0])
            if future_snr > 15:
                y = 0
            elif future_snr > 10:
                y = 1
            elif future_snr > 5:
                y = 2
            else:
                y = 3

            X_train.append(x)
            y_train.append(y)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Normalize features
        self._mean = np.mean(X_train, axis=0)
        self._std = np.std(X_train, axis=0) + 1e-10
        X_norm = (X_train - self._mean) / self._std

        # Simple gradient descent training
        lr = 0.01
        for epoch in range(200):
            # Forward
            H = self._relu(X_norm @ self.W1 + self.b1)
            logits = H @ self.W2 + self.b2
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Cross-entropy gradient
            n = len(y_train)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), y_train] = 1
            d_logits = (probs - one_hot) / n

            # Backward
            dW2 = H.T @ d_logits
            db2 = np.sum(d_logits, axis=0)
            dH = d_logits @ self.W2.T
            dH[H <= 0] = 0  # ReLU gradient

            dW1 = X_norm.T @ dH
            db1 = np.sum(dH, axis=0)

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        self._is_trained = True

    def add_measurement(self, snr_db: float, ber: float,
                       roi_size: float, angular_velocity: float):
        """Add a new channel measurement to the history buffer."""
        self.history.append([snr_db, ber, roi_size, angular_velocity])

    def predict(self) -> Optional[Dict]:
        """
        Predict channel quality for the next time window.
        Returns None if insufficient history.
        """
        if len(self.history) < self.lookback:
            return None

        # Build input from recent history
        recent = list(self.history)[-self.lookback:]
        x = np.array(recent).flatten()

        # Normalize
        x_norm = (x - self._mean) / self._std

        # Predict
        probs = self._forward(x_norm)
        predicted_class = int(np.argmax(probs))

        return {
            'predicted_quality': self.QUALITY_LABELS[predicted_class],
            'probabilities': {
                label: round(float(p), 4)
                for label, p in zip(self.QUALITY_LABELS, probs)
            },
            'confidence': round(float(np.max(probs)), 3),
            'horizon_steps': self.horizon,
            'history_length': len(self.history),
        }

    def predict_trend(self) -> Optional[str]:
        """Predict if channel is improving, stable, or degrading."""
        if len(self.history) < self.lookback:
            return None

        recent_snr = [h[0] for h in list(self.history)[-self.lookback:]]
        first_half = np.mean(recent_snr[:self.lookback // 2])
        second_half = np.mean(recent_snr[self.lookback // 2:])

        diff = second_half - first_half
        if diff > 2:
            return 'improving'
        elif diff < -2:
            return 'degrading'
        else:
            return 'stable'

    def get_summary(self) -> Dict:
        """Return predictor status summary."""
        prediction = self.predict()
        trend = self.predict_trend()

        return {
            'is_trained': self._is_trained,
            'history_length': len(self.history),
            'min_required': self.lookback,
            'ready': len(self.history) >= self.lookback,
            'prediction': prediction,
            'trend': trend,
        }
