"""
U-Flash Convolutional Encoder + Viterbi Decoder
K=7, rate=1/2 with burst-aware soft-decision Viterbi decoding.
Optimized for underwater burst error channels.
"""

import numpy as np
from typing import Tuple, List, Optional


class ConvolutionalEncoder:
    """
    Rate 1/2, constraint length K=7 convolutional encoder.
    Generator polynomials: G1=171o (1111001), G2=133o (1011011)
    """

    def __init__(self, K: int = 7):
        self.K = K
        self.n_states = 2 ** (K - 1)  # 64 states
        # Generator polynomials (binary representation)
        self.g1 = [1, 1, 1, 1, 0, 0, 1]  # 171 octal
        self.g2 = [1, 0, 1, 1, 0, 1, 1]  # 133 octal

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode input bits with convolutional code."""
        # Add tail bits to flush encoder
        input_bits = np.concatenate([bits, np.zeros(self.K - 1, dtype=int)])
        state = np.zeros(self.K, dtype=int)
        output = []

        for bit in input_bits:
            # Shift new bit into register
            state = np.roll(state, 1)
            state[0] = int(bit)

            # Compute output bits
            c0 = int(np.sum(state * self.g1) % 2)
            c1 = int(np.sum(state * self.g2) % 2)
            output.extend([c0, c1])

        return np.array(output, dtype=int)

    def get_rate(self) -> float:
        return 0.5

    def get_overhead_bits(self) -> int:
        return (self.K - 1) * 2  # Tail bits doubled by rate 1/2


class ViterbiDecoder:
    """
    Soft-decision Viterbi decoder with burst-aware branch metrics.
    Uses weighted burst distance metric as described in U-Flash.
    """

    def __init__(self, K: int = 7, burst_weight: float = 1.5):
        self.K = K
        self.n_states = 2 ** (K - 1)
        self.g1 = [1, 1, 1, 1, 0, 0, 1]
        self.g2 = [1, 0, 1, 1, 0, 1, 1]
        self.burst_weight = burst_weight  # w2/w1 ratio for burst penalty

        # Precompute state transition table
        self._build_trellis()

    def _build_trellis(self):
        """Precompute all state transitions and expected outputs."""
        self.next_state = np.zeros((self.n_states, 2), dtype=int)
        self.output = np.zeros((self.n_states, 2, 2), dtype=int)

        for state in range(self.n_states):
            for input_bit in range(2):
                # Build full register: [input_bit, state_bits]
                reg = np.zeros(self.K, dtype=int)
                reg[0] = input_bit
                for i in range(self.K - 1):
                    reg[i + 1] = (state >> (self.K - 2 - i)) & 1

                # Next state
                ns = (input_bit << (self.K - 2)) | (state >> 1)
                self.next_state[state, input_bit] = ns

                # Output bits
                c0 = int(np.sum(reg * self.g1) % 2)
                c1 = int(np.sum(reg * self.g2) % 2)
                self.output[state, input_bit] = [c0, c1]

    def decode(self, received: np.ndarray, confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode received bits using soft-decision Viterbi algorithm.
        
        Args:
            received: Received bit pairs (hard decisions)
            confidence: Per-bit confidence scores [0,1]. None = hard decision.
        
        Returns:
            Decoded bit array
        """
        n_pairs = len(received) // 2
        if confidence is None:
            confidence = np.ones(len(received), dtype=float)

        # Path metrics: cost to reach each state
        INF = 1e9
        path_metric = np.full(self.n_states, INF)
        path_metric[0] = 0.0

        # Survivor paths
        survivors = np.zeros((n_pairs, self.n_states), dtype=int)
        prev_error = np.zeros(self.n_states, dtype=int)  # Track consecutive errors for burst metric

        for t in range(n_pairs):
            r0 = int(received[2 * t])
            r1 = int(received[2 * t + 1])
            c0 = confidence[2 * t]
            c1 = confidence[2 * t + 1]

            new_metric = np.full(self.n_states, INF)
            new_survivors = np.zeros(self.n_states, dtype=int)
            new_errors = np.zeros(self.n_states, dtype=int)

            for state in range(self.n_states):
                if path_metric[state] >= INF:
                    continue

                for input_bit in range(2):
                    ns = self.next_state[state, input_bit]
                    e0, e1 = self.output[state, input_bit]

                    # Soft branch metric with burst awareness
                    bm = 0.0
                    if r0 != e0:
                        bm += (1.0 - c0) + c0 * 1.0  # Penalize disagreement proportional to confidence
                    else:
                        bm += (1.0 - c0) * 0.5  # Small cost for low-confidence agreement

                    if r1 != e1:
                        bm += (1.0 - c1) + c1 * 1.0
                    else:
                        bm += (1.0 - c1) * 0.5

                    # Burst-aware weighting: penalize consecutive errors more
                    is_error = (r0 != e0) or (r1 != e1)
                    if is_error and prev_error[state] > 0:
                        bm *= self.burst_weight  # Increase cost for burst continuation

                    total = path_metric[state] + bm

                    if total < new_metric[ns]:
                        new_metric[ns] = total
                        new_survivors[ns] = state
                        new_errors[ns] = (prev_error[state] + 1) if is_error else 0

            path_metric = new_metric
            survivors[t] = new_survivors
            prev_error = new_errors

        # Traceback from best final state
        best_state = np.argmin(path_metric)
        decoded = np.zeros(n_pairs, dtype=int)

        for t in range(n_pairs - 1, -1, -1):
            prev_state = survivors[t, best_state]
            # Determine input bit from state transition
            for input_bit in range(2):
                if self.next_state[prev_state, input_bit] == best_state:
                    decoded[t] = input_bit
                    break
            best_state = prev_state

        # Remove tail bits
        tail_length = self.K - 1
        if len(decoded) > tail_length:
            decoded = decoded[:len(decoded) - tail_length]

        return decoded

    def decode_hard(self, received: np.ndarray) -> np.ndarray:
        """Hard-decision Viterbi decoding (no confidence scores)."""
        return self.decode(received, confidence=None)
