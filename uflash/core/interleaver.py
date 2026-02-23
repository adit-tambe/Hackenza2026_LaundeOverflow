"""
U-Flash Block Interleaver
Spreads burst errors across multiple codewords to convert
them into random-like errors for the FEC decoder.
"""

import numpy as np


class BlockInterleaver:
    """
    Block interleaver that writes row-by-row and reads column-by-column.
    Burst error of length B affects at most ceil(B/depth) bits per row.
    """

    def __init__(self, depth: int = 20, width: int = 6):
        """
        Args:
            depth: Number of rows (frames). Burst of length depth
                   gets spread to 1 error per row.
            width: Number of columns (bits per frame).
        """
        self.depth = depth
        self.width = width
        self.block_size = depth * width

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        """Interleave bits by writing row-wise and reading column-wise."""
        # Pad to multiple of block_size
        pad_len = (self.block_size - len(bits) % self.block_size) % self.block_size
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

        output = []
        for block_start in range(0, len(bits), self.block_size):
            block = bits[block_start:block_start + self.block_size]
            matrix = block.reshape(self.depth, self.width)
            # Read column-by-column
            output.extend(matrix.T.flatten())

        return np.array(output, dtype=int)

    def deinterleave(self, bits: np.ndarray) -> np.ndarray:
        """Reverse the interleaving process."""
        pad_len = (self.block_size - len(bits) % self.block_size) % self.block_size
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

        output = []
        for block_start in range(0, len(bits), self.block_size):
            block = bits[block_start:block_start + self.block_size]
            matrix = block.reshape(self.width, self.depth)
            # Read row-by-row (reverse of column-wise read)
            output.extend(matrix.T.flatten())

        return np.array(output, dtype=int)

    def interleave_soft(self, bits: np.ndarray, confidence: np.ndarray):
        """Interleave both bits and their confidence scores."""
        i_bits = self.interleave(bits)
        i_conf = self.interleave_float(confidence)
        return i_bits, i_conf

    def deinterleave_soft(self, bits: np.ndarray, confidence: np.ndarray):
        """Deinterleave both bits and their confidence scores."""
        d_bits = self.deinterleave(bits)
        d_conf = self.deinterleave_float(confidence)
        return d_bits, d_conf

    def interleave_float(self, values: np.ndarray) -> np.ndarray:
        """Interleave float array (for confidence scores)."""
        pad_len = (self.block_size - len(values) % self.block_size) % self.block_size
        if pad_len > 0:
            values = np.concatenate([values, np.zeros(pad_len)])

        output = []
        for block_start in range(0, len(values), self.block_size):
            block = values[block_start:block_start + self.block_size]
            matrix = block.reshape(self.depth, self.width)
            output.extend(matrix.T.flatten())

        return np.array(output)

    def deinterleave_float(self, values: np.ndarray) -> np.ndarray:
        """Deinterleave float array."""
        pad_len = (self.block_size - len(values) % self.block_size) % self.block_size
        if pad_len > 0:
            values = np.concatenate([values, np.zeros(pad_len)])

        output = []
        for block_start in range(0, len(values), self.block_size):
            block = values[block_start:block_start + self.block_size]
            matrix = block.reshape(self.width, self.depth)
            output.extend(matrix.T.flatten())

        return np.array(output)

    def get_burst_protection(self) -> int:
        """Max burst length that results in at most 1 error per row."""
        return self.depth
