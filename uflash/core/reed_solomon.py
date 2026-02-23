"""
U-Flash Reed-Solomon Codec
RS(255, 223) over GF(2^8) for outer error correction.
Corrects up to 16 symbol (byte) errors per block.
"""

import numpy as np
from typing import Optional


class GaloisField:
    """GF(2^8) arithmetic with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1."""

    def __init__(self):
        self.size = 256
        self.prim_poly = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
        self.exp_table = np.zeros(512, dtype=int)
        self.log_table = np.zeros(256, dtype=int)
        self._build_tables()

    def _build_tables(self):
        """Build exponentiation and logarithm lookup tables."""
        x = 1
        for i in range(255):
            self.exp_table[i] = x
            self.log_table[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.prim_poly
        # Extend exp table for easier multiplication
        for i in range(255, 512):
            self.exp_table[i] = self.exp_table[i - 255]

    def multiply(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp_table[self.log_table[a] + self.log_table[b]]

    def divide(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(256)")
        if a == 0:
            return 0
        return self.exp_table[(self.log_table[a] - self.log_table[b]) % 255]

    def power(self, a: int, n: int) -> int:
        if a == 0:
            return 0
        return self.exp_table[(self.log_table[a] * n) % 255]

    def inverse(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Inverse of zero")
        return self.exp_table[255 - self.log_table[a]]

    def poly_multiply(self, p1: list, p2: list) -> list:
        """Multiply two polynomials over GF(256)."""
        result = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] ^= self.multiply(c1, c2)
        return result

    def poly_eval(self, poly: list, x: int) -> int:
        """Evaluate polynomial at x using Horner's method."""
        result = 0
        for coeff in poly:
            result = self.multiply(result, x) ^ coeff
        return result


class ReedSolomonCodec:
    """
    RS(255, 223) codec over GF(256).
    - 223 data bytes + 32 parity bytes = 255 total
    - Corrects up to 16 byte errors per block
    """

    def __init__(self, n: int = 255, k: int = 223):
        self.n = n           # Block length
        self.k = k           # Data length
        self.nsym = n - k    # Number of parity symbols (32)
        self.t = self.nsym // 2  # Error correction capability (16)
        self.gf = GaloisField()
        self.generator = self._build_generator()

    def _build_generator(self) -> list:
        """Build the generator polynomial."""
        g = [1]
        for i in range(self.nsym):
            g = self.gf.poly_multiply(g, [1, self.gf.exp_table[i]])
        return g

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data bytes with Reed-Solomon parity.
        Input: array of data bytes (0-255), length <= k
        Output: array of n bytes (data + parity)
        """
        # Pad data to k bytes
        if len(data) < self.k:
            data = np.concatenate([data, np.zeros(self.k - len(data), dtype=int)])

        data = data[:self.k].astype(int).tolist()

        # Polynomial division to compute parity
        msg_out = data + [0] * self.nsym

        for i in range(self.k):
            coeff = msg_out[i]
            if coeff != 0:
                for j in range(1, len(self.generator)):
                    msg_out[i + j] ^= self.gf.multiply(self.generator[j], coeff)

        # Parity bytes are the remainder
        parity = msg_out[self.k:]
        result = data + parity

        return np.array(result, dtype=int)

    def decode(self, received: np.ndarray) -> Optional[np.ndarray]:
        """
        Decode Reed-Solomon encoded block.
        Returns corrected data bytes, or None if uncorrectable.
        """
        received = received.astype(int).tolist()

        # Calculate syndromes
        syndromes = [0] * self.nsym
        has_errors = False
        for i in range(self.nsym):
            syndromes[i] = self.gf.poly_eval(received, self.gf.exp_table[i])
            if syndromes[i] != 0:
                has_errors = True

        if not has_errors:
            return np.array(received[:self.k], dtype=int)

        # Berlekamp-Massey algorithm to find error locator polynomial
        err_loc = self._berlekamp_massey(syndromes)
        if err_loc is None:
            return None

        # Chien search to find error positions
        err_pos = self._chien_search(err_loc)
        if err_pos is None or len(err_pos) != len(err_loc) - 1:
            return None

        # Forney algorithm to find error magnitudes
        corrected = list(received)
        err_mag = self._forney(syndromes, err_loc, err_pos)
        if err_mag is None:
            return None

        for pos, mag in zip(err_pos, err_mag):
            corrected[pos] ^= mag

        # Verify correction
        for i in range(self.nsym):
            if self.gf.poly_eval(corrected, self.gf.exp_table[i]) != 0:
                return None

        return np.array(corrected[:self.k], dtype=int)

    def _berlekamp_massey(self, syndromes: list) -> Optional[list]:
        """Berlekamp-Massey algorithm for error locator polynomial."""
        n = len(syndromes)
        C = [1] + [0] * n
        B = [1] + [0] * n
        L = 0
        m = 1
        b = 1

        for n_step in range(n):
            # Calculate discrepancy
            d = syndromes[n_step]
            for i in range(1, L + 1):
                d ^= self.gf.multiply(C[i], syndromes[n_step - i])

            if d == 0:
                m += 1
            elif 2 * L <= n_step:
                T = list(C)
                coeff = self.gf.divide(d, b)
                for i in range(m, n + 1):
                    if i - m < len(B) and B[i - m] != 0:
                        C[i] ^= self.gf.multiply(coeff, B[i - m])
                L = n_step + 1 - L
                B = T
                b = d
                m = 1
            else:
                coeff = self.gf.divide(d, b)
                for i in range(m, n + 1):
                    if i - m < len(B) and B[i - m] != 0:
                        C[i] ^= self.gf.multiply(coeff, B[i - m])
                m += 1

        # Trim polynomial
        err_loc = C[:L + 1]
        if L > self.t:
            return None
        return err_loc

    def _chien_search(self, err_loc: list) -> Optional[list]:
        """Find error positions using Chien search."""
        positions = []
        for i in range(self.n):
            val = self.gf.poly_eval(err_loc, self.gf.exp_table[i])
            if val == 0:
                positions.append(self.n - 1 - i)

        if len(positions) != len(err_loc) - 1:
            return None
        return positions

    def _forney(self, syndromes: list, err_loc: list, err_pos: list) -> Optional[list]:
        """Compute error magnitudes by solving the syndrome linear system directly.
        
        The syndromes satisfy: S_i = sum_j(e_j * alpha^{pos_j * i})
        Given the positions, this is a linear system in the error magnitudes e_j.
        We solve it using Gaussian elimination over GF(256).
        """
        num_errors = len(err_pos)
        
        # Build the matrix A where A[i][j] = alpha^{pos_j * i}
        # We need num_errors equations (rows) and num_errors unknowns (columns)
        A = [[0] * num_errors for _ in range(num_errors)]
        S = [syndromes[i] for i in range(num_errors)]
        
        for i in range(num_errors):
            for j in range(num_errors):
                # A[i][j] = alpha^((n-1-pos_j) * i) because poly_eval uses 
                # descending order: position p = coefficient of x^{n-1-p}
                # So S_i = R(alpha^i) = sum_j(e_j * alpha^{(n-1-pos_j)*i})
                exponent = ((self.n - 1 - err_pos[j]) * i) % 255 if i > 0 else 0
                A[i][j] = self.gf.exp_table[exponent] if i > 0 else 1
        
        # Gaussian elimination over GF(256)
        for col in range(num_errors):
            # Find pivot
            pivot_row = None
            for row in range(col, num_errors):
                if A[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                return None
            
            # Swap rows
            if pivot_row != col:
                A[col], A[pivot_row] = A[pivot_row], A[col]
                S[col], S[pivot_row] = S[pivot_row], S[col]
            
            # Scale pivot row
            inv_pivot = self.gf.inverse(A[col][col])
            for k in range(num_errors):
                A[col][k] = self.gf.multiply(A[col][k], inv_pivot)
            S[col] = self.gf.multiply(S[col], inv_pivot)
            
            # Eliminate column in other rows
            for row in range(num_errors):
                if row != col and A[row][col] != 0:
                    factor = A[row][col]
                    for k in range(num_errors):
                        A[row][k] ^= self.gf.multiply(factor, A[col][k])
                    S[row] ^= self.gf.multiply(factor, S[col])
        
        return S  # S now contains the error magnitudes

    def bits_to_bytes(self, bits: np.ndarray) -> np.ndarray:
        """Convert bit array to byte array."""
        pad_len = (8 - len(bits) % 8) % 8
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
        n_bytes = len(bits) // 8
        bytes_arr = np.zeros(n_bytes, dtype=int)
        for i in range(n_bytes):
            for j in range(8):
                bytes_arr[i] |= int(bits[i * 8 + j]) << (7 - j)
        return bytes_arr

    def bytes_to_bits(self, bytes_arr: np.ndarray) -> np.ndarray:
        """Convert byte array to bit array."""
        bits = []
        for byte in bytes_arr:
            for j in range(7, -1, -1):
                bits.append((int(byte) >> j) & 1)
        return np.array(bits, dtype=int)
