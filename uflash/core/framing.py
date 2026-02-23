"""
U-Flash Packet Framing Module
Implements packet structure with preamble, headers, CRC-16 error detection.
"""

import numpy as np
import struct
from typing import Optional, Tuple


class CRC16:
    """CRC-16-CCITT implementation."""

    POLYNOMIAL = 0x1021
    INITIAL = 0xFFFF

    @staticmethod
    def compute(data: bytes) -> int:
        """Compute CRC-16-CCITT checksum."""
        crc = CRC16.INITIAL
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ CRC16.POLYNOMIAL
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    @staticmethod
    def verify(data: bytes, expected_crc: int) -> bool:
        """Verify CRC checksum."""
        return CRC16.compute(data) == expected_crc


class Frame:
    """
    Frame structure:
    [Preamble 16b][Type 4b][SeqNum 12b][Length 8b][Payload 0-255B][CRC-16 16b]
    """

    # Frame types
    DATA = 0x0
    ACK = 0x1
    NACK = 0x2
    BEACON = 0x3
    COMMAND = 0x4

    # Preamble pattern for synchronization
    PREAMBLE = 0b1010101011001100  # 16-bit alternating pattern

    def __init__(self, frame_type: int, seq_num: int, payload: bytes,
                 device_id: int = 0):
        self.frame_type = frame_type & 0xF
        self.seq_num = seq_num & 0xFFF
        self.payload = payload
        self.device_id = device_id & 0xFF
        self.crc = 0

    def to_bytes(self) -> bytes:
        """Serialize frame to bytes."""
        # Pack header
        header = struct.pack('>H', self.PREAMBLE)  # 2 bytes preamble
        # Type (4 bits) + SeqNum (12 bits) = 16 bits
        type_seq = (self.frame_type << 12) | self.seq_num
        header += struct.pack('>H', type_seq)
        header += struct.pack('B', self.device_id)
        header += struct.pack('B', len(self.payload))

        # Combine header + payload
        frame_data = header + self.payload

        # Compute and append CRC
        self.crc = CRC16.compute(frame_data)
        frame_data += struct.pack('>H', self.crc)

        return frame_data

    def to_bits(self) -> np.ndarray:
        """Convert frame to bit array for transmission."""
        frame_bytes = self.to_bytes()
        bits = []
        for byte in frame_bytes:
            for j in range(7, -1, -1):
                bits.append((byte >> j) & 1)
        return np.array(bits, dtype=int)

    @staticmethod
    def from_bytes(data: bytes) -> Optional['Frame']:
        """Deserialize frame from bytes."""
        if len(data) < 8:  # Minimum frame size
            return None

        # Check preamble
        preamble = struct.unpack('>H', data[0:2])[0]
        if preamble != Frame.PREAMBLE:
            return None

        # Parse header
        type_seq = struct.unpack('>H', data[2:4])[0]
        frame_type = (type_seq >> 12) & 0xF
        seq_num = type_seq & 0xFFF
        device_id = data[4]
        payload_len = data[5]

        if len(data) < 8 + payload_len:
            return None

        payload = data[6:6 + payload_len]

        # Verify CRC
        received_crc = struct.unpack('>H', data[6 + payload_len:8 + payload_len])[0]
        frame_data = data[:6 + payload_len]
        if not CRC16.verify(frame_data, received_crc):
            return None

        frame = Frame(frame_type, seq_num, payload, device_id)
        frame.crc = received_crc
        return frame

    @staticmethod
    def from_bits(bits: np.ndarray) -> Optional['Frame']:
        """Reconstruct frame from bit array."""
        n_bytes = len(bits) // 8
        data = bytearray()
        for i in range(n_bytes):
            byte = 0
            for j in range(8):
                byte |= int(bits[i * 8 + j]) << (7 - j)
            data.append(byte)
        return Frame.from_bytes(bytes(data))


class FrameManager:
    """Manages frame sequencing, ACK tracking, and retransmission."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.tx_seq = 0
        self.rx_expected_seq = 0
        self.pending_acks = {}  # seq_num -> Frame
        self.delivered = []

    def create_data_frame(self, payload: bytes) -> Frame:
        """Create a new data frame with auto-incrementing sequence number."""
        frame = Frame(Frame.DATA, self.tx_seq, payload, self.device_id)
        self.pending_acks[self.tx_seq] = frame
        self.tx_seq = (self.tx_seq + 1) % 4096
        return frame

    def create_ack(self, seq_num: int) -> Frame:
        """Create ACK frame for received data."""
        return Frame(Frame.ACK, seq_num, b'', self.device_id)

    def create_beacon(self) -> Frame:
        """Create beacon frame for device discovery."""
        payload = struct.pack('B', self.device_id)
        return Frame(Frame.BEACON, 0, payload, self.device_id)

    def process_received(self, frame: Frame) -> Optional[bytes]:
        """
        Process received frame. Returns payload if data frame accepted,
        None otherwise.
        """
        if frame.frame_type == Frame.DATA:
            if frame.seq_num == self.rx_expected_seq:
                self.rx_expected_seq = (self.rx_expected_seq + 1) % 4096
                self.delivered.append(frame.payload)
                return frame.payload
        elif frame.frame_type == Frame.ACK:
            if frame.seq_num in self.pending_acks:
                del self.pending_acks[frame.seq_num]
        return None

    def get_stats(self) -> dict:
        """Return frame manager statistics."""
        return {
            'tx_seq': self.tx_seq,
            'rx_expected_seq': self.rx_expected_seq,
            'pending_acks': len(self.pending_acks),
            'delivered': len(self.delivered),
        }
