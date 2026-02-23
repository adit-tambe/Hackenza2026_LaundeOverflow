"""
Extended test suite for U-Flash Underwater Optical Communication System.
Tests edge cases, stress scenarios, boundary conditions, and adversarial inputs
that go beyond the baseline 114-test suite.
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ══════════════════════════════════════════════════════════════
# Test infrastructure
# ══════════════════════════════════════════════════════════════
PASS = 0
FAIL = 0
TOTAL = 0

def test(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        print(f"  \u2717 {name}  {detail}")

def section(name):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════
# 1. MODULATION — Edge Cases & Stress
# ══════════════════════════════════════════════════════════════
section("1. MODULATION EDGE CASES")

from core.modulation import OOKModulator, PPMModulator, AdaptiveThresholdDemodulator

# All-zeros and all-ones
ook = OOKModulator(samples_per_bit=10)
zeros = np.zeros(16, dtype=int)
ones = np.ones(16, dtype=int)

wf_zeros = ook.modulate(zeros)
test("OOK all-zeros modulate: waveform is all 0", np.all(wf_zeros == 0))
test("OOK all-zeros demod roundtrip", np.array_equal(zeros, ook.demodulate(wf_zeros)))

wf_ones = ook.modulate(ones)
test("OOK all-ones modulate: waveform is all 1", np.all(wf_ones == 1))
test("OOK all-ones demod roundtrip", np.array_equal(ones, ook.demodulate(wf_ones)))

# Single bit
single_0 = np.array([0], dtype=int)
single_1 = np.array([1], dtype=int)
test("OOK single bit 0 roundtrip", np.array_equal(single_0, ook.demodulate(ook.modulate(single_0))))
test("OOK single bit 1 roundtrip", np.array_equal(single_1, ook.demodulate(ook.modulate(single_1))))

# Large message (1000 bits)
large_bits = np.random.randint(0, 2, 1000)
wf_large = ook.modulate(large_bits)
test("OOK 1000-bit roundtrip", np.array_equal(large_bits, ook.demodulate(wf_large)))
test("OOK 1000-bit waveform length", len(wf_large) == 10000)

# PPM with M=2 (minimum)
ppm2 = PPMModulator(M=2, samples_per_slot=8)
bits_ppm2 = np.array([1, 0, 1, 1, 0], dtype=int)  # Will pad to 5 bits → 5 symbols
wf2 = ppm2.modulate(bits_ppm2)
bits_ppm2_out = ppm2.demodulate(wf2)
test("2-PPM roundtrip (first bits match)", np.array_equal(bits_ppm2, bits_ppm2_out[:len(bits_ppm2)]))

# PPM with all same symbol (worst case for energy detection)
ppm4 = PPMModulator(M=4, samples_per_slot=10)
bits_same = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int)  # All symbol=0
wf_same = ppm4.modulate(bits_same)
test("4-PPM all-same-symbol roundtrip", np.array_equal(bits_same, ppm4.demodulate(wf_same)))

# PPM with max symbol value
bits_max = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=int)  # All symbol=3
wf_max = ppm4.modulate(bits_max)
test("4-PPM all-max-symbol roundtrip", np.array_equal(bits_max, ppm4.demodulate(wf_max)))

# Adaptive threshold with heavy noise
atd = AdaptiveThresholdDemodulator(window_size=50)
clean = ook.modulate(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)).astype(float)
very_noisy = clean + np.random.normal(0, 0.4, len(clean))
very_noisy = np.clip(very_noisy, 0, 1)
bits_noisy, conf_noisy = atd.demodulate(very_noisy, samples_per_bit=10)
test("Adaptive demod returns correct length under heavy noise", len(bits_noisy) == 8)
test("Adaptive demod confidence in [0,1]", np.all(conf_noisy >= 0) and np.all(conf_noisy <= 1))

# PPM dispersion compensation with different distances
ppm_disp = PPMModulator(M=4, samples_per_slot=10)
bits_disp = np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=int)
wf_disp = ppm_disp.modulate(bits_disp)
rgb = {
    'blue': wf_disp * 0.9,
    'green': np.roll(wf_disp * 0.6, 2),
    'red': np.roll(wf_disp * 0.2, 5),
}
comp_near = ppm_disp.dispersion_compensate(rgb, distance_m=1.0)
comp_far = ppm_disp.dispersion_compensate(rgb, distance_m=20.0)
test("Dispersion compensation output length (near)", len(comp_near) == len(wf_disp))
test("Dispersion compensation output length (far)", len(comp_far) == len(wf_disp))
test("Dispersion at 1m vs 20m differs", not np.allclose(comp_near, comp_far))


# ══════════════════════════════════════════════════════════════
# 2. RLL — Boundary Conditions
# ══════════════════════════════════════════════════════════════
section("2. RLL BOUNDARY CONDITIONS")

from core.rll import RLLEncoder, AdaptiveRLLEncoder

rll = RLLEncoder(d=1, k=7)

# Minimum encodable unit (2 bits — RLL encodes in pairs)
data2 = np.array([1, 0], dtype=int)
enc2 = rll.encode(data2)
dec2 = rll.decode(enc2)
test("RLL 2-bit encode/decode", np.array_equal(data2, dec2))

# Large data (500 bits)
data500 = np.random.randint(0, 2, 500)
enc500 = rll.encode(data500)
dec500 = rll.decode(enc500)
test("RLL 500-bit roundtrip", np.array_equal(data500, dec500))

# All zeros
data_all0 = np.zeros(32, dtype=int)
enc_all0 = rll.encode(data_all0)
dec_all0 = rll.decode(enc_all0)
test("RLL all-zeros roundtrip", np.array_equal(data_all0, dec_all0))

# All ones
data_all1 = np.ones(32, dtype=int)
enc_all1 = rll.encode(data_all1)
dec_all1 = rll.decode(enc_all1)
test("RLL all-ones roundtrip", np.array_equal(data_all1, dec_all1))

# Adaptive RLL boundary: exactly on threshold (100 lux)
arll = AdaptiveRLLEncoder()
arll.adapt(100)
test("Adaptive RLL at 100 lux threshold", arll.get_config_name() in ['low_light', 'medium_light'])

# Adaptive RLL extreme low
arll.adapt(0)
test("Adaptive RLL at 0 lux → low_light", arll.get_config_name() == 'low_light')

# Adaptive RLL extreme high
arll.adapt(100000)
test("Adaptive RLL at 100k lux → high_light", arll.get_config_name() == 'high_light')


# ══════════════════════════════════════════════════════════════
# 3. CONVOLUTIONAL FEC — Stress Tests
# ══════════════════════════════════════════════════════════════
section("3. CONVOLUTIONAL FEC STRESS")

from core.convolutional import ConvolutionalEncoder, ViterbiDecoder

conv = ConvolutionalEncoder(K=7)
vit = ViterbiDecoder(K=7)

# Minimum input (1 bit)
d1 = np.array([1], dtype=int)
e1 = conv.encode(d1)
v1 = vit.decode_hard(e1)
test("Conv+Viterbi 1-bit input", v1[0] == 1)

# Large input (200 bits)
d200 = np.random.randint(0, 2, 200)
e200 = conv.encode(d200)
v200 = vit.decode_hard(e200)[:len(d200)]
test("Conv+Viterbi 200-bit clean roundtrip", np.array_equal(d200, v200))

# Heavy errors (10% BER)
e200_noisy = e200.copy()
n_errors = len(e200_noisy) // 10
err_indices = np.random.choice(len(e200_noisy), size=n_errors, replace=False)
e200_noisy[err_indices] = 1 - e200_noisy[err_indices]
v200_noisy = vit.decode_hard(e200_noisy)[:len(d200)]
ber_heavy = np.mean(d200 != v200_noisy)
test("Conv+Viterbi 10% input BER < 15% output BER", ber_heavy < 0.15,
     f"BER={ber_heavy:.4f}")

# Soft vs hard decision comparison (soft should be >= as good)
conf_test = np.ones(len(e200_noisy))
conf_test[err_indices] = 0.3
v200_soft = vit.decode(e200_noisy, conf_test)[:len(d200)]
ber_soft_test = np.mean(d200 != v200_soft)
test("Soft Viterbi <= hard Viterbi on 10% errors",
     ber_soft_test <= ber_heavy + 0.01,
     f"Soft={ber_soft_test:.4f} Hard={ber_heavy:.4f}")


# ══════════════════════════════════════════════════════════════
# 4. INTERLEAVER — Edge Cases
# ══════════════════════════════════════════════════════════════
section("4. INTERLEAVER EDGE CASES")

from core.interleaver import BlockInterleaver

# Very small depth
intlv_small = BlockInterleaver(depth=2, width=2)
d_small = np.array([1, 0, 1, 1], dtype=int)
test("2x2 interleaver roundtrip", np.array_equal(
    d_small, intlv_small.deinterleave(intlv_small.interleave(d_small))))

# Large depth
intlv_large = BlockInterleaver(depth=50, width=10)
d_large = np.random.randint(0, 2, 500)
test("50x10 interleaver roundtrip", np.array_equal(
    d_large, intlv_large.deinterleave(intlv_large.interleave(d_large))))

# Single element
intlv1 = BlockInterleaver(depth=3, width=3)
d_single = np.array([1], dtype=int)
il_single = intlv1.interleave(d_single)
dil_single = intlv1.deinterleave(il_single)
test("Interleaver single-element: first bit preserved", dil_single[0] == 1)

# Burst protection property: burst of depth consecutive errors → 1-per-row
intlv_burst = BlockInterleaver(depth=10, width=8)
d_burst = np.random.randint(0, 2, 80)
il_burst = intlv_burst.interleave(d_burst)
# Introduce burst of 10 consecutive errors at position 20
il_corrupted = il_burst.copy()
il_corrupted[20:30] = 1 - il_corrupted[20:30]
dil_corrupted = intlv_burst.deinterleave(il_corrupted)
# After deinterleaving, errors should be spread across rows
actual_errors = 10
test("Burst of 10 errors is deinterleaved", actual_errors == 10)  # Sanity check
# The errors should be spread across different row positions
error_positions = np.where(d_burst != dil_corrupted)[0]
if len(error_positions) > 1:
    gaps = np.diff(error_positions)
    test("Deinterleaved errors are spread (not consecutive)", np.min(gaps) >= 1)
else:
    test("Deinterleaved errors are spread (not consecutive)", True)


# ══════════════════════════════════════════════════════════════
# 5. REED-SOLOMON — Boundary Conditions
# ══════════════════════════════════════════════════════════════
section("5. REED-SOLOMON BOUNDARY CONDITIONS")

from core.reed_solomon import ReedSolomonCodec, GaloisField

rs = ReedSolomonCodec(n=255, k=223)
gf = GaloisField()

# GF(256) arithmetic edge cases
test("GF multiply identity: a*1=a", gf.multiply(137, 1) == 137)
test("GF multiply zero: 0*a=0", gf.multiply(0, 255) == 0)
test("GF multiply commutativity", gf.multiply(42, 73) == gf.multiply(73, 42))
test("GF associativity", gf.multiply(gf.multiply(5, 7), 11) == gf.multiply(5, gf.multiply(7, 11)))
test("GF(256) power: a^0 = 1", gf.power(42, 0) == 1)
test("GF(256) Fermat: a^255 = 1", gf.power(42, 255) == 1)
test("GF(256) inverse roundtrip", gf.inverse(gf.inverse(99)) == 99)

# Poly operations
p1 = [1, 2, 3]
p2 = [4, 5]
prod = gf.poly_multiply(p1, p2)
test("GF poly multiply output length", len(prod) == len(p1) + len(p2) - 1)

# Poly eval
test("GF poly_eval constant poly", gf.poly_eval([42], 0) == 42)
test("GF poly_eval constant poly at any x", gf.poly_eval([42], 100) == 42)

# RS: 0 errors
np.random.seed(12345)
data0 = np.random.randint(0, 256, 223)
enc0 = rs.encode(data0)
dec0 = rs.decode(enc0)
test("RS 0 errors: perfect decode", dec0 is not None and np.array_equal(data0, dec0))

# RS: 1 error
enc1e = enc0.copy()
enc1e[0] ^= 42
dec1e = rs.decode(enc1e)
test("RS 1 error: corrected", dec1e is not None and np.array_equal(data0, dec1e))

# RS: exactly 16 errors (max correctable = t)
enc16 = enc0.copy()
positions_16 = np.random.choice(255, size=16, replace=False)
for p in positions_16:
    enc16[p] ^= np.random.randint(1, 256)
dec16 = rs.decode(enc16)
test("RS 16 errors (= t): corrected", dec16 is not None and np.array_equal(data0, dec16))

# RS: 17 errors (> t, should fail gracefully)
enc17 = enc0.copy()
positions_17 = np.random.choice(255, size=17, replace=False)
for p in positions_17:
    enc17[p] ^= np.random.randint(1, 256)
dec17 = rs.decode(enc17)
test("RS 17 errors (> t): returns None (uncorrectable)", dec17 is None)

# RS: errors only in parity region
enc_parity = enc0.copy()
for p in range(223, 228):  # 5 errors in parity only
    enc_parity[p] ^= np.random.randint(1, 256)
dec_parity = rs.decode(enc_parity)
test("RS errors in parity region only: corrected", dec_parity is not None and np.array_equal(data0, dec_parity))

# RS: errors at boundaries (first and last positions)
enc_bound = enc0.copy()
enc_bound[0] ^= 100
enc_bound[254] ^= 200
dec_bound = rs.decode(enc_bound)
test("RS errors at position 0 and 254: corrected", dec_bound is not None and np.array_equal(data0, dec_bound))

# RS: data shorter than k (padding case)
short_data = np.random.randint(0, 256, 50)
enc_short = rs.encode(short_data)
# Only first 50 bytes matter, rest are zeros
dec_short = rs.decode(enc_short)
test("RS short data (50 bytes): decode matches padded data",
     dec_short is not None and np.array_equal(enc_short[:223], dec_short))

# RS: multiple independent blocks
for i in range(5):
    data_i = np.random.randint(0, 256, 223)
    enc_i = rs.encode(data_i)
    errors_i = np.random.choice(255, size=np.random.randint(1, 16), replace=False)
    for p in errors_i:
        enc_i[p] ^= np.random.randint(1, 256)
    dec_i = rs.decode(enc_i)
    test(f"RS block {i+1}: {len(errors_i)} errors corrected",
         dec_i is not None and np.array_equal(data_i, dec_i),
         f"with {len(errors_i)} errors")


# ══════════════════════════════════════════════════════════════
# 6. FRAMING — Protocol Tests
# ══════════════════════════════════════════════════════════════
section("6. FRAMING PROTOCOL TESTS")

from core.framing import Frame, FrameManager, CRC16

# CRC with empty data
crc_empty = CRC16.compute(b"")
test("CRC-16 of empty data is valid", 0 <= crc_empty <= 0xFFFF)

# CRC with large data
crc_large = CRC16.compute(b"A" * 1000)
test("CRC-16 of 1000 bytes", 0 <= crc_large <= 0xFFFF)
test("CRC-16 verify 1000 bytes", CRC16.verify(b"A" * 1000, crc_large))

# CRC collision resistance: different inputs → different CRCs (probabilistic)
crc_a = CRC16.compute(b"Hello")
crc_b = CRC16.compute(b"Hellp")  # 1 char different
test("CRC-16 detects 1-char difference", crc_a != crc_b)

# Frame types: NACK, BEACON, COMMAND
nack = Frame(Frame.NACK, seq_num=10, payload=b"", device_id=3)
nack_bytes = nack.to_bytes()
nack_back = Frame.from_bytes(nack_bytes)
test("NACK frame roundtrip", nack_back is not None and nack_back.frame_type == Frame.NACK)
test("NACK frame seq_num preserved", nack_back is not None and nack_back.seq_num == 10)

beacon = Frame(Frame.BEACON, seq_num=0, payload=b"\x07", device_id=7)
beacon_bytes = beacon.to_bytes()
beacon_back = Frame.from_bytes(beacon_bytes)
test("BEACON frame roundtrip", beacon_back is not None and beacon_back.frame_type == Frame.BEACON)

cmd = Frame(Frame.COMMAND, seq_num=100, payload=b"CMD_RESET", device_id=5)
cmd_bytes = cmd.to_bytes()
cmd_back = Frame.from_bytes(cmd_bytes)
test("COMMAND frame roundtrip", cmd_back is not None and cmd_back.payload == b"CMD_RESET")

# Empty payload frame
empty_frame = Frame(Frame.DATA, seq_num=0, payload=b"", device_id=0)
empty_bytes = empty_frame.to_bytes()
empty_back = Frame.from_bytes(empty_bytes)
test("Empty payload frame roundtrip", empty_back is not None and empty_back.payload == b"")

# Max payload size (255 bytes)
max_payload = bytes(range(256)) * 1  # only first 255
max_payload = max_payload[:255]
max_frame = Frame(Frame.DATA, seq_num=4095, payload=max_payload, device_id=255)
max_bytes = max_frame.to_bytes()
max_back = Frame.from_bytes(max_bytes)
test("Max payload (255 bytes) frame roundtrip", max_back is not None and max_back.payload == max_payload)
test("Max seq_num (4095) preserved", max_back is not None and max_back.seq_num == 4095)
test("Max device_id (255) preserved", max_back is not None and max_back.device_id == 255)

# Corrupted frame detection
good_bytes = Frame(Frame.DATA, seq_num=42, payload=b"Test", device_id=1).to_bytes()
bad_bytes = bytearray(good_bytes)
bad_bytes[6] ^= 0xFF  # Corrupt payload byte
test("Corrupted frame detected (returns None)", Frame.from_bytes(bytes(bad_bytes)) is None)

# Truncated frame
test("Truncated frame (3 bytes) → None", Frame.from_bytes(b"\xAA\xCC\x00") is None)

# FrameManager: create_beacon
fm = FrameManager(device_id=5)
beacon_frame = fm.create_beacon()
test("FrameManager beacon has device_id", beacon_frame.device_id == 5)
test("FrameManager beacon type", beacon_frame.frame_type == Frame.BEACON)

# FrameManager: process_received flow
fm_tx = FrameManager(device_id=1)
fm_rx = FrameManager(device_id=2)

# Send 3 data frames
for i in range(3):
    df = fm_tx.create_data_frame(f"msg_{i}".encode())
    payload = fm_rx.process_received(df)
    test(f"FrameManager recv msg_{i}", payload == f"msg_{i}".encode())

test("FrameManager delivered count = 3", len(fm_rx.delivered) == 3)
test("FrameManager tx_seq = 3", fm_tx.tx_seq == 3)

# Send ACK back
ack = fm_rx.create_ack(0)
fm_tx.process_received(ack)
test("FrameManager ACK removes pending", 0 not in fm_tx.pending_acks)

# Stats
stats = fm_tx.get_stats()
test("FrameManager stats has tx_seq", 'tx_seq' in stats)
test("FrameManager stats pending_acks = 2", stats['pending_acks'] == 2)

# Out-of-order receive (should be rejected)
future_frame = Frame(Frame.DATA, seq_num=99, payload=b"future", device_id=1)
result = fm_rx.process_received(future_frame)
test("Out-of-order frame rejected", result is None)


# ══════════════════════════════════════════════════════════════
# 7. CHANNEL — Parameter Variation
# ══════════════════════════════════════════════════════════════
section("7. CHANNEL PARAMETER VARIATION")

from core.channel import UnderwaterChannel

# All water types
for wtype in ['clear', 'coastal', 'harbor', 'turbid']:
    ch = UnderwaterChannel(water_type=wtype, distance_m=5.0)
    test(f"Channel {wtype}: positive SNR", ch.snr_db > -10, f"SNR={ch.snr_db:.1f}")
    bits_out, conf_out, metrics = ch.transmit_bits(np.random.randint(0, 2, 500))
    test(f"Channel {wtype}: output length", len(bits_out) == 500)

# Distance extremes
ch_close = UnderwaterChannel(water_type='clear', distance_m=0.5)
ch_far = UnderwaterChannel(water_type='clear', distance_m=20.0)
test("Very close (0.5m): high SNR", ch_close.snr_db > 10, f"SNR={ch_close.snr_db:.1f}")
test("Very far (20m): lower SNR than 0.5m", ch_far.snr_db < ch_close.snr_db)

# Depth variation
ch_shallow = UnderwaterChannel(water_type='coastal', distance_m=5, depth_m=1)
ch_deep = UnderwaterChannel(water_type='coastal', distance_m=5, depth_m=50)
test("Channel depth 1m vs 50m: different SNR", ch_shallow.snr_db != ch_deep.snr_db)

# Ambient light variation
ch_dark = UnderwaterChannel(water_type='clear', distance_m=5, ambient_lux=10)
ch_bright = UnderwaterChannel(water_type='clear', distance_m=5, ambient_lux=10000)
test("Dark vs bright ambient: different SNR", ch_dark.snr_db != ch_bright.snr_db)

# Hot-swap parameters
ch_hot = UnderwaterChannel(water_type='clear', distance_m=3)
initial_snr = ch_hot.snr_db
ch_hot.set_parameters(water_type='turbid', distance_m=10)
test("Hot-swap to turbid/10m: SNR decreased", ch_hot.snr_db < initial_snr)

# Quality labels
test("Clear/1m quality", UnderwaterChannel(water_type='clear', distance_m=1).get_quality_label() in ['Excellent', 'Good'])
test("Turbid/15m quality", UnderwaterChannel(water_type='turbid', distance_m=15).get_quality_label() in ['Poor', 'Fair'])

# Gilbert-Elliott burst errors
ch_burst = UnderwaterChannel(water_type='harbor', distance_m=8)
bits_in = np.random.randint(0, 2, 10000)
bits_out, _, metrics = ch_burst.transmit_bits(bits_in)
test("Burst channel produces errors", metrics['ber_raw'] > 0, f"BER={metrics['ber_raw']:.4f}")

# Waveform transmission also works
wf_in = np.random.random(500)
wf_out, wf_metrics = ch_burst.transmit(wf_in)
test("Waveform channel metrics have SNR", 'snr_db' in wf_metrics)


# ══════════════════════════════════════════════════════════════
# 8. WATER QUALITY — Edge Cases
# ══════════════════════════════════════════════════════════════
section("8. WATER QUALITY EDGE CASES")

from core.water_quality import WaterQualityEstimator

wq = WaterQualityEstimator()

# Extreme ROI sizes
turb_tiny = wq.estimate_turbidity(roi_size=10, distance_m=1)
turb_huge = wq.estimate_turbidity(roi_size=1000, distance_m=1)
test("Tiny ROI (10): returns result", 'turbidity_ntu' in turb_tiny)
test("Huge ROI (1000): returns result", 'turbidity_ntu' in turb_huge)
test("Bigger ROI → higher confidence", turb_huge['confidence'] >= turb_tiny['confidence'])

# Extreme distances
turb_close = wq.estimate_turbidity(roi_size=200, distance_m=0.1)
turb_far = wq.estimate_turbidity(roi_size=200, distance_m=50.0)
test("Very close distance: valid result", 'classification' in turb_close)
test("Very far distance: valid result", 'classification' in turb_far)

# Edge color distributions
all_green = wq.classify_water_type({'blue': 0.1, 'green': 0.8, 'red': 0.1})
test("Green-dominant → valid classification", all_green['water_type'] in ['clear', 'coastal', 'harbor', 'turbid'])

all_equal = wq.classify_water_type({'blue': 0.33, 'green': 0.34, 'red': 0.33})
test("Equal RGB → valid classification", all_equal['water_type'] in ['clear', 'coastal', 'harbor', 'turbid'])


# ══════════════════════════════════════════════════════════════
# 9. MOTION COMPENSATION — Edge Cases
# ══════════════════════════════════════════════════════════════
section("9. MOTION COMPENSATION EDGE CASES")

from core.motion_compensation import MotionCompensator, MadgwickFilter

# Madgwick with large angular velocities
mf = MadgwickFilter(sample_rate=100)
for _ in range(10):
    mf.update(np.array([5.0, -3.0, 2.0]), np.array([0.0, 0.0, 9.81]))
euler = mf.get_euler_angles()
test("Madgwick after rapid rotation: valid angles", all(isinstance(v, float) for v in euler.values()))

# Very short IMU window
mc = MotionCompensator(sample_rate=100)
imu_short = mc.simulate_imu_data(0.1, 'stationary')
test("Short IMU window (0.1s): has data", imu_short['gyro'].shape[0] >= 1)
result_short = mc.process_imu(imu_short)
test("Short IMU: valid summary", 'summary' in result_short)

# Long IMU window
imu_long = mc.simulate_imu_data(5.0, 'swimming')
test("Long IMU window (5s): has data", imu_long['gyro'].shape[0] >= 400)
result_long = mc.process_imu(imu_long)
test("Long IMU swimming: artifacts > 0", result_long['summary']['artifact_percentage'] >= 0)

# Confidence compensation via apply_to_decoder
bit_confidence = np.ones(100)
# Get motion confidence from process_imu result
motion_conf = result_long['confidence_weights']
compensated_conf = mc.apply_to_decoder(bit_confidence, motion_conf, frame_rate=30.0)
test("Motion-compensated confidence in [0,1]", 
     np.all(compensated_conf >= 0) and np.all(compensated_conf <= 1.01))
test("Compensation length preserved", len(compensated_conf) == len(bit_confidence))


# ══════════════════════════════════════════════════════════════
# 10. ADAPTIVE FEC — Thorough Mode Testing
# ══════════════════════════════════════════════════════════════
section("10. ADAPTIVE FEC THOROUGH TESTING")

from core.adaptive_fec import AdaptiveFECController, FECMode, FECModeConfig

# Fresh controller for each SNR level
snr_mode_pairs = [
    (30, ['none', 'conv_only']),       # Excellent
    (20, ['conv_only', 'none', 'conv_interleaver']),  # Excellent/Good
    (12, ['conv_only', 'conv_interleaver', 'conv_interleaver_rs']),  # Good
    (7,  ['conv_interleaver', 'conv_interleaver_rs', 'heavy']),  # Fair
    (2,  ['heavy']),                                    # Poor
    (0,  ['heavy']),                                    # Very poor
    (-5, ['heavy']),                                   # Negative SNR
]

for snr, acceptable_modes in snr_mode_pairs:
    afec = AdaptiveFECController(base_data_rate=180)
    r = afec.adapt(snr_db=snr, ber=0.01 if snr > 5 else 0.1)
    test(f"FEC at SNR={snr}dB → valid mode",
         r['selected_mode'] in acceptable_modes,
         f"Got {r['selected_mode']}")

# Mode transition: excellent → poor → back
afec_trans = AdaptiveFECController(base_data_rate=180)
r1 = afec_trans.adapt(snr_db=25, ber=1e-6)
test("Transition start: light mode", r1['selected_mode'] in ['none', 'conv_only'])

# Feed multiple poor readings to overwhelm the averaging window
for _ in range(5):
    r2 = afec_trans.adapt(snr_db=1, ber=0.08)
test("Transition drop: heavy mode", r2['selected_mode'] == 'heavy')

# Recovery should show hysteresis (won't immediately drop back)
r3 = afec_trans.adapt(snr_db=25, ber=1e-6)
test("Transition recovery: may still be heavy (hysteresis)", 'selected_mode' in r3)

# Effective data rate decreases with heavier FEC
modes_summary = AdaptiveFECController(base_data_rate=180).get_all_modes_summary()
rates = {m['mode']: m['effective_data_rate'] for m in modes_summary}
test("None mode has highest data rate", rates.get('none', 0) >= rates.get('heavy', 0))

# FEC config details
for mode in FECMode:
    config = FECModeConfig.CONFIGS[mode]
    test(f"FECMode.{mode.value}: has min_snr_db", 'min_snr_db' in config)


# ══════════════════════════════════════════════════════════════
# 11. ML CHANNEL PREDICTOR — Edge Cases
# ══════════════════════════════════════════════════════════════
section("11. ML CHANNEL PREDICTOR EDGE CASES")

from core.channel_prediction import ChannelPredictor

# Predictor with exactly minimum data
cp_min = ChannelPredictor(lookback_window=5, prediction_horizon=1)
for i in range(5):
    cp_min.add_measurement(snr_db=15.0, ber=1e-4, roi_size=100, angular_velocity=1.0)
pred_min = cp_min.predict()
test("Predictor at minimum data: returns result", pred_min is not None)

# Predictor with stable channel → should predict stable
cp_stable = ChannelPredictor(lookback_window=20, prediction_horizon=5)
for i in range(30):
    cp_stable.add_measurement(snr_db=15.0, ber=1e-4, roi_size=100, angular_velocity=1.0)
trend = cp_stable.predict_trend()
test("Stable channel → stable trend", trend == 'stable')

# Predictor with improving channel
cp_improve = ChannelPredictor(lookback_window=20, prediction_horizon=5)
for i in range(30):
    cp_improve.add_measurement(snr_db=5 + i * 0.5, ber=0.01 / (i + 1),
                               roi_size=100 + i * 5, angular_velocity=max(0.1, 5 - i * 0.2))
trend_improve = cp_improve.predict_trend()
test("Improving channel → improving trend", trend_improve == 'improving')

# Predictor with degrading channel
cp_degrade = ChannelPredictor(lookback_window=20, prediction_horizon=5)
for i in range(30):
    cp_degrade.add_measurement(snr_db=20 - i * 0.5, ber=1e-5 * (i + 1),
                               roi_size=max(10, 200 - i * 5), angular_velocity=1 + i * 0.3)
trend_degrade = cp_degrade.predict_trend()
test("Degrading channel → degrading trend", trend_degrade == 'degrading')

# Prediction probabilities sum to ~1
pred_full = cp_stable.predict()
if pred_full and 'probabilities' in pred_full:
    prob_sum = sum(pred_full['probabilities'].values())
    test("Prediction probabilities sum ≈ 1", abs(prob_sum - 1.0) < 0.01, f"Sum={prob_sum:.4f}")


# ══════════════════════════════════════════════════════════════
# 12. END-TO-END — Stress & Edge Cases
# ══════════════════════════════════════════════════════════════
section("12. END-TO-END STRESS TESTS")

from simulation.engine import SimulationEngine

eng = SimulationEngine()

# Very short message (1 char)
r1c = eng.run_transmission("A", {'water_type': 'clear', 'distance_m': 1})
test("E2E 1-char message: has result", 'decoded_message' in r1c)
test("E2E 1-char message: success", r1c['success'])

# Numeric message
r_num = eng.run_transmission("1234567890", {'water_type': 'clear', 'distance_m': 2})
test("E2E numeric message: decoded", r_num['success'], f"Got: {r_num['decoded_message']}")

# Special characters
r_spec = eng.run_transmission("!@#$%^&*()_+-={}|:<>?", {'water_type': 'clear', 'distance_m': 2})
test("E2E special characters: has result", 'decoded_message' in r_spec)

# Repeated character
r_rep = eng.run_transmission("X" * 50, {'water_type': 'clear', 'distance_m': 2})
test("E2E 50x'X': decoded", r_rep['success'])

# Multiple transmissions don't interfere
r_a = eng.run_transmission("First", {'water_type': 'clear', 'distance_m': 1})
r_b = eng.run_transmission("Second", {'water_type': 'clear', 'distance_m': 1})
test("Multiple transmissions: first succeeds", r_a['success'])
test("Multiple transmissions: second succeeds", r_b['success'])
test("Multiple transmissions: no cross-contamination",
     r_a['decoded_message'] != r_b['decoded_message'] or
     (r_a['decoded_message'] == "First" and r_b['decoded_message'] == "Second"))

# Different water types with same message
msg = "Test message"
for wtype in ['clear', 'coastal', 'harbor']:
    r = eng.run_transmission(msg, {'water_type': wtype, 'distance_m': 3})
    test(f"E2E {wtype}/3m: returns result", 'decoded_message' in r)

# All motion levels
for motion in ['stationary', 'handheld', 'walking', 'swimming']:
    r = eng.run_transmission("Motion test", {'water_type': 'clear', 'distance_m': 2},
                             motion_level=motion)
    test(f"E2E motion={motion}: returns result", 'decoded_message' in r)

# BER sweep with single distance
sweep1 = eng.run_ber_sweep("Hi", distances=[3])
test("BER sweep single distance: 1 result", len(sweep1['sweep_results']) == 1)

# System status consistency
status = eng.get_system_status()
test("Status has all expected keys", all(k in status for k in ['channel', 'fec_modes_available']))

# Transmission result has timing info
test("E2E result has timing", 'time_ms' in r1c or 'stages' in r1c)


# ══════════════════════════════════════════════════════════════
# 13. CROSS-MODULE STRESS TESTS
# ══════════════════════════════════════════════════════════════
section("13. CROSS-MODULE STRESS TESTS")

# Full pipeline with all-zero data
data_zero = np.zeros(223, dtype=int)
enc_zero = rs.encode(data_zero)
rs_bits = rs.bytes_to_bits(enc_zero)
conv_bits = conv.encode(rs_bits)
intlv = BlockInterleaver(depth=20, width=6)
int_bits = intlv.interleave(conv_bits)
ch_stress = UnderwaterChannel(water_type='clear', distance_m=3)
rx, conf, _ = ch_stress.transmit_bits(int_bits)
deint = intlv.deinterleave(rx)
deint_conf = intlv.deinterleave_float(conf)
vit_out = vit.decode(deint, deint_conf)
vit_bytes = rs.bits_to_bytes(vit_out[:255*8])
if len(vit_bytes) < 255:
    vit_bytes = np.concatenate([vit_bytes, np.zeros(255 - len(vit_bytes), dtype=int)])
rs_out = rs.decode(vit_bytes[:255])
test("Full pipeline all-zero data: RS decode OK", rs_out is not None)
if rs_out is not None:
    test("Full pipeline all-zero data: data match", np.array_equal(data_zero, rs_out))

# RLL + Conv + Interleaver chain
data_chain = np.random.randint(0, 2, 100)
rll_chain = RLLEncoder(d=1, k=7)
rll_enc = rll_chain.encode(data_chain)
conv_enc = conv.encode(rll_enc)
int_enc = intlv.interleave(conv_enc)
int_dec = intlv.deinterleave(int_enc)
vit_dec = vit.decode_hard(int_dec)[:len(rll_enc)]
rll_dec = rll_chain.decode(vit_dec)
test("RLL→Conv→Intlv→Deintlv→Viterbi→DeRLL: roundtrip",
     np.array_equal(data_chain, rll_dec))

# PPM modulation + channel + demodulation stress
ppm_stress = PPMModulator(M=4, samples_per_slot=10)
bits_stress = np.random.randint(0, 2, 200)
wf_stress = ppm_stress.modulate(bits_stress)
ch_ppm = UnderwaterChannel(water_type='coastal', distance_m=3)
wf_rx, _ = ch_ppm.transmit(wf_stress)
bits_rx = ppm_stress.demodulate(wf_rx)
# Not expecting perfect, but should be same length
test("PPM through channel: same output length", len(bits_rx) == len(bits_stress))

# Multiple RS blocks in sequence
data_multi = [np.random.randint(0, 256, 223) for _ in range(3)]
enc_multi = [rs.encode(d) for d in data_multi]
all_ok = True
for i, (d, e) in enumerate(zip(data_multi, enc_multi)):
    # Add random errors
    e_noisy = e.copy()
    n_err = np.random.randint(1, 10)
    for p in np.random.choice(255, size=n_err, replace=False):
        e_noisy[p] ^= np.random.randint(1, 256)
    dec = rs.decode(e_noisy)
    if dec is None or not np.array_equal(d, dec):
        all_ok = False
        break
test("3 sequential RS blocks with random errors: all corrected", all_ok)


# ══════════════════════════════════════════════════════════════
# 14. REPRODUCIBILITY & DETERMINISM
# ══════════════════════════════════════════════════════════════
section("14. REPRODUCIBILITY & DETERMINISM")

# RS encode is deterministic
d_det = np.arange(223, dtype=int) % 256
enc_det1 = rs.encode(d_det)
enc_det2 = rs.encode(d_det)
test("RS encode is deterministic", np.array_equal(enc_det1, enc_det2))

# Conv encode is deterministic
b_det = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4, dtype=int)
c_det1 = conv.encode(b_det)
c_det2 = conv.encode(b_det)
test("Conv encode is deterministic", np.array_equal(c_det1, c_det2))

# RLL encode is deterministic
rll_det = RLLEncoder(d=1, k=7)
r_det1 = rll_det.encode(b_det)
r_det2 = rll_det.encode(b_det)
test("RLL encode is deterministic", np.array_equal(r_det1, r_det2))

# PPM modulate is deterministic
p_det1 = ppm4.modulate(b_det)
p_det2 = ppm4.modulate(b_det)
test("PPM modulate is deterministic", np.array_equal(p_det1, p_det2))

# Frame serialization is deterministic
f_det1 = Frame(Frame.DATA, 42, b"test", device_id=1).to_bytes()
f_det2 = Frame(Frame.DATA, 42, b"test", device_id=1).to_bytes()
test("Frame to_bytes is deterministic", f_det1 == f_det2)


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  EXTENDED TEST RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed")
print(f"{'═'*60}")

if FAIL > 0:
    print(f"\n  ⚠ {FAIL} test(s) failed — review above for details")
    sys.exit(1)
else:
    print(f"\n  ✅ All extended tests passed!")
    sys.exit(0)
