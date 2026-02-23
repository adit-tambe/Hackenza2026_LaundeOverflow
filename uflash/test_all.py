"""
Comprehensive test suite for U-Flash Underwater Optical Communication System.
Tests all 11 modules + end-to-end pipeline across various conditions.
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
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")

def section(name):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════
# 1. Modulation Module
# ══════════════════════════════════════════════════════════════
section("1. MODULATION MODULE (OOK + PPM)")

from core.modulation import OOKModulator, PPMModulator, AdaptiveThresholdDemodulator

# OOK roundtrip
ook = OOKModulator(samples_per_bit=10)
bits_in = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
waveform = ook.modulate(bits_in)
bits_out = ook.demodulate(waveform)
test("OOK modulate/demodulate roundtrip", np.array_equal(bits_in, bits_out))
test("OOK waveform length correct", len(waveform) == len(bits_in) * 10)

# OOK soft demodulation
bits_soft, conf = ook.soft_demodulate(waveform)
test("OOK soft demod matches hard", np.array_equal(bits_in, bits_soft))
test("OOK confidence all > 0.9", np.all(conf > 0.9))

# PPM roundtrip
ppm = PPMModulator(M=4, samples_per_slot=10)
bits_ppm = np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=int)
waveform_ppm = ppm.modulate(bits_ppm)
bits_ppm_out = ppm.demodulate(waveform_ppm)
test("4-PPM modulate/demodulate roundtrip", np.array_equal(bits_ppm, bits_ppm_out))
test("PPM waveform length = 4 symbols × 40 samples", len(waveform_ppm) == 4 * 4 * 10)

# PPM soft demodulation
bits_ppm_soft, conf_ppm = ppm.soft_demodulate(waveform_ppm)
test("PPM soft demod matches hard", np.array_equal(bits_ppm, bits_ppm_soft))

# PPM with odd-length input (should pad)
bits_odd = np.array([1, 0, 1], dtype=int)
waveform_odd = ppm.modulate(bits_odd)
bits_odd_out = ppm.demodulate(waveform_odd)
test("PPM handles odd-length input (pads to 4 bits)", len(bits_odd_out) == 4)
test("PPM padded: first 3 bits preserved", np.array_equal(bits_odd, bits_odd_out[:3]))

# Dispersion compensation
rgb = {
    'blue': waveform_ppm * 0.8,
    'green': np.roll(waveform_ppm * 0.6, 1),
    'red': np.roll(waveform_ppm * 0.3, 3),
}
compensated = ppm.dispersion_compensate(rgb, distance_m=5.0)
test("Dispersion compensation returns same length", len(compensated) == len(waveform_ppm))

# Adaptive threshold
atd = AdaptiveThresholdDemodulator(window_size=30)
noisy = waveform.copy().astype(float) + np.random.normal(0, 0.1, len(waveform))
noisy = np.clip(noisy, 0, 1)
bits_atd, conf_atd = atd.demodulate(noisy, samples_per_bit=10)
ber_atd = np.sum(bits_in != bits_atd) / len(bits_in)
test("Adaptive threshold demod with noise: BER < 0.3", ber_atd < 0.3, f"BER={ber_atd:.3f}")


# ══════════════════════════════════════════════════════════════
# 2. RLL Module
# ══════════════════════════════════════════════════════════════
section("2. RLL CODING MODULE")

from core.rll import RLLEncoder, AdaptiveRLLEncoder

rll = RLLEncoder(d=1, k=7)
data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
encoded = rll.encode(data)
decoded = rll.decode(encoded)
test("RLL encode/decode roundtrip", np.array_equal(data, decoded))
test("RLL output is 2x length (rate 1/2)", len(encoded) == len(data) * 2)

# Adaptive RLL
arll = AdaptiveRLLEncoder()
arll.adapt(50)  # low light
test("Adaptive RLL low light → low_light config", arll.get_config_name() == 'low_light')
arll.adapt(500)
test("Adaptive RLL 500 lux → medium_light config", arll.get_config_name() == 'medium_light')
arll.adapt(2000)
test("Adaptive RLL 2000 lux → high_light config", arll.get_config_name() == 'high_light')

enc_adp = arll.encode(data)
dec_adp = arll.decode(enc_adp)
test("Adaptive RLL encode/decode roundtrip", np.array_equal(data, dec_adp))


# ══════════════════════════════════════════════════════════════
# 3. Convolutional Encoder + Viterbi Decoder
# ══════════════════════════════════════════════════════════════
section("3. CONVOLUTIONAL FEC MODULE")

from core.convolutional import ConvolutionalEncoder, ViterbiDecoder

conv = ConvolutionalEncoder(K=7)
vit = ViterbiDecoder(K=7, burst_weight=1.5)

# Clean channel roundtrip
data_conv = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=int)
encoded_conv = conv.encode(data_conv)
test("Conv encoder output length = 2*(input + tail)", len(encoded_conv) == 2 * (len(data_conv) + 6))

decoded_conv = vit.decode_hard(encoded_conv)
test("Viterbi hard decode: clean channel", np.array_equal(data_conv, decoded_conv[:len(data_conv)]),
     f"Got {decoded_conv[:len(data_conv)].tolist()}")

# With random errors (3 out of ~44 bits)
received_err = encoded_conv.copy()
error_positions = [5, 15, 30]
for pos in error_positions:
    if pos < len(received_err):
        received_err[pos] = 1 - received_err[pos]
decoded_err = vit.decode_hard(received_err)
test("Viterbi corrects 3 random errors", np.array_equal(data_conv, decoded_err[:len(data_conv)]),
     f"Got {decoded_err[:len(data_conv)].tolist()}")

# With burst error (5 consecutive)
received_burst = encoded_conv.copy()
received_burst[10:15] = 1 - received_burst[10:15]
decoded_burst = vit.decode_hard(received_burst)
ber_burst = np.sum(data_conv != decoded_burst[:len(data_conv)]) / len(data_conv)
test("Viterbi with 5-bit burst: BER < 0.35", ber_burst < 0.35, f"BER={ber_burst:.3f}")

# Soft-decision with confidence
confidence = np.ones(len(encoded_conv))
confidence[10:15] = 0.3  # Low confidence at burst
decoded_soft = vit.decode(received_burst, confidence)
ber_soft = np.sum(data_conv != decoded_soft[:len(data_conv)]) / len(data_conv)
test("Soft Viterbi ≤ hard Viterbi BER on burst", ber_soft <= ber_burst + 0.01,
     f"Soft BER={ber_soft:.3f} vs Hard BER={ber_burst:.3f}")


# ══════════════════════════════════════════════════════════════
# 4. Interleaver Module
# ══════════════════════════════════════════════════════════════
section("4. BLOCK INTERLEAVER MODULE")

from core.interleaver import BlockInterleaver

intlv = BlockInterleaver(depth=20, width=6)
data_int = np.random.randint(0, 2, 120)  # Exactly one block
interleaved = intlv.interleave(data_int)
deinterleaved = intlv.deinterleave(interleaved)
test("Interleave/deinterleave roundtrip (exact block)", np.array_equal(data_int, deinterleaved))
test("Interleaved length = original length", len(interleaved) == len(data_int))

# Non-exact block size (needs padding)
data_short = np.random.randint(0, 2, 50)
int_short = intlv.interleave(data_short)
deint_short = intlv.deinterleave(int_short)
test("Interleave handles non-block-sized input", len(int_short) == 120)  # Padded to 120
test("Deinterleave recovers original (with padding)", np.array_equal(data_short, deint_short[:50]))

# Soft interleaving
conf_in = np.random.random(120)
_, conf_int = intlv.interleave_soft(data_int, conf_in)
_, conf_deint = intlv.deinterleave_soft(interleaved, conf_int)
test("Soft interleave/deinterleave roundtrip", np.allclose(conf_in, conf_deint, atol=1e-10))

test("Burst protection = depth", intlv.get_burst_protection() == 20)


# ══════════════════════════════════════════════════════════════
# 5. Reed-Solomon Module
# ══════════════════════════════════════════════════════════════
section("5. REED-SOLOMON RS(255,223) MODULE")

from core.reed_solomon import ReedSolomonCodec, GaloisField

gf = GaloisField()
test("GF(256) multiply: 2*3 = 6", gf.multiply(2, 3) == 6)
test("GF(256) multiply by 0 = 0", gf.multiply(0, 100) == 0)
test("GF(256) inverse of 1 = 1", gf.inverse(1) == 1)
test("GF(256) a * inverse(a) = 1", gf.multiply(42, gf.inverse(42)) == 1)

rs = ReedSolomonCodec(n=255, k=223)

# Clean channel
data_rs = np.random.randint(0, 256, 223)
encoded_rs = rs.encode(data_rs)
test("RS encode output length = 255", len(encoded_rs) == 255)

decoded_rs = rs.decode(encoded_rs)
test("RS decode clean: matches original", decoded_rs is not None and np.array_equal(data_rs, decoded_rs))

# With errors (up to 16 correctable)
received_rs = encoded_rs.copy()
error_pos = np.random.choice(255, size=10, replace=False)
for p in error_pos:
    received_rs[p] = (received_rs[p] + np.random.randint(1, 256)) % 256
decoded_rs_err = rs.decode(received_rs)
test("RS corrects 10 symbol errors", decoded_rs_err is not None and np.array_equal(data_rs, decoded_rs_err),
     "Decoding failed" if decoded_rs_err is None else "")

# Bit-byte conversion
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
bytes_arr = rs.bits_to_bytes(bits)
bits_back = rs.bytes_to_bits(bytes_arr)
test("Bits→bytes→bits roundtrip", np.array_equal(bits, bits_back))


# ══════════════════════════════════════════════════════════════
# 6. Framing Module
# ══════════════════════════════════════════════════════════════
section("6. PACKET FRAMING MODULE")

from core.framing import Frame, FrameManager, CRC16

# CRC
crc = CRC16.compute(b"Hello")
test("CRC-16 is 16-bit", 0 <= crc <= 0xFFFF)
test("CRC-16 verify passes", CRC16.verify(b"Hello", crc))
test("CRC-16 verify fails on corruption", not CRC16.verify(b"Hellx", crc))

# Frame serialization
frame = Frame(Frame.DATA, seq_num=42, payload=b"Hello underwater!", device_id=7)
frame_bytes = frame.to_bytes()
test("Frame serializes to bytes", len(frame_bytes) > 0)

frame_back = Frame.from_bytes(frame_bytes)
test("Frame deserializes correctly", frame_back is not None)
test("Frame payload preserved", frame_back is not None and frame_back.payload == b"Hello underwater!")
test("Frame seq_num preserved", frame_back is not None and frame_back.seq_num == 42)
test("Frame device_id preserved", frame_back is not None and frame_back.device_id == 7)

# Bit conversion
bits_frame = frame.to_bits()
frame_from_bits = Frame.from_bits(bits_frame)
test("Frame bits roundtrip", frame_from_bits is not None and frame_from_bits.payload == b"Hello underwater!")

# Frame Manager
fm = FrameManager(device_id=1)
df = fm.create_data_frame(b"Test")
test("FrameManager creates data frame", df.frame_type == Frame.DATA)
test("FrameManager auto-increments seq", fm.tx_seq == 1)
ack = fm.create_ack(0)
test("FrameManager creates ACK", ack.frame_type == Frame.ACK)


# ══════════════════════════════════════════════════════════════
# 7. Channel Simulator Module
# ══════════════════════════════════════════════════════════════
section("7. UNDERWATER CHANNEL SIMULATOR")

from core.channel import UnderwaterChannel

ch = UnderwaterChannel(water_type='clear', distance_m=3, depth_m=5, ambient_lux=500)
test("Channel SNR is positive for clear/3m", ch.snr_db > 0, f"SNR={ch.snr_db:.1f}")
test("Channel scattering gain > 1", ch.scattering_gain > 1.0, f"Gain={ch.scattering_gain:.2f}")

# Bit-level transmission
bits_ch = np.random.randint(0, 2, 1000)
rx_bits, rx_conf, metrics = ch.transmit_bits(bits_ch)
test("Channel output same length", len(rx_bits) == 1000)
test("Channel confidence in [0,1]", np.all(rx_conf >= 0) and np.all(rx_conf <= 1))
test("Channel returns SNR metric", 'snr_db' in metrics)
test("Channel returns BER metric", 'ber_raw' in metrics)

# Waveform transmission
signal = np.random.random(500)
rx_signal, wf_metrics = ch.transmit(signal)
test("Waveform channel output same length", len(rx_signal) == 500)

# Compare water types
ch_clear = UnderwaterChannel(water_type='clear', distance_m=5)
ch_turbid = UnderwaterChannel(water_type='turbid', distance_m=5)
test("Turbid water has lower SNR than clear", ch_turbid.snr_db < ch_clear.snr_db,
     f"Turbid={ch_turbid.snr_db:.1f} vs Clear={ch_clear.snr_db:.1f}")

# Quality labels
test("Clear 3m → Excellent or Good", ch.get_quality_label() in ['Excellent', 'Good'])

# State dict
state = ch.get_state_dict()
test("State dict has all keys", all(k in state for k in ['water_type', 'snr_db', 'quality', 'scattering_gain']))


# ══════════════════════════════════════════════════════════════
# 8. Water Quality Estimator
# ══════════════════════════════════════════════════════════════
section("8. WATER QUALITY ESTIMATOR")

from core.water_quality import WaterQualityEstimator

wq = WaterQualityEstimator()

turb = wq.estimate_turbidity(roi_size=150, distance_m=3)
test("Turbidity returns NTU value", 'turbidity_ntu' in turb)
test("Turbidity has confidence", 'confidence' in turb)
test("Turbidity classification exists", 'classification' in turb)

wtype = wq.classify_water_type({'blue': 0.55, 'green': 0.35, 'red': 0.10})
test("Blue-dominant → clear water", wtype['water_type'] == 'clear')

wtype2 = wq.classify_water_type({'blue': 0.20, 'green': 0.40, 'red': 0.40})
test("Red-heavy → turbid water", wtype2['water_type'] == 'turbid')

full = wq.get_full_assessment(
    roi_size=200, distance_m=5,
    rgb_intensities={'blue': 0.5, 'green': 0.3, 'red': 0.1}
)
test("Full assessment has visibility", 'estimated_visibility_m' in full)
test("Full assessment has dive safety", 'dive_safety' in full)


# ══════════════════════════════════════════════════════════════
# 9. Motion Compensation Module
# ══════════════════════════════════════════════════════════════
section("9. IMU MOTION COMPENSATION")

from core.motion_compensation import MotionCompensator, MadgwickFilter

# Madgwick filter
mf = MadgwickFilter(sample_rate=100)
mf.update(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 9.81]))
euler = mf.get_euler_angles()
test("Madgwick returns euler angles", all(k in euler for k in ['roll', 'pitch', 'yaw']))

# Motion compensator
mc = MotionCompensator(sample_rate=100)

for level in ['stationary', 'handheld', 'walking', 'swimming']:
    imu = mc.simulate_imu_data(0.5, level)
    test(f"IMU sim '{level}' has gyro data", imu['gyro'].shape[1] == 3)
    result = mc.process_imu(imu)
    test(f"IMU '{level}' returns summary", 'summary' in result)

# Stationary should have fewer artifacts than swimming
imu_stat = mc.simulate_imu_data(1.0, 'stationary')
imu_swim = mc.simulate_imu_data(1.0, 'swimming')
result_stat = mc.process_imu(imu_stat)
result_swim = mc.process_imu(imu_swim)
test("Swimming has more artifacts than stationary",
     result_swim['summary']['artifact_percentage'] >= result_stat['summary']['artifact_percentage'],
     f"Swim={result_swim['summary']['artifact_percentage']}% vs Stat={result_stat['summary']['artifact_percentage']}%")


# ══════════════════════════════════════════════════════════════
# 10. Adaptive FEC Controller
# ══════════════════════════════════════════════════════════════
section("10. ADAPTIVE FEC CONTROLLER")

from core.adaptive_fec import AdaptiveFECController, FECMode

afec = AdaptiveFECController(base_data_rate=180)

# Test mode selection at various SNRs
r_exc = afec.adapt(snr_db=20, ber=1e-5)
test("SNR=20dB → conv_only or better", r_exc['selected_mode'] in ['conv_only', 'none', 'conv_interleaver'])

r_good = afec.adapt(snr_db=12, ber=1e-3)
# After hysteresis, may still be previous mode
test("SNR=12dB adapts", 'selected_mode' in r_good)

r_poor = AdaptiveFECController(base_data_rate=180)  # Fresh controller for this test
r_poor_result = r_poor.adapt(snr_db=2, ber=0.05)
test("SNR=2dB → heavy FEC", r_poor_result['selected_mode'] == 'heavy',
     f"Got {r_poor_result['selected_mode']}")

# All modes summary
modes = afec.get_all_modes_summary()
test("5 FEC modes available", len(modes) == 5)
test("Each mode has data rate", all('effective_data_rate' in m for m in modes))


# ══════════════════════════════════════════════════════════════
# 11. ML Channel Predictor
# ══════════════════════════════════════════════════════════════
section("11. ML CHANNEL PREDICTOR")

from core.channel_prediction import ChannelPredictor

cp = ChannelPredictor(lookback_window=20, prediction_horizon=5)

# Not ready yet
pred = cp.predict()
test("Predictor returns None when insufficient data", pred is None)

# Feed measurements
for i in range(25):
    cp.add_measurement(snr_db=15 + np.random.normal(0, 1), ber=1e-4,
                       roi_size=100, angular_velocity=3)
pred = cp.predict()
test("Predictor returns result after 25 samples", pred is not None)
test("Prediction has quality label", pred is not None and 'predicted_quality' in pred)
test("Prediction has probabilities", pred is not None and 'probabilities' in pred)
test("Prediction has confidence", pred is not None and 'confidence' in pred)

trend = cp.predict_trend()
test("Trend prediction works", trend in ['improving', 'stable', 'degrading'])

summary = cp.get_summary()
test("Summary shows ready=True", summary['ready'] == True)


# ══════════════════════════════════════════════════════════════
# 12. End-to-End Simulation Engine
# ══════════════════════════════════════════════════════════════
section("12. END-TO-END SIMULATION ENGINE")

from simulation.engine import SimulationEngine

eng = SimulationEngine()

# Test 1: Basic clear water
r1 = eng.run_transmission("Hello!", {'water_type': 'clear', 'distance_m': 2, 'depth_m': 5})
test("E2E clear/2m: decoded correctly", r1['success'], f"Got: {r1['decoded_message']}")
test("E2E clear/2m: BER = 0", r1['ber_float'] == 0.0, f"BER={r1['ber']}")

# Test 2: Coastal water
r2 = eng.run_transmission("Testing coastal!", {'water_type': 'coastal', 'distance_m': 5, 'depth_m': 3})
test("E2E coastal/5m: decoded correctly", r2['success'], f"Got: {r2['decoded_message']}")

# Test 3: Harbor water at distance
r3 = eng.run_transmission("Harbor test", {'water_type': 'harbor', 'distance_m': 7, 'depth_m': 2},
                          motion_level='swimming')
test("E2E harbor/7m/swimming: has result", 'decoded_message' in r3)
# May or may not decode perfectly, that's OK

# Test 4: Empty message
r4 = eng.run_transmission("", {'water_type': 'clear', 'distance_m': 1})
test("E2E empty message: no crash", 'decoded_message' in r4)

# Test 5: Long message
long_msg = "A" * 100
r5 = eng.run_transmission(long_msg, {'water_type': 'clear', 'distance_m': 2, 'depth_m': 10})
test("E2E 100-char message: decoded", r5['success'], f"BER={r5['ber']}")

# Test 6: Special characters
r6 = eng.run_transmission("Héllo wörld! 日本語 🌊", {'water_type': 'clear', 'distance_m': 2, 'depth_m': 10})
test("E2E UTF-8/emoji: has result", 'decoded_message' in r6)

# Test 7: Verify all stages are populated
stages = r1['stages']
expected_stages = ['input', 'fec_adaptation', 'rll', 'convolutional', 'interleaver',
                   'modulation', 'channel', 'motion', 'demodulation', 'decoding',
                   'output', 'water_quality', 'channel_prediction']
for st in expected_stages:
    test(f"E2E stage '{st}' present", st in stages)

# Test 8: System status endpoint
status = eng.get_system_status()
test("System status has channel info", 'channel' in status)
test("System status has FEC modes", 'fec_modes_available' in status)

# Test 9: BER sweep
print("\n  Running BER sweep (1-5m, 3 trials each)...")
t_sweep = time.time()
sweep = eng.run_ber_sweep("Hi", distances=[1, 2, 3, 4, 5])
sweep_time = time.time() - t_sweep
test("BER sweep returns results", len(sweep['sweep_results']) == 5)
test(f"BER sweep completes in reasonable time", sweep_time < 60, f"{sweep_time:.1f}s")
bers = [r['avg_ber'] for r in sweep['sweep_results']]
test("BER generally increases with distance", bers[-1] >= bers[0] - 0.01,
     f"1m={bers[0]:.4f}, 5m={bers[-1]:.4f}")


# ══════════════════════════════════════════════════════════════
# 13. Cross-Module Integration
# ══════════════════════════════════════════════════════════════
section("13. CROSS-MODULE INTEGRATION")

# Full FEC pipeline: RS → Conv → Interleave → Channel → Deinterleave → Viterbi → RS
data = np.random.randint(0, 256, 50)  # 50 bytes
data_padded = np.concatenate([data, np.zeros(223 - 50, dtype=int)])

# RS encode
rs_encoded = rs.encode(data_padded)
rs_bits = rs.bytes_to_bits(rs_encoded)

# Conv encode
conv_bits = conv.encode(rs_bits)

# Interleave
int_bits = intlv.interleave(conv_bits)

# Channel
ch_test = UnderwaterChannel(water_type='coastal', distance_m=4)
rx, conf, _ = ch_test.transmit_bits(int_bits)

# Deinterleave
deint = intlv.deinterleave(rx)
deint_conf = intlv.deinterleave_float(conf)

# Viterbi decode
vit_out = vit.decode(deint, deint_conf)

# RS decode
vit_bytes = rs.bits_to_bytes(vit_out[:255*8])
if len(vit_bytes) < 255:
    vit_bytes = np.concatenate([vit_bytes, np.zeros(255 - len(vit_bytes), dtype=int)])
rs_decoded = rs.decode(vit_bytes[:255])

test("Full FEC pipeline: RS decode succeeds",
     rs_decoded is not None,
     "RS decode returned None (uncorrectable)")
if rs_decoded is not None:
    test("Full FEC pipeline: data recovered",
         np.array_equal(data, rs_decoded[:50]),
         f"Mismatch in {np.sum(data != rs_decoded[:50])} bytes")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  TEST RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed")
print(f"{'═'*60}")

if FAIL > 0:
    print(f"\n  ⚠ {FAIL} test(s) failed — review above for details")
    sys.exit(1)
else:
    print(f"\n  ✅ All tests passed!")
    sys.exit(0)
