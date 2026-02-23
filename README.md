# Project Proposal: DeepLink (U-Flash)
**High-Reliability Underwater Optical Camera Communication (OCC) via Scattering Optimization, Sensor Fusion, & Temporal Diversity**

**Team Name:** Launde Overflow
**Team Members:** Adit, Samyagya, Chitraksh, Vinit
**Repository:** (https://github.com/adit-tambe/Hackenza2026_LaundeOverflow)

---

## 1. Executive Summary & Problem Context
The underwater environment is hostile to standard forms of digital communication. Radio Frequency (RF) signals (Wi-Fi, Bluetooth, LTE) suffer total attenuation within centimeters of submergence. Acoustic modems, while capable of long ranges, are exorbitantly expensive, bulky, and power-hungry, restricting their use to massive commercial or military operations. Consequently, recreational divers, search-and-rescue teams, and underwater maintenance crews are forced to rely on manual hand signals. This archaic method fails instantaneously in low-visibility or murky water (high turbidity), during night dives, or when visual line-of-sight is obstructed by marine life, bubbles, or structure.

**DeepLink** is a purely software-driven system that democratizes underwater communication. By exploiting the physics of the most ubiquitous hardware available—the standard smartphone—we transform mobile devices into military-grade optical modems. Using only the built-in LED flashlight (transmitter) and CMOS image sensor (receiver), DeepLink transmits high-speed binary data. Instead of fighting the chaotic underwater environment of scattering, glint, and motion blur, we mathematically outsmart it using advanced signaling theory, computer vision, Machine Learning, and multi-layered Forward Error Correction (FEC).

---

## 2. Core Innovations: The Physical & Optical Layer
Attempting to capture standard pulses of light via video frames is structurally limited by the camera's frame rate (e.g., 30 FPS equals a maximum theoretical data rate of 15 bits per second). Worse, murky water disperses the beam. Our pipeline bypasses native OEM image processing to achieve robust, high-frequency signal detection without custom hardware addons.

### 2.1 The "Scattering" Pivot (Physics-Aware Detection)
Traditional optical communication systems treat water scattering (turbidity) as "noise" that blurs the signal. DeepLink reverses this logic. When light scatters in murky water, it creates a massive "halo" of illumination. Our OpenCV pipeline actively exploits this phenomenon, using the expanded scattering volume to increase the camera's hit area (Region of Interest, or RoI). Because the scattering enlarges the signal footprint, device alignment is mathematically more forgiving underwater than in clear air.

### 2.2 Rolling-Shutter Extraction & Oversampling
While smartphone flashlights are natively locked to relatively low toggle speeds (~10-30 Hz) due to hardware I2C bus limits, we engineered a **Temporal Oversampling Pipeline**. Most smartphone CMOS sensors use a "Rolling Shutter," scanning the image row-by-row. By toggling the LED at high frequencies, we paint distinct horizontal spatial stripes across a single captured image frame. Our algorithm isolates these pixel rows (e.g., bright = 1, dark = 0), effectively achieving a sampling rate orders of magnitude higher than the 30 FPS native video capture rate.

### 2.3 The "Bokeh" Hack (Virtual Scattering for Clear Water)
In highly un-turbid (clear) water, the transmitter flashlight appears as a tiny, hard-to-hit pixel dot—providing an insufficient surface area for rolling-shutter stripes. We solve this by using the `Camera2` API to forcefully lock the receiver's lens to Minimum Focal Distance (Macro Mode). This violently blurs the distant light source, blooming it into a massive "Bokeh ball." We thus artificially recreate the beneficial scattering effect in clear water entirely through software.

### 2.4 Environmental Filtering 
The sun, wave reflections (caustics), and AC-powered room lights (tubelights) easily trigger false positives. DeepLink uses a physics-based filter stack:
1. **Background Subtraction (Frame Differencing):** We subtract the previous frame from the current one. Static lights (the sun) mathematically evaluate to zero and disappear.
2. **Tubelight Rejection (Gate Filter):** Tubelights cause fat, slow 100Hz stripes. Our pipeline categorizes stripe width, rejecting thick bands and accepting only the ultra-thin bands indicative of our high-speed LED signal.

---

## 3. Core Innovations: Sensor Fusion & Motion Resilience 
An underwater diver's hands are never still. Shaking causes the light blob to leap out of the processing frame.

### 3.1 IMU-Fused Predictive Tracking (Kalman Filter)
Rather than forcing the camera's computer vision to execute expensive, frantic searches for the lost light blob, we tap into the smartphone's Gyroscope and Accelerometer at 100-200Hz. If the IMU detects the phone tilting UP, our Kalman Filter instantly predicts that the signal blob will shift DOWN in the camera's frame. We "teleport" the RoI bounding box to the predicted location *before* the next frame is processed, establishing a sticky, unbreakable lock.

### 3.2 RGB-Weighted Chrominance Isolation
White LED light separates underwater. Red light is absorbed almost instantly, leaving the red color channel flooded with dark noise. Our algorithm abandons grayscale conversion. We dynamically extract the RGB channels and apply the formula `Signal = 0.7*Green + 0.2*Blue`. By systematically ignoring the Red channel, we strip out a massive vector of visual noise.

---

## 4. Core Innovations: The Data Link Layer
Attempting to send pure ASCII English text (e.g., "OXYGEN LOW" takes 80 bits) takes seconds to transmit. In the chaotic ocean, long transmissions are fatally vulnerable to "burst errors" (a bubble passing the lens). We rebuilt the protocol to be blazingly fast and mathematical bulletproof.

### 4.1 Tokenized "Diver Syntax"
We mapped critical diver communications into native binary tokens. Instead of 80 bits of English, "OXYGEN LOW" is mapped to a 4-bit identifier (e.g., `0010`). This compresses logical transmission time by 95%, allowing mission-critical alerts to flash across the channel in fractions of a second.

### 4.2 Military-Grade Synchronization: Barker Codes
How do we know where a message starts if the screen is nothing but murky gray pixels? We prepend every packet with an **11-Bit Barker Code** (`10110111000`). Barker Codes are mathematically proven to possess perfect auto-correlation. Treating the pixel column as a waveform, we use the device processor/GPU to "slide" this numerical template over the noisy image (Matched Filtering). Even if the signal is technically fainter than ambient sunlight, the cross-correlation math triggers a massive, unmistakable numerical spike exactly at the start of the message.

### 4.3 The "Healer" Pipeline (Concatenated Forward Error Correction)
Because we have no audio back-channel to request a packet resend, the receiver must mathematically resurrect broken data on its own.
* **Inner Code (Hamming):** We use Hamming (7,4) parity blocks targeting bit-flips caused by micro-noise, such as silt blurring a single stripe.
* **Outer Code (Reed-Solomon):** We use polynomial Reed-Solomon math to mathematically repair total "burst" erasures. If a fish entirely obscures the camera for a second, Reed-Solomon interpolates the missing blocks.
* **Temporal Diversity (Rule of Three):** The payload is transmitted three times sequentially, and the receiver applies bit-level Majority Voting. If all three packets arrive brutally mangled, the algorithm overlays them to vote on the most likely true arrangement.

### 4.4 AI-Powered Soft Autocorrect (Contextual RNN)
If the noise profile is so severe that even Reed-Solomon math fails, we route the fragmented string into a tiny, on-device Recurrent Neural Network (RNN) trained exclusively on our Diver Vocabulary. If the math pipeline outputs `OXY_EN L_W`, the AI probabilistically "hallucinates" the missing letters based on context, displaying the accurate command without crashing the pipeline.

---

## 5. The Application Ecosystem & UX
The algorithmic power is housed within a stunning, professional native application ecosystem crafted for harsh environments.

### 5.1 Hydro-Lock & Haptics
Capacitive touchscreens misfire entirely underwater due to conductivity. DeepLink introduces **Hydro-Lock**: we disable the touchscreen trigger and route the "Start/Stop Receiver" logic to the smartphone's Proximity Sensor. The diver covers the top bezel with their thumb to initiate the decode sequence.
Simultaneously, we map the Signal-to-Noise Ratio (SNR) to haptic feedback. A short buzz indicates the Barker Code was detected; a heavy pulse indicates a successful decode, allowing divers to operate the modem by "feel" alone.

### 5.2 The Dashboard & Visualizer
The central interface is built using a Flask framework, offering a beautiful, mobile-first Glassmorphism UI. 
* **Live Analytics:** Using Chart.js, the dashboard renders the raw signal waveform live on the screen, verifying the Rolling Shutter capture.
* **Adaptive Settings:** Users can toggle environmental presets (Harbor vs. Clear Water), instantly adapting the algorithm's thresholding metrics for the local geography.
* **Error Tracking:** Real-time Bit Error Rate (BER) logs guarantee scientific validity during demonstrations.

### 5.3 UI Mockups & Flows
We designed high-fidelity graphical mockups demonstrating the professional styling of the DeepLink interface.

````carousel
![Home Dashboard](C:\Users\adits\.gemini\antigravity\brain\d06981de-542d-44a5-803d-19ed74334f30\home_mockup.png)
<!-- slide -->
![Transmission Monitor](C:\Users\adits\.gemini\antigravity\brain\d06981de-542d-44a5-803d-19ed74334f30\transmit_mockup.png)
<!-- slide -->
![Live Analytics](C:\Users\adits\.gemini\antigravity\brain\d06981de-542d-44a5-803d-19ed74334f30\results_mockup.png)
````

**Live Application Recording (Simulation Engine)**
The following demonstrates the live, responsive nature of the DeepLink web application running the simulator backend:
![DeepLink App Navigation](C:\Users\adits\.gemini\antigravity\brain\d06981de-542d-44a5-803d-19ed74334f30\capture_screens.webp)

---

## 6. Vision and Scalability
DeepLink proves that the ocean is no longer a localized digital dead zone. What begins as a diver-to-diver messaging app scales into a multi-billion-dollar framework:
1. **Underwater IoT:** Telemetry sent from seafloor environmental sensors directly to a passing diver's smartphone.
2. **AUV Data Offload:** Autonomous Underwater Vehicles (drones) transmitting massive data payloads directly to research vessel receivers using gigabit laser analogues of our algorithm. 
3. **AR Head-Up Navigation:** Overly signaling paths and waypoints via signal strength gradients integrated into the diver's camera feed.

**DeepLink** brings the reliability of deep-space telecommunications down to the ocean floor, using devices everyone already has in their pockets.
