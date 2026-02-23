# U-Flash: Underwater Smartphone Optical Communication

U-Flash is an innovative underwater smartphone-to-smartphone optical communication system. It enables low-bandwidth, high-reliability data transmission using standard smartphone LEDs and cameras, uniquely leveraging water scattering to enhance performance.

## 🌊 Key Features

- **Scattering-Enhanced Range**: Unlike traditional systems, U-Flash exploits Mies and Rayleigh scattering to enlarge the optical Region of Interest (RoI).
- **Multi-Layer FEC**: Robust error correction using Convolutional Coding + Reed-Solomon + Interleaving to combat underwater burst errors.
- **IMU Motion Compensation**: Real-time stabilization using smartphone sensors to reduce Bit Error Rate (BER) by 2-5× in handheld scenarios.
- **Water Quality Estimation**: Dual-purpose technology that transmits data while simultaneously measuring water turbidity (NTU).
- **ML Channel Prediction**: LSTM-based models to forecast channel quality and adapt transmission parameters.

## 📱 Mobile App UI

The project includes a mobile-first web dashboard built with Flask, featuring:
- **Home Dashboard**: Real-time system status and quick actions.
- **Transmission Control**: Configurable parameters for water type, distance, depth, and ambient light.
- **Live Results**: Visual waveforms and BER analytics.
- **System Architecture**: Detailed breakdown of the transmission pipeline.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, NumPy
- **Core Algorithms**: Reed-Solomon, Viterbi (Convolutional), Madgwick Filter, LSTM
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (Chart.js)
- **Concepts**: PPM Modulation, Rolling-Shutter Decoding, Beer-Lambert Law

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Flask (`pip install flask`)
- NumPy (`pip install numpy`)

### Installation & Running
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd hackenza/uflash
   ```
2. Start the simulation server:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser (use Mobile View in Chrome DevTools for the best experience).

## 📄 References
Inspired by the research paper: *Chi et al., "U-Flash: Improving Underwater Optical Communication by Scattering Effect," ACM IMWUT, 2024.*

---
*Created for Hackenza 2026. This project demonstrates a low-cost, democratization of underwater communication using COTS (Commercial Off-The-Shelf) hardware.*
