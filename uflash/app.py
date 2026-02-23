"""
U-Flash Dashboard Server
Flask application serving the interactive web dashboard.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from simulation.engine import SimulationEngine

app = Flask(__name__)
engine = SimulationEngine()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/transmit', methods=['POST'])
def transmit():
    """Run a transmission simulation."""
    data = request.get_json()
    message = data.get('message', 'Hello Underwater World!')
    motion_level = data.get('motion_level', 'handheld')

    channel_params = {}
    if 'water_type' in data:
        channel_params['water_type'] = data['water_type']
    if 'distance_m' in data:
        channel_params['distance_m'] = float(data['distance_m'])
    if 'depth_m' in data:
        channel_params['depth_m'] = float(data['depth_m'])
    if 'ambient_lux' in data:
        channel_params['ambient_lux'] = float(data['ambient_lux'])

    result = engine.run_transmission(message, channel_params or None, motion_level)
    return jsonify(result)


@app.route('/api/ber_sweep', methods=['POST'])
def ber_sweep():
    """Run BER vs distance sweep."""
    data = request.get_json()
    message = data.get('message', 'Hello')
    distances = data.get('distances', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = engine.run_ber_sweep(message, distances)
    return jsonify(result)


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    return jsonify(engine.get_system_status())


@app.route('/api/channel', methods=['POST'])
def update_channel():
    """Update channel parameters."""
    data = request.get_json()
    engine.channel.set_parameters(**data)
    return jsonify(engine.channel.get_state_dict())


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  U-Flash Underwater Optical Communication Simulator")
    print("  Dashboard: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
