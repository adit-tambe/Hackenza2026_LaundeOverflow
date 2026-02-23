"""
U-Flash Water Quality Estimation Module
Estimates turbidity and water type from communication signal characteristics.
Leverages scattering-RoI relationship and RGB channel analysis.
"""

import numpy as np
from typing import Dict, Tuple


class WaterQualityEstimator:
    """
    Dual-purpose module: estimates water quality metrics
    from the same signal used for communication.
    
    - RoI size -> Turbidity (NTU)
    - RGB channel ratios -> Water type classification
    """

    # Calibration data: (distance_m, roi_size) -> turbidity_ntu
    # From controlled experiments with known turbidity standards
    CALIBRATION = {
        'reference_distance': 2.0,
        'curve_a': 0.15,    # Turbidity = a * RoI_normalized^b + c
        'curve_b': 1.8,
        'curve_c': 0.2,
    }

    # RGB ratio signatures for water type classification
    WATER_TYPE_SIGNATURES = {
        'clear': {'blue_ratio': 0.55, 'green_ratio': 0.35, 'red_ratio': 0.10},
        'coastal': {'blue_ratio': 0.40, 'green_ratio': 0.40, 'red_ratio': 0.20},
        'harbor': {'blue_ratio': 0.25, 'green_ratio': 0.45, 'red_ratio': 0.30},
        'turbid': {'blue_ratio': 0.20, 'green_ratio': 0.40, 'red_ratio': 0.40},
    }

    def __init__(self):
        self.history = []
        self.max_history = 100

    def estimate_turbidity(self, roi_size: float, distance_m: float) -> Dict:
        """
        Estimate turbidity from RoI size and distance.
        
        Args:
            roi_size: Measured RoI area in pixels
            distance_m: Estimated distance to transmitter
        
        Returns:
            Dict with turbidity estimate and confidence
        """
        # Normalize RoI by distance squared (inverse square law)
        ref_dist = self.CALIBRATION['reference_distance']
        roi_normalized = roi_size * (ref_dist / max(distance_m, 0.5)) ** 2

        # Apply calibration curve
        a = self.CALIBRATION['curve_a']
        b = self.CALIBRATION['curve_b']
        c = self.CALIBRATION['curve_c']

        turbidity_ntu = a * (roi_normalized ** b) + c
        turbidity_ntu = max(0.1, min(turbidity_ntu, 200.0))  # Clamp to valid range

        # Store in history for averaging
        self.history.append(turbidity_ntu)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Smoothed estimate using recent history
        if len(self.history) >= 5:
            smoothed = np.median(self.history[-10:])
            confidence = 1.0 - min(np.std(self.history[-10:]) / max(smoothed, 1), 0.5)
        else:
            smoothed = turbidity_ntu
            confidence = 0.5

        return {
            'turbidity_ntu': round(float(smoothed), 2),
            'turbidity_raw': round(float(turbidity_ntu), 2),
            'confidence': round(float(confidence), 3),
            'roi_normalized': round(float(roi_normalized), 2),
            'classification': self._classify_turbidity(smoothed),
        }

    def classify_water_type(self, rgb_intensities: Dict[str, float]) -> Dict:
        """
        Classify water type from RGB channel intensity ratios.
        
        Args:
            rgb_intensities: {'red': value, 'green': value, 'blue': value}
        """
        total = sum(rgb_intensities.values()) + 1e-10
        ratios = {
            'blue_ratio': rgb_intensities.get('blue', 0) / total,
            'green_ratio': rgb_intensities.get('green', 0) / total,
            'red_ratio': rgb_intensities.get('red', 0) / total,
        }

        # Find closest water type signature
        best_type = 'coastal'
        best_distance = float('inf')
        scores = {}

        for water_type, signature in self.WATER_TYPE_SIGNATURES.items():
            distance = sum(
                (ratios[k] - signature[k]) ** 2
                for k in ratios
            )
            scores[water_type] = round(1.0 / (1.0 + distance * 100), 3)
            if distance < best_distance:
                best_distance = distance
                best_type = water_type

        return {
            'water_type': best_type,
            'confidence': scores[best_type],
            'all_scores': scores,
            'rgb_ratios': {k: round(v, 3) for k, v in ratios.items()},
        }

    def _classify_turbidity(self, turbidity_ntu: float) -> str:
        """Classify turbidity level."""
        if turbidity_ntu < 1:
            return 'Very Clear'
        elif turbidity_ntu < 5:
            return 'Clear'
        elif turbidity_ntu < 20:
            return 'Moderate'
        elif turbidity_ntu < 50:
            return 'Turbid'
        else:
            return 'Very Turbid'

    def get_full_assessment(self, roi_size: float, distance_m: float,
                           rgb_intensities: Dict[str, float]) -> Dict:
        """Complete water quality assessment."""
        turbidity = self.estimate_turbidity(roi_size, distance_m)
        water_type = self.classify_water_type(rgb_intensities)

        # Visibility estimation (Secchi depth approximation)
        estimated_visibility = max(0.5, 10.0 / max(turbidity['turbidity_ntu'], 0.1))

        return {
            'turbidity': turbidity,
            'water_type': water_type,
            'estimated_visibility_m': round(estimated_visibility, 1),
            'dive_safety': 'Safe' if estimated_visibility > 2 else 'Caution',
        }
