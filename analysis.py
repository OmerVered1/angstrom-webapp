"""
Analysis engine for Radial Heat Wave / Angstrom Method calculations.
Adapted from the original desktop application.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List
import re
import io


@dataclass
class AnalysisParams:
    """Parameters for the analysis."""
    model_name: str
    test_date: str
    test_time: str
    t_cal: str  # C80 start time (HH:MM:SS)
    t_src: str  # Keithley start time (HH:MM:SS)
    r1_mm: float  # Inner radius in mm
    r2_mm: float  # Outer radius in mm
    c80_time_unit: str
    c80_pwr_unit: str
    src_time_unit: str
    src_pwr_unit: str
    use_calibration: bool
    system_lag: float  # seconds
    analysis_mode: str  # "Auto" or "Manual"


@dataclass
class AnalysisResults:
    """Results from the analysis."""
    # Signal parameters
    amplitude_a1: float
    amplitude_a2: float
    period_t: float
    frequency_f: float
    angular_freq_w: float
    raw_lag_dt: float
    raw_phase_phi: float
    ln_term: float
    
    # Thermal diffusivity results
    alpha_combined_raw: float
    alpha_combined_cal: float
    alpha_phase_raw: float
    alpha_phase_cal: float
    
    # Calibrated values
    net_lag_dt: float
    net_phase_phi: float
    
    # Selected range
    t_min: float
    t_max: float
    
    # Marker positions (for visualization)
    marker_src_time: float
    marker_cal_time: float


# Unit conversion factors
TIME_FACTORS = {"Seconds": 1.0, "Minutes": 60.0, "Hours": 3600.0, "ms": 0.001}
POWER_FACTORS = {"mW": 1.0, "Watts": 1000.0, "uW": 0.001}


def clean_read(content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """
    Parse CSV/TSV data from uploaded file content.
    Tries multiple encodings and delimiters.
    """
    for enc in ['utf-16', 'utf-8-sig', 'utf-8', 'cp1255', 'latin-1']:
        try:
            text = content.decode(enc, errors='ignore')
            lines = text.splitlines()
            data = []
            
            for line in lines:
                row = re.split(r'[,\t;]', line.strip())
                if len(row) >= 2:
                    try:
                        time_val = float(row[0].replace('"', '').strip())
                        value_val = float(row[1].replace('"', '').strip())
                        data.append([time_val, value_val])
                    except (ValueError, IndexError):
                        continue
            
            if len(data) > 10:
                return pd.DataFrame(data, columns=['time', 'value'])
        except Exception:
            continue
    
    return None


def sync_and_filter_data(
    df_cal: pd.DataFrame,
    df_src: pd.DataFrame,
    params: AnalysisParams
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Apply unit conversions, time synchronization, and filtering.
    Returns: (t_src_sync, v_src, t_cal_sync, v_cal)
    """
    # Apply unit conversions
    df_cal = df_cal.copy()
    df_src = df_src.copy()
    
    df_cal['time'] *= TIME_FACTORS[params.c80_time_unit]
    df_cal['value'] *= POWER_FACTORS[params.c80_pwr_unit]
    df_src['time'] *= TIME_FACTORS[params.src_time_unit]
    df_src['value'] *= POWER_FACTORS[params.src_pwr_unit]
    
    # Calculate time offset between start times
    t_cal_start = datetime.strptime(params.t_cal, '%H:%M:%S')
    t_src_start = datetime.strptime(params.t_src, '%H:%M:%S')
    t_offset = (t_src_start - t_cal_start).total_seconds()
    
    # Synchronize source time
    t_src_sync = df_src['time'] + t_offset
    v_src = df_src['value']
    
    # Filter calorimeter data to overlapping region
    mask_c = (df_cal['time'] >= t_src_sync.min() - 500) & (df_cal['time'] <= t_src_sync.max() + 500)
    t_cal_sync = df_cal['time'][mask_c].reset_index(drop=True)
    v_cal = df_cal['value'][mask_c].reset_index(drop=True)
    
    # Reset source index
    t_src_sync = t_src_sync.reset_index(drop=True)
    v_src = v_src.reset_index(drop=True)
    
    return t_src_sync, v_src, t_cal_sync, v_cal


def auto_detect_peaks(
    t_arr: pd.Series,
    v_arr: pd.Series,
    min_distance_fraction: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect peaks in the signal.
    Returns: (peak_indices, peak_times)
    """
    distance = max(int(len(v_arr) * min_distance_fraction), 10)
    peaks, _ = find_peaks(v_arr.values, distance=distance)
    return peaks, t_arr.iloc[peaks].values


def snap_to_peak(
    t_click: float,
    t_arr: pd.Series,
    v_arr: pd.Series,
    window: int = 50
) -> Tuple[float, float]:
    """
    Snap a clicked point to the nearest peak within a window.
    Returns: (peak_time, peak_value)
    """
    idx = (np.abs(t_arr.values - t_click)).argmin()
    start = max(0, idx - window)
    end = min(len(t_arr), idx + window)
    
    local_region = v_arr.iloc[start:end]
    local_max_idx = local_region.idxmax()
    
    return t_arr.loc[local_max_idx], v_arr.loc[local_max_idx]


def run_auto_analysis(
    t_cal: pd.Series,
    v_cal: pd.Series,
    t_src: pd.Series,
    v_src: pd.Series,
    params: AnalysisParams,
    t_min: float,
    t_max: float
) -> AnalysisResults:
    """
    Run automatic analysis using peak detection.
    """
    # Find peaks in source and response signals
    peaks_src, _ = find_peaks(v_src.values, distance=len(v_src) // 20)
    peaks_cal, _ = find_peaks(v_cal.values, distance=len(v_cal) // 20)
    
    if len(peaks_src) < 2 or len(peaks_cal) < 1:
        raise ValueError("Not enough peaks detected. Try manual mode or adjust the selection range.")
    
    # Calculate period from source peaks
    peak_times_src = t_src.iloc[peaks_src].values
    T = float(np.mean(np.diff(peak_times_src)))
    w = 2 * np.pi / T
    
    # Calculate time lag (dt) between source and response
    ts_first = t_src.iloc[peaks_src[0]]
    tc_first = t_cal.iloc[peaks_cal[0]]
    dt = tc_first - ts_first
    
    # If negative, use second response peak
    if dt < 0 and len(peaks_cal) > 1:
        tc_first = t_cal.iloc[peaks_cal[1]]
        dt = tc_first - ts_first
    
    # Calculate amplitudes
    a1 = (v_src.max() - v_src.min()) / 2
    a2 = (v_cal.max() - v_cal.min()) / 2
    
    # Marker positions for visualization
    marker_src = ts_first
    marker_cal = tc_first
    
    return _calculate_thermal_diffusivity(
        a1, a2, T, w, dt, params, t_min, t_max, marker_src, marker_cal
    )


def run_manual_analysis(
    t_cal: pd.Series,
    v_cal: pd.Series,
    t_src: pd.Series,
    v_src: pd.Series,
    params: AnalysisParams,
    t_min: float,
    t_max: float,
    click_coords: List[Tuple[float, float]]  # [(t1_src, v1), (t2_src, v2), (t1_cal, v1)]
) -> AnalysisResults:
    """
    Run manual analysis using user-selected peak positions.
    click_coords: [first_src_peak, second_src_peak, response_peak]
    """
    if len(click_coords) < 3:
        raise ValueError("Need 3 click coordinates for manual analysis")
    
    # Snap to nearest peaks
    ts1, vs1 = snap_to_peak(click_coords[0][0], t_src, v_src)
    ts2, vs2 = snap_to_peak(click_coords[1][0], t_src, v_src)
    tc1, vc1 = snap_to_peak(click_coords[2][0], t_cal, v_cal)
    
    # Calculate period and angular frequency
    T = abs(ts2 - ts1)
    w = 2 * np.pi / T
    
    # Calculate time lag
    dt = tc1 - ts1
    
    # Calculate amplitudes
    a1 = (v_src.max() - v_src.min()) / 2
    a2 = (v_cal.max() - v_cal.min()) / 2
    
    return _calculate_thermal_diffusivity(
        a1, a2, T, w, dt, params, t_min, t_max, ts1, tc1
    )


def _calculate_thermal_diffusivity(
    a1: float,
    a2: float,
    T: float,
    w: float,
    dt: float,
    params: AnalysisParams,
    t_min: float,
    t_max: float,
    marker_src: float,
    marker_cal: float
) -> AnalysisResults:
    """
    Core thermal diffusivity calculation using the Angstrom method.
    """
    # Convert radii to meters
    r1 = params.r1_mm / 1000
    r2 = params.r2_mm / 1000
    
    # Calculate phase from raw time lag
    phi = w * abs(dt)
    
    # Calculate logarithmic term
    if a1 <= 0 or a2 <= 0 or r1 <= 0 or r2 <= 0:
        raise ValueError("Amplitudes and radii must be positive")
    
    ln_val = np.log((a1 / a2) * np.sqrt(r1 / r2))
    
    if ln_val <= 0:
        raise ValueError("Log term is non-positive. Check amplitude ratio and radii.")
    
    # Raw thermal diffusivity - Combined method
    # α = ω(r2-r1)² / (2φ·ln(A1/A2·√(r1/r2)))
    alpha_combined_raw = (w * (r2 - r1) ** 2) / (2 * phi * ln_val)
    
    # Raw thermal diffusivity - Phase-only method
    # α = (ω/2) · ((r2-r1)/φ)²
    alpha_phase_raw = (w / 2) * ((r2 - r1) / phi) ** 2
    
    # Apply calibration if enabled
    lag = params.system_lag if params.use_calibration else 0
    dt_net = abs(dt) - lag
    phi_net = w * dt_net if dt_net > 0 else 0
    
    # Calibrated thermal diffusivity
    if dt_net > 0 and phi_net > 0:
        alpha_combined_cal = (w * (r2 - r1) ** 2) / (2 * phi_net * ln_val)
        alpha_phase_cal = (w / 2) * ((r2 - r1) / phi_net) ** 2
    else:
        alpha_combined_cal = -1
        alpha_phase_cal = -1
    
    return AnalysisResults(
        amplitude_a1=a1,
        amplitude_a2=a2,
        period_t=T,
        frequency_f=1 / T,
        angular_freq_w=w,
        raw_lag_dt=abs(dt),
        raw_phase_phi=phi,
        ln_term=ln_val,
        alpha_combined_raw=alpha_combined_raw,
        alpha_combined_cal=alpha_combined_cal,
        alpha_phase_raw=alpha_phase_raw,
        alpha_phase_cal=alpha_phase_cal,
        net_lag_dt=dt_net,
        net_phase_phi=phi_net,
        t_min=t_min,
        t_max=t_max,
        marker_src_time=marker_src,
        marker_cal_time=marker_cal
    )


def format_scientific(value: float, precision: int = 2) -> str:
    """Format a value in scientific notation."""
    if value < 0:
        return "N/A"
    return f"{value:.{precision}e}"
