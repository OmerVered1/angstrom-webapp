"""
Database module for storing Angstrom analysis results.
Uses Supabase for persistent cloud storage.
"""

import json
import os
import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict, Any
import base64

# Initialize Supabase client
def get_supabase_client():
    """Get Supabase client using credentials from secrets or environment."""
    from supabase import create_client, Client
    
    # Try to get from Streamlit secrets first, then environment variables
    try:
        url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
        key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
    except:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in secrets or environment")
    
    return create_client(url, key)


def init_database():
    """Initialize database - table must be created in Supabase dashboard."""
    # Table is created via Supabase SQL editor, not programmatically
    pass


def save_analysis(
    model_name: str,
    test_date: str,
    test_time: str,
    c80_filename: str,
    keithley_filename: str,
    r1_mm: float,
    r2_mm: float,
    t_cal: str,
    t_src: str,
    c80_time_unit: str,
    c80_pwr_unit: str,
    src_time_unit: str,
    src_pwr_unit: str,
    analysis_mode: str,
    use_calibration: bool,
    system_lag: float,
    t_min: float,
    t_max: float,
    amplitude_a1: float,
    amplitude_a2: float,
    period_t: float,
    frequency_f: float,
    angular_freq_w: float,
    raw_lag_dt: float,
    raw_phase_phi: float,
    ln_term: float,
    alpha_combined_raw: float,
    alpha_combined_cal: float,
    alpha_phase_raw: float,
    alpha_phase_cal: float,
    net_lag_dt: float,
    net_phase_phi: float,
    temperature_c: Optional[float] = None,
    graph_image: Optional[bytes] = None,
    extra_data: Optional[Dict] = None
) -> int:
    """Save an analysis result to the database. Returns the ID of the new record."""
    
    supabase = get_supabase_client()
    
    # Encode graph image to base64 if provided
    graph_image_b64 = base64.b64encode(graph_image).decode('utf-8') if graph_image else None
    
    # Serialize extra data to JSON
    extra_data_json = json.dumps(extra_data) if extra_data else None
    
    data = {
        'model_name': model_name,
        'test_date': test_date,
        'test_time': test_time,
        'c80_filename': c80_filename,
        'keithley_filename': keithley_filename,
        'r1_mm': r1_mm,
        'r2_mm': r2_mm,
        't_cal': t_cal,
        't_src': t_src,
        'c80_time_unit': c80_time_unit,
        'c80_pwr_unit': c80_pwr_unit,
        'src_time_unit': src_time_unit,
        'src_pwr_unit': src_pwr_unit,
        'analysis_mode': analysis_mode,
        'use_calibration': use_calibration,
        'system_lag': system_lag,
        't_min': t_min,
        't_max': t_max,
        'amplitude_a1': amplitude_a1,
        'amplitude_a2': amplitude_a2,
        'period_t': period_t,
        'frequency_f': frequency_f,
        'angular_freq_w': angular_freq_w,
        'raw_lag_dt': raw_lag_dt,
        'raw_phase_phi': raw_phase_phi,
        'ln_term': ln_term,
        'alpha_combined_raw': alpha_combined_raw,
        'alpha_combined_cal': alpha_combined_cal,
        'alpha_phase_raw': alpha_phase_raw,
        'alpha_phase_cal': alpha_phase_cal,
        'net_lag_dt': net_lag_dt,
        'net_phase_phi': net_phase_phi,
        'temperature_c': temperature_c,
        'graph_image': graph_image_b64,
        'extra_data': extra_data_json
    }
    
    result = supabase.table('analyses').insert(data).execute()
    
    if result.data:
        return result.data[0]['id']
    return -1


def get_all_analyses() -> List[Dict[str, Any]]:
    """Get all analyses from the database."""
    supabase = get_supabase_client()
    
    result = supabase.table('analyses').select(
        'id, created_at, model_name, test_date, analysis_mode, '
        'r1_mm, r2_mm, amplitude_a1, amplitude_a2, period_t, frequency_f, angular_freq_w, '
        'raw_lag_dt, raw_phase_phi, ln_term, '
        'alpha_combined_raw, alpha_combined_cal, alpha_phase_raw, alpha_phase_cal, '
        'use_calibration, system_lag, net_lag_dt, temperature_c'
    ).order('created_at', desc=True).execute()
    
    return result.data if result.data else []


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific analysis by ID."""
    supabase = get_supabase_client()
    
    result = supabase.table('analyses').select('*').eq('id', analysis_id).execute()
    
    if result.data and len(result.data) > 0:
        row = result.data[0]
        # Decode graph image from base64
        if row.get('graph_image'):
            row['graph_image_bytes'] = base64.b64decode(row['graph_image'])
        # Parse extra data from JSON
        if row.get('extra_data'):
            row['extra_data'] = json.loads(row['extra_data'])
        return row
    return None


def delete_analysis(analysis_id: int) -> bool:
    """Delete an analysis by ID. Returns True if successful."""
    supabase = get_supabase_client()
    
    result = supabase.table('analyses').delete().eq('id', analysis_id).execute()
    
    return result.data is not None and len(result.data) > 0


def get_analysis_count() -> int:
    """Get total number of analyses in the database."""
    supabase = get_supabase_client()
    
    result = supabase.table('analyses').select('id', count='exact').execute()
    
    return result.count if result.count else 0


def update_model_name(analysis_id: int, new_name: str) -> bool:
    """Update the model name for an analysis. Returns True if successful."""
    supabase = get_supabase_client()
    
    result = supabase.table('analyses').update({'model_name': new_name}).eq('id', analysis_id).execute()
    
    return result.data is not None and len(result.data) > 0


# Initialize database on module import
init_database()
