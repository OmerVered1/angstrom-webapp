"""
Database module for storing Angstrom analysis results.
Uses SQLite for persistent storage.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import base64


DB_PATH = os.path.join(os.path.dirname(__file__), "angstrom_results.db")


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            test_date TEXT,
            test_time TEXT,
            
            -- Input file info
            c80_filename TEXT,
            keithley_filename TEXT,
            
            -- Parameters
            r1_mm REAL,
            r2_mm REAL,
            t_cal TEXT,
            t_src TEXT,
            c80_time_unit TEXT,
            c80_pwr_unit TEXT,
            src_time_unit TEXT,
            src_pwr_unit TEXT,
            analysis_mode TEXT,
            
            -- Calibration
            use_calibration INTEGER,
            system_lag REAL,
            
            -- Selection range
            t_min REAL,
            t_max REAL,
            
            -- Results - Signal Parameters
            amplitude_a1 REAL,
            amplitude_a2 REAL,
            period_t REAL,
            frequency_f REAL,
            angular_freq_w REAL,
            raw_lag_dt REAL,
            raw_phase_phi REAL,
            ln_term REAL,
            
            -- Results - Thermal Diffusivity
            alpha_combined_raw REAL,
            alpha_combined_cal REAL,
            alpha_phase_raw REAL,
            alpha_phase_cal REAL,
            
            -- Calibrated values
            net_lag_dt REAL,
            net_phase_phi REAL,
            
            -- Graph image (base64 encoded)
            graph_image TEXT,
            
            -- Full JSON data for any additional info
            extra_data TEXT
        )
    """)
    
    conn.commit()
    conn.close()


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
    graph_image: Optional[bytes] = None,
    extra_data: Optional[Dict] = None
) -> int:
    """Save an analysis result to the database. Returns the ID of the new record."""
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Encode graph image to base64 if provided
    graph_image_b64 = base64.b64encode(graph_image).decode('utf-8') if graph_image else None
    
    # Serialize extra data to JSON
    extra_data_json = json.dumps(extra_data) if extra_data else None
    
    cursor.execute("""
        INSERT INTO analyses (
            model_name, test_date, test_time,
            c80_filename, keithley_filename,
            r1_mm, r2_mm, t_cal, t_src,
            c80_time_unit, c80_pwr_unit, src_time_unit, src_pwr_unit,
            analysis_mode, use_calibration, system_lag,
            t_min, t_max,
            amplitude_a1, amplitude_a2, period_t, frequency_f, angular_freq_w,
            raw_lag_dt, raw_phase_phi, ln_term,
            alpha_combined_raw, alpha_combined_cal, alpha_phase_raw, alpha_phase_cal,
            net_lag_dt, net_phase_phi,
            graph_image, extra_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name, test_date, test_time,
        c80_filename, keithley_filename,
        r1_mm, r2_mm, t_cal, t_src,
        c80_time_unit, c80_pwr_unit, src_time_unit, src_pwr_unit,
        analysis_mode, int(use_calibration), system_lag,
        t_min, t_max,
        amplitude_a1, amplitude_a2, period_t, frequency_f, angular_freq_w,
        raw_lag_dt, raw_phase_phi, ln_term,
        alpha_combined_raw, alpha_combined_cal, alpha_phase_raw, alpha_phase_cal,
        net_lag_dt, net_phase_phi,
        graph_image_b64, extra_data_json
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return analysis_id


def get_all_analyses() -> List[Dict[str, Any]]:
    """Get all analyses from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, created_at, model_name, test_date, analysis_mode,
               r1_mm, r2_mm, alpha_combined_raw, alpha_combined_cal,
               alpha_phase_raw, alpha_phase_cal, use_calibration
        FROM analyses
        ORDER BY created_at DESC
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific analysis by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        # Decode graph image from base64
        if result.get('graph_image'):
            result['graph_image_bytes'] = base64.b64decode(result['graph_image'])
        # Parse extra data from JSON
        if result.get('extra_data'):
            result['extra_data'] = json.loads(result['extra_data'])
        return result
    return None


def delete_analysis(analysis_id: int) -> bool:
    """Delete an analysis by ID. Returns True if successful."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    return deleted


def get_analysis_count() -> int:
    """Get total number of analyses in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM analyses")
    count = cursor.fetchone()[0]
    conn.close()
    
    return count


# Initialize database on module import
init_database()
