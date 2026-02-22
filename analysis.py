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
    Parse CSV/TSV/XLS data from uploaded file content.
    Automatically detects Keithley and C80 file formats.
    """
    filename_lower = filename.lower()
    
    # Handle Excel files (.xls, .xlsx)
    if filename_lower.endswith(('.xls', '.xlsx')):
        result = _read_excel_file(content, filename)
        if result is not None:
            return result
        # Fallback: some .xls files are actually tab-separated text
        # Try reading as text file
        return _read_text_file(content, filename)
    
    # Handle text files (CSV, TXT, DAT)
    return _read_text_file(content, filename)


def _read_excel_file(content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Read Excel file, detecting C80 format."""
    try:
        # Try reading as Excel
        excel_file = io.BytesIO(content)
        df_raw = None
        
        # Try different engines
        engines = ['xlrd', 'openpyxl', None]  # None = let pandas choose
        for engine in engines:
            try:
                excel_file.seek(0)
                if engine:
                    df_raw = pd.read_excel(excel_file, header=None, engine=engine)
                else:
                    df_raw = pd.read_excel(excel_file, header=None)
                break
            except Exception:
                continue
        
        if df_raw is None or len(df_raw) == 0:
            return None
        
        # Look for C80 format markers
        # C80 files have "Time(s)" header and "HeatFlow" somewhere
        header_row = None
        time_col = None
        power_col = None
        
        for idx in range(min(20, len(df_raw))):  # Check first 20 rows
            row = df_raw.iloc[idx]
            row_str = ' '.join([str(v) for v in row.values if pd.notna(v)]).lower()
            
            # Look for column headers containing time and heatflow
            if ('time' in row_str) and ('heatflow' in row_str or 'heat flow' in row_str):
                header_row = idx
                # Find the column indices
                for col_idx, val in enumerate(row.values):
                    if pd.notna(val):
                        val_str = str(val).lower()
                        if 'time' in val_str and time_col is None:
                            time_col = col_idx
                        elif 'heatflow' in val_str or 'heat flow' in val_str:
                            power_col = col_idx
                break
        
        if header_row is not None and time_col is not None and power_col is not None:
            # Read data starting from row after header
            data = []
            for idx in range(header_row + 1, len(df_raw)):
                try:
                    time_val = df_raw.iloc[idx, time_col]
                    power_val = df_raw.iloc[idx, power_col]
                    
                    # Handle potential string values with comma decimal
                    if isinstance(time_val, str):
                        time_val = float(time_val.replace(',', '.'))
                    else:
                        time_val = float(time_val)
                    
                    if isinstance(power_val, str):
                        power_val = float(power_val.replace(',', '.'))
                    else:
                        power_val = float(power_val)
                    
                    data.append([time_val, power_val])
                except (ValueError, TypeError):
                    continue
            
            if len(data) > 10:
                return pd.DataFrame(data, columns=['time', 'value'])
        
        # Fallback: try first two numeric columns
        return _fallback_parse_dataframe(df_raw)
        
    except Exception as e:
        print(f"Excel read error: {e}")
        return None
        
        # Fallback: try first two numeric columns
        return _fallback_parse_dataframe(df_raw)
        
    except Exception as e:
        print(f"Excel read error: {e}")
        return None


def _detect_encoding_order(content: bytes) -> list:
    """
    Return a prioritised list of encodings to try based on BOM detection.
    UTF-16 files (common from Windows instruments) often have a BOM; we
    put the matching codec first so we decode correctly on the first try.
    """
    if content.startswith(b'\xff\xfe'):          # UTF-16 LE BOM
        return ['utf-16', 'utf-16-le', 'utf-8-sig', 'utf-8', 'cp1255', 'latin-1']
    elif content.startswith(b'\xfe\xff'):         # UTF-16 BE BOM
        return ['utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'cp1255', 'latin-1']
    elif content.startswith(b'\xef\xbb\xbf'):     # UTF-8 BOM
        return ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1255', 'latin-1']
    else:                                          # No BOM – try common ones
        return ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1255', 'latin-1']


def _read_text_file(content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Read text file, detecting Keithley or C80 format."""
    for enc in _detect_encoding_order(content):
        try:
            text = content.decode(enc, errors='ignore')
            # Strip BOM character that utf-16-le/be codecs leave in the string.
            if enc in ('utf-16-le', 'utf-16-be'):
                text = text.lstrip('\ufeff')
            # Skip decodings that look like they've interpreted wide-char
            # (e.g. UTF-16) bytes incorrectly as single-byte encodings.
            # Such decodes often contain NUL characters in the ASCII range
            # which lead to garbage lines and failed parsing. If we see
            # NULs near the start, try the next encoding.
            if '\x00' in text[:200]:
                continue
            lines = text.splitlines()
            
            if not lines:
                continue
            
            # Detect file format
            format_type = _detect_format(lines)
            
            if format_type == 'keithley':
                return _parse_keithley(lines)
            elif format_type == 'c80':
                return _parse_c80(lines)
            else:
                # If detection failed, still try the C80/Keithley parsers
                # before falling back to the very generic parser. Some
                # exports include metadata at the top which can push the
                # real header past the initial detection window.
                parsed = _parse_c80(lines)
                if parsed is not None:
                    return parsed
                parsed = _parse_keithley(lines)
                if parsed is not None:
                    return parsed
                # Generic fallback parser
                return _parse_generic(lines)
                
        except Exception as e:
            continue
    
    return None


def _detect_format(lines: list) -> str:
    """Detect the file format based on content."""
    header_text = '\n'.join(lines[:15]).lower()
    
    # Keithley format: starts with "# Run" comments and has specific columns
    if '# run' in header_text or 'elapsed(s)' in header_text or 'power(w)' in header_text:
        return 'keithley'
    
    # C80 format: has "HeatFlow" and "Time(s)" patterns
    if 'heatflow' in header_text or ('time(s)' in header_text and 'temperature' in header_text):
        return 'c80'
    
    return 'generic'


def _parse_keithley(lines: list) -> Optional[pd.DataFrame]:
    """
    Parse Keithley format:
    - Header lines start with #
    - Column header: Index,Computer_Time,Elapsed(s),Voltage(V),Current(A),Resistance(Ohm),Power(W)
    - Time: Elapsed(s) (col 2), Power: Power(W) (col 6) (0-indexed)
    """
    data = []
    time_col = None
    power_col = None
    data_started = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip comment lines
        if line.startswith('#'):
            continue
        
        # Split by comma
        row = [v.strip() for v in line.split(',')]
        
        # Detect header row
        if not data_started:
            row_lower = [v.lower() for v in row]
            
            # Find column indices
            for i, col in enumerate(row_lower):
                if 'elapsed' in col and ('s)' in col or 'sec' in col):
                    time_col = i
                elif 'power' in col and ('w)' in col or 'watt' in col):
                    power_col = i
            
            # If we found the columns, mark header found
            if time_col is not None and power_col is not None:
                data_started = True
                continue
            elif time_col is not None or power_col is not None:
                # Partial match, keep looking
                continue
        
        # Parse data row
        if data_started and time_col is not None and power_col is not None:
            try:
                if len(row) > max(time_col, power_col):
                    time_val = float(row[time_col])
                    power_val = float(row[power_col])
                    data.append([time_val, power_val])
            except (ValueError, IndexError):
                continue
    
    if len(data) > 10:
        return pd.DataFrame(data, columns=['time', 'value'])
    
    return None


def _parse_c80(lines: list) -> Optional[pd.DataFrame]:
    """
    Parse C80 format:
    - Metadata header lines
    - Column header: Time(s), Sample Temperature(°C), HeatFlow(mW)
    - Time: col 0, Power: col 2 (HeatFlow)
    - Tab or comma separated
    """
    data = []
    time_col = None
    power_col = None
    data_started = False
    delimiter = '\t'  # Default to tab
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Try to detect delimiter from header line
        if not data_started:
            if '\t' in line:
                delimiter = '\t'
            elif ',' in line:
                delimiter = ','
            elif ';' in line:
                delimiter = ';'
        
        # Split by detected delimiter
        row = [v.strip() for v in line.split(delimiter)]
        
        # If only one column, try other delimiters
        if len(row) <= 1 and not data_started:
            for delim in ['\t', ',', ';']:
                test_row = [v.strip() for v in line.split(delim)]
                if len(test_row) > 1:
                    row = test_row
                    delimiter = delim
                    break
        
        # Detect header row
        if not data_started:
            row_lower = [v.lower() for v in row]
            row_text = ' '.join(row_lower)
            
            # Find column indices - look for time and heatflow
            for i, col in enumerate(row_lower):
                if ('time' in col and ('s)' in col or 'sec' in col or col == 'time')):
                    time_col = i
                elif 'heatflow' in col or 'heat flow' in col or 'heat_flow' in col:
                    power_col = i
            
            # If we found the columns, mark header found
            if time_col is not None and power_col is not None:
                data_started = True
                continue
        
        # Parse data row
        if data_started and time_col is not None and power_col is not None:
            try:
                if len(row) > max(time_col, power_col):
                    time_val = float(row[time_col].replace(',', '.'))
                    power_val = float(row[power_col].replace(',', '.'))
                    data.append([time_val, power_val])
            except (ValueError, IndexError):
                continue
    
    if len(data) > 10:
        return pd.DataFrame(data, columns=['time', 'value'])
    
    return None


def _parse_generic(lines: list) -> Optional[pd.DataFrame]:
    """Generic parser - tries to find first two numeric columns."""
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
    
    return None


def _fallback_parse_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fallback: find first two numeric columns in a DataFrame."""
    data = []
    
    for idx in range(len(df)):
        row_vals = []
        for col in range(len(df.columns)):
            try:
                val = float(df.iloc[idx, col])
                row_vals.append(val)
                if len(row_vals) == 2:
                    break
            except (ValueError, TypeError):
                continue
        
        if len(row_vals) == 2:
            data.append(row_vals)
    
    if len(data) > 10:
        return pd.DataFrame(data, columns=['time', 'value'])
    
    return None


def extract_start_time(content: bytes, filename: str) -> Optional[str]:
    """
    Extract start time from file header.
    Returns time in HH:MM:SS format, or None if not found.
    """
    filename_lower = filename.lower()
    
    # Handle Excel files
    if filename_lower.endswith(('.xls', '.xlsx')):
        result = _extract_start_time_excel(content)
        if result:
            return result
        # Fallback: some .xls files are actually text
        return _extract_start_time_text(content)
    
    # Handle text files
    return _extract_start_time_text(content)


def _extract_start_time_excel(content: bytes) -> Optional[str]:
    """Extract start time from Excel file header."""
    try:
        excel_file = io.BytesIO(content)
        df_raw = None
        
        # Try different engines
        for engine in ['xlrd', 'openpyxl', None]:
            try:
                excel_file.seek(0)
                if engine:
                    df_raw = pd.read_excel(excel_file, header=None, engine=engine)
                else:
                    df_raw = pd.read_excel(excel_file, header=None)
                break
            except Exception:
                continue
        
        if df_raw is None:
            return None
        
        # Look for "Zone Start Time" pattern (C80 format)
        for idx in range(min(15, len(df_raw))):
            row_text = ' '.join([str(v) for v in df_raw.iloc[idx].values if pd.notna(v)])
            
            # C80 pattern: "Zone Start Time : 16/02/2026 12:52:04"
            if 'zone start time' in row_text.lower():
                match = re.search(r'(\d{1,2}:\d{2}:\d{2})', row_text)
                if match:
                    return match.group(1)
        
        return None
    except Exception:
        return None


def _extract_start_time_text(content: bytes) -> Optional[str]:
    """Extract start time from text file header."""
    for enc in _detect_encoding_order(content):
        try:
            text = content.decode(enc, errors='ignore')
            # Strip BOM character that utf-16-le/be codecs leave in the string.
            if enc in ('utf-16-le', 'utf-16-be'):
                text = text.lstrip('\ufeff')
            # If this decode contains NULs early on it is likely the wrong
            # single-byte interpretation of a wide-char encoding (e.g. UTF-16).
            # Skip such decodes to prefer the correct encoding.
            if '\x00' in text[:200]:
                continue
            lines = text.splitlines()[:40]  # Check first 40 lines
            
            # First pass: look for specific patterns (more reliable)
            for line in lines:
                line_lower = line.lower()
                
                # C80 pattern: "Zone Start Time : 16/02/2026 12:52:04"
                if 'zone start time' in line_lower:
                    match = re.search(r'(\d{1,2}:\d{2}:\d{2})', line)
                    if match:
                        return match.group(1)
                
                # Keithley pattern: "# Started: 2026-02-16 14:48:43"
                if line.strip().startswith('#') and 'started' in line_lower:
                    match = re.search(r'(\d{1,2}:\d{2}:\d{2})', line)
                    if match:
                        return match.group(1)
            
            # Second pass: generic fallback (less reliable)
            for line in lines:
                line_lower = line.lower()
                # Skip lines that are clearly not start times
                if 'creation' in line_lower or 'user' in line_lower:
                    continue
                    
                # Look for lines that might contain start time
                if any(x in line_lower for x in ['start time', 'begin', 'started']):
                    match = re.search(r'(\d{1,2}:\d{2}:\d{2})', line)
                    if match:
                        return match.group(1)
            
            return None
        except Exception:
            continue
    
    return None


def detect_file_type(content: bytes, filename: str) -> str:
    """
    Detect file type: 'keithley', 'c80', or 'unknown'.
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith(('.xls', '.xlsx')):
        # Excel files are typically C80
        return 'c80'
    
    for enc in _detect_encoding_order(content):
        try:
            text = content.decode(enc, errors='ignore')
            # Strip BOM character that utf-16-le/be codecs leave in the string.
            if enc in ('utf-16-le', 'utf-16-be'):
                text = text.lstrip('\ufeff')
            # Prefer decodes without early NULs
            if '\x00' in text[:200]:
                continue
            header_text = '\n'.join(text.splitlines()[:20]).lower()
            
            if '# run' in header_text or 'elapsed(s)' in header_text or 'power(w)' in header_text:
                return 'keithley'
            
            if 'heatflow' in header_text or ('time(s)' in header_text and 'temperature' in header_text):
                return 'c80'
            
            return 'unknown'
        except Exception:
            continue
    
    return 'unknown'


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract test date from filename.
    Looks for patterns like:
    - 160226 (DDMMYY) -> 16/02/2026
    - 16-02-26, 16.02.26, 16_02_26
    - 2026-02-16 (YYYY-MM-DD)
    Returns date in DD/MM/YYYY format or None.
    """
    import os
    # Get just the filename without extension
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # Pattern 1: DDMMYY (6 digits at end or standalone) - most common for your files
    match = re.search(r'(\d{2})(\d{2})(\d{2})(?:\D|$)', basename)
    if match:
        dd, mm, yy = match.groups()
        # Assume 20xx for year
        year = f"20{yy}"
        try:
            # Validate it's a real date
            datetime.strptime(f"{dd}/{mm}/{year}", "%d/%m/%Y")
            return f"{dd}/{mm}/{year}"
        except ValueError:
            pass
    
    # Pattern 2: DD-MM-YY or DD.MM.YY or DD_MM_YY
    match = re.search(r'(\d{2})[-._](\d{2})[-._](\d{2,4})', basename)
    if match:
        dd, mm, yy = match.groups()
        year = f"20{yy}" if len(yy) == 2 else yy
        try:
            datetime.strptime(f"{dd}/{mm}/{year}", "%d/%m/%Y")
            return f"{dd}/{mm}/{year}"
        except ValueError:
            pass
    
    # Pattern 3: YYYY-MM-DD (ISO format)
    match = re.search(r'(\d{4})[-._](\d{2})[-._](\d{2})', basename)
    if match:
        yyyy, mm, dd = match.groups()
        try:
            datetime.strptime(f"{dd}/{mm}/{yyyy}", "%d/%m/%Y")
            return f"{dd}/{mm}/{yyyy}"
        except ValueError:
            pass
    
    return None


def extract_temperature_from_filename(filename: str) -> Optional[float]:
    """
    Extract temperature from filename.
    Looks for patterns like:
    - 50c, 50C, 50°C -> 50.0
    - 100c, 150C -> 100.0, 150.0
    - temp50, T50
    Returns temperature as float or None.
    """
    import os
    basename = os.path.splitext(os.path.basename(filename))[0].lower()
    
    # Helper to validate temperature range
    def valid_temp(val):
        try:
            fval = float(val)
            return -50.0 <= fval <= 500.0
        except Exception:
            return False

    # Pattern 1: Number followed by 'c' (most common) - e.g., "50c", "100c"
    match = re.search(r'(\d+(?:\.\d+)?)\s*[°]?c(?:\s|$|[^a-z])', basename)
    if match and valid_temp(match.group(1)):
        return float(match.group(1))

    # Pattern 2: "temp" or "t" followed by number - e.g., "temp50", "t100"
    match = re.search(r'(?:temp|t)[\s_-]*(\d+(?:\.\d+)?)', basename)
    if match and valid_temp(match.group(1)):
        return float(match.group(1))

    # Pattern 3: Number followed by "deg" or "degrees" - e.g., "50deg"
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:deg|degrees)', basename)
    if match and valid_temp(match.group(1)):
        return float(match.group(1))

    return None


def extract_sample_name_from_filename(filename: str) -> Optional[str]:
    """
    Extract sample/model name from filename.
    Tries to find meaningful name parts excluding date and temperature.
    """
    import os
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # Remove common prefixes
    name = re.sub(r'^(?:c80|keithley)\s*', '', basename, flags=re.IGNORECASE)
    
    # Remove date patterns (DDMMYY)
    name = re.sub(r'\d{6}', '', name)
    
    # Remove temperature patterns (50c, 100c, etc.)
    name = re.sub(r'\d+\s*[°]?c(?:\s|$)', '', name, flags=re.IGNORECASE)
    
    # Clean up
    name = re.sub(r'[-_]+', ' ', name)  # Replace separators with spaces
    name = re.sub(r'\s+', ' ', name)    # Collapse multiple spaces
    name = name.strip()
    
    if name and len(name) > 1:
        return name.title()  # Capitalize first letter of each word
    
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
    
    ln_arg = (a1 / a2) * np.sqrt(r1 / r2)
    if ln_arg <= 0:
        ln_val = 1.0
    else:
        ln_val = np.log(ln_arg)
        if ln_val <= 0:
            ln_val = 1.0
    
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
