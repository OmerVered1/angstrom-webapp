"""
OCR module for extracting results from uploaded images.
Uses easyocr for text extraction and regex for parsing values.
"""

import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractedResults:
    """Extracted results from an image."""
    model_name: Optional[str] = None
    test_date: Optional[str] = None
    r1_mm: Optional[float] = None
    r2_mm: Optional[float] = None
    amplitude_a1: Optional[float] = None
    amplitude_a2: Optional[float] = None
    period_t: Optional[float] = None
    frequency_f: Optional[float] = None
    angular_freq_w: Optional[float] = None
    raw_lag_dt: Optional[float] = None
    raw_phase_phi: Optional[float] = None
    ln_term: Optional[float] = None
    alpha_combined_raw: Optional[float] = None
    alpha_combined_cal: Optional[float] = None
    alpha_phase_raw: Optional[float] = None
    alpha_phase_cal: Optional[float] = None
    system_lag: Optional[float] = None
    net_lag_dt: Optional[float] = None
    use_calibration: bool = False
    confidence: float = 0.0


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using easyocr.
    Falls back to empty string if OCR fails.
    """
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        import io
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Initialize reader (English)
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Extract text
        results = reader.readtext(image_np)
        
        # Combine all text
        text_lines = [result[1] for result in results]
        return '\n'.join(text_lines)
    
    except ImportError:
        return ""
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def parse_scientific_notation(text: str) -> Optional[float]:
    """Parse scientific notation like 1.23e-06 or 1.23×10⁻⁶"""
    # Standard scientific notation: 1.23e-06
    match = re.search(r'([\d.]+)[eE]([+-]?\d+)', text)
    if match:
        try:
            return float(f"{match.group(1)}e{match.group(2)}")
        except:
            pass
    
    # Unicode superscript: 1.23×10⁻⁶
    superscript_map = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                       '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
                       '⁻': '-', '⁺': '+'}
    match = re.search(r'([\d.]+)\s*[×x]\s*10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)', text)
    if match:
        try:
            exp = ''.join(superscript_map.get(c, c) for c in match.group(2))
            return float(f"{match.group(1)}e{exp}")
        except:
            pass
    
    return None


def extract_float(text: str, pattern: str) -> Optional[float]:
    """Extract a float value following a pattern. Handles negative numbers."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    return None


def extract_float_negative(text: str, pattern: str) -> Optional[float]:
    """Extract a float value that may be negative."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    return None


def parse_results_from_text(text: str) -> ExtractedResults:
    """
    Parse extracted OCR text to find result values.
    Uses various regex patterns to match common formats.
    """
    results = ExtractedResults()
    confidence_count = 0
    total_fields = 0
    
    # Clean text
    text = text.replace('\n', ' ').replace('  ', ' ')
    
    # Model name
    match = re.search(r'Model[:\s]+([A-Za-z0-9_\-]+)', text, re.IGNORECASE)
    if match:
        results.model_name = match.group(1)
        confidence_count += 1
    total_fields += 1
    
    # Date
    match = re.search(r'Date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', text, re.IGNORECASE)
    if match:
        results.test_date = match.group(1)
        confidence_count += 1
    total_fields += 1
    
    # Radii
    r1 = extract_float(text, r'r1\s*[=:]\s*([\d.]+)')
    if r1 is None:
        r1 = extract_float(text, r'r₁\s*[=:]\s*([\d.]+)')
    if r1:
        results.r1_mm = r1
        confidence_count += 1
    total_fields += 1
    
    r2 = extract_float(text, r'r2\s*[=:]\s*([\d.]+)')
    if r2 is None:
        r2 = extract_float(text, r'r₂\s*[=:]\s*([\d.]+)')
    if r2:
        results.r2_mm = r2
        confidence_count += 1
    total_fields += 1
    
    # Amplitudes
    a1 = extract_float(text, r'A1\s*[=:]\s*([\d.]+)')
    if a1 is None:
        a1 = extract_float(text, r'A₁\s*[=:]\s*([\d.]+)')
    if a1:
        results.amplitude_a1 = a1
        confidence_count += 1
    total_fields += 1
    
    a2 = extract_float(text, r'A2\s*[=:]\s*([\d.]+)')
    if a2 is None:
        a2 = extract_float(text, r'A₂\s*[=:]\s*([\d.]+)')
    if a2:
        results.amplitude_a2 = a2
        confidence_count += 1
    total_fields += 1
    
    # Period
    period = extract_float(text, r'Period\s*\(?T\)?\s*[=:]\s*([\d.]+)')
    if period is None:
        period = extract_float(text, r'T\s*[=:]\s*([\d.]+)\s*s')
    if period:
        results.period_t = period
        confidence_count += 1
    total_fields += 1
    
    # Frequency
    freq = extract_float(text, r'Frequency\s*\(?f\)?\s*[=:]\s*([\d.]+)')
    if freq:
        results.frequency_f = freq
        confidence_count += 1
    total_fields += 1
    
    # Angular frequency
    omega = extract_float(text, r'[ωw]\s*[=:]\s*([\d.]+)')
    if omega is None:
        omega = extract_float(text, r'Angular\s*Freq[:\s]*([\d.]+)')
    if omega:
        results.angular_freq_w = omega
        confidence_count += 1
    total_fields += 1
    
    # Raw lag (delta t)
    lag = extract_float(text, r'Raw\s*Lag[^:]*[:\s]*([\d.]+)')
    if lag is None:
        lag = extract_float(text, r'[Δδ]t\s*[=:]\s*([\d.]+)')
    if lag is None:
        lag = extract_float(text, r'dt\s*[=:]\s*([\d.]+)')
    if lag:
        results.raw_lag_dt = lag
        confidence_count += 1
    total_fields += 1
    
    # Phase (Raw Phase φ)
    phase = extract_float(text, r'[Rr]aw\s*[Pp]hase[^:]*[:\s]*([\d.]+)')
    if phase is None:
        phase = extract_float(text, r'[φϕ]\s*[):\s]*([\d.]+)')
    if phase is None:
        phase = extract_float(text, r'Phase\s*\([φϕ]\)[:\s]*([\d.]+)')
    if phase is None:
        phase = extract_float(text, r'Phase[:\s]*([\d.]+)\s*rad')
    if phase is None:
        # Try to find a number followed by "rad" that's likely the phase
        phase = extract_float(text, r'([\d.]+)\s*rad')
    if phase:
        results.raw_phase_phi = phase
        confidence_count += 1
    total_fields += 1
    
    # Log term - can be negative! Look for various patterns
    ln_term = extract_float_negative(text, r'[Ll]og\s*[Tt]erm[^=]*[=:\s]*(-?[\d.]+)')
    if ln_term is None:
        ln_term = extract_float_negative(text, r'[Ll]n\s*[Tt]erm[^=]*[=:\s]*(-?[\d.]+)')
    if ln_term is None:
        ln_term = extract_float_negative(text, r'ln\s*\([^)]+\)[^=]*[=:\s]*(-?[\d.]+)')
    if ln_term is None:
        ln_term = extract_float_negative(text, r'ln[^=]*[=:]\s*(-?[\d.]+)')
    if ln_term is None:
        # Look for ln(A1/A2) pattern
        ln_term = extract_float_negative(text, r'ln\s*\(\s*A\s*[12]\s*/\s*A\s*[12]\s*\)[^=]*[=:\s]*(-?[\d.]+)')
    if ln_term is None:
        # Look for negative decimal near "ln" 
        match = re.search(r'ln.*?(-\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            try:
                ln_term = float(match.group(1))
            except:
                pass
    if ln_term:
        results.ln_term = ln_term
        confidence_count += 1
    total_fields += 1
    
    # Thermal diffusivity - look for various α patterns
    # Pattern: α_raw, α raw, alpha raw, α_comb, etc.
    
    # First try to find α_raw or alpha_raw pattern (simple format from images)
    alpha_raw = None
    for pattern in [
        r'α[_\s]*raw[:\s]*([\d.]+[eE][+-]?\d+)',
        r'alpha[_\s]*raw[:\s]*([\d.]+[eE][+-]?\d+)',
        r'α[_\s]*(?:comb|combined)[^:]*(?:raw)?[:\s]*([\d.]+[eE][+-]?\d+)',
        r'[Rr]aw.*?α[:\s]*([\d.]+[eE][+-]?\d+)',
        r'α[:\s]*([\d.]+[eE][+-]?\d+)\s*m',  # α: 1.23e-06 m²/s
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                alpha_raw = float(match.group(1))
                break
            except:
                pass
    
    if alpha_raw:
        results.alpha_combined_raw = alpha_raw
        confidence_count += 1
    else:
        # Try to find any scientific notation in typical thermal diffusivity range
        all_sci = re.findall(r'([\d.]+[eE][+-]?\d+)', text)
        for val_str in all_sci:
            try:
                val = float(val_str)
                if 1e-10 < val < 1e-4:  # Typical range for thermal diffusivity
                    if results.alpha_combined_raw is None:
                        results.alpha_combined_raw = val
                        confidence_count += 1
                    elif results.alpha_phase_raw is None:
                        results.alpha_phase_raw = val
                        confidence_count += 1
                    break
            except:
                pass
    total_fields += 1
    
    # Try to find calibrated alpha values: α_cal, alpha_cal, etc.
    alpha_cal = None
    for pattern in [
        r'α[_\s]*cal[:\s]*([\d.]+[eE][+-]?\d+)',
        r'alpha[_\s]*cal[:\s]*([\d.]+[eE][+-]?\d+)',
        r'α[_\s]*(?:comb|combined)[^:]*cal[:\s]*([\d.]+[eE][+-]?\d+)',
        r'[Cc]al.*?α[:\s]*([\d.]+[eE][+-]?\d+)',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                alpha_cal = float(match.group(1))
                break
            except:
                pass
    
    if alpha_cal:
        results.alpha_combined_cal = alpha_cal
        results.use_calibration = True
        confidence_count += 1
    total_fields += 1
    
    # Phase method
    alpha_phase_matches = re.findall(r'α[_\s]*phase[^:]*(?:raw)?[:\s]*([\d.]+[eE][+-]?\d+)', text, re.IGNORECASE)
    if alpha_phase_matches:
        try:
            results.alpha_phase_raw = float(alpha_phase_matches[0])
            confidence_count += 1
        except:
            pass
    total_fields += 1
    
    alpha_phase_cal_matches = re.findall(r'α[_\s]*phase[^:]*cal[:\s]*([\d.]+[eE][+-]?\d+)', text, re.IGNORECASE)
    if alpha_phase_cal_matches:
        try:
            results.alpha_phase_cal = float(alpha_phase_cal_matches[0])
            results.use_calibration = True
            confidence_count += 1
        except:
            pass
    total_fields += 1
    
    # System lag
    sys_lag = extract_float(text, r'System\s*Lag[:\s]*([\d.]+)')
    if sys_lag:
        results.system_lag = sys_lag
        results.use_calibration = True
        confidence_count += 1
    total_fields += 1
    
    # Net lag - various formats: "Net Lag (Δt_cal): 364.82", "Net Δt: 123", etc.
    net_lag = None
    for pattern in [
        r'Net\s*Lag[^:]*[:\s]*([\d.]+)',
        r'Net\s*[Δδ]t[^:]*[:\s]*([\d.]+)',
        r'[Δδ]t[_\s]*cal[:\s]*([\d.]+)',
        r'[Δδ]t[_\s]*net[:\s]*([\d.]+)',
        r'Calibrated.*?[Δδ]t[:\s]*([\d.]+)',
    ]:
        net_lag = extract_float(text, pattern)
        if net_lag:
            break
    
    if net_lag:
        results.net_lag_dt = net_lag
        confidence_count += 1
    total_fields += 1
    
    # Calculate confidence
    results.confidence = confidence_count / max(total_fields, 1) * 100
    
    return results


def process_image(image_bytes: bytes) -> Tuple[ExtractedResults, str]:
    """
    Process an uploaded image and extract results.
    Returns (ExtractedResults, raw_ocr_text)
    """
    # Extract text using OCR
    raw_text = extract_text_from_image(image_bytes)
    
    # Parse results from text
    results = parse_results_from_text(raw_text)
    
    return results, raw_text
