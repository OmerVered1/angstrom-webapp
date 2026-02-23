"""
Radial Heat Wave Analysis - Streamlit Web Application
Angstrom Method for Thermal Diffusivity Measurement
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import base64

from analysis import (
    AnalysisParams, AnalysisResults,
    clean_read, sync_and_filter_data,
    run_auto_analysis, run_manual_analysis,
    auto_detect_peaks, format_scientific,
    extract_start_time, detect_file_type,
    extract_date_from_filename, extract_temperature_from_filename,
    extract_sample_name_from_filename
)
import database as db
from ocr_extractor import process_image, ExtractedResults


# Page configuration
st.set_page_config(
    page_title="Radial Heat Wave Analysis",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'df_cal': None,
        'df_src': None,
        'cal_filename': None,
        'src_filename': None,
        't_src_sync': None,
        'v_src': None,
        't_cal_sync': None,
        'v_cal': None,
        'selection_range': None,
        'analysis_results': None,
        'step': 1,  # 1=Upload, 2=Select Range, 3=Analysis, 4=Results
        'manual_clicks': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_overview_plot(t_src, v_src, t_cal, v_cal, selection_range=None):
    """Create the full experiment overview plot with optional selection highlight."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Source signal
    fig.add_trace(
        go.Scatter(
            x=t_src, y=v_src,
            name='Source (Keithley)',
            line=dict(color='#3498db', width=1),
            hovertemplate='Time: %{x:.1f}s<br>Power: %{y:.3f} mW<extra>Source</extra>'
        ),
        secondary_y=False
    )
    
    # Response signal
    fig.add_trace(
        go.Scatter(
            x=t_cal, y=v_cal,
            name='Response (C80)',
            line=dict(color='#e74c3c', width=1),
            hovertemplate='Time: %{x:.1f}s<br>Power: %{y:.3f} mW<extra>Response</extra>'
        ),
        secondary_y=True
    )
    
    # Add selection highlight
    if selection_range:
        t_min, t_max = selection_range
        fig.add_vrect(
            x0=t_min, x1=t_max,
            fillcolor="green", opacity=0.15,
            layer="below", line_width=0,
            annotation_text="Selected Region",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title='Full Experiment Overview',
        xaxis_title='Time (s)',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Source Power (mW)", secondary_y=False, color='#3498db')
    fig.update_yaxes(title_text="Response Power (mW)", secondary_y=True, color='#e74c3c')
    
    return fig


def create_selection_plot(t_src, v_src, t_cal, v_cal):
    """Create interactive plot for range selection."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=t_src, y=v_src, name='Source', line=dict(color='#3498db')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=t_cal, y=v_cal, name='Response', line=dict(color='#e74c3c', width=1), opacity=0.6),
        secondary_y=True
    )
    
    fig.update_layout(
        title='📍 Drag to select a STABLE REGION for analysis',
        xaxis_title='Time (s)',
        height=450,
        hovermode='x unified',
        dragmode='select',  # Enable box selection
        selectdirection='h',  # Horizontal selection only
    )
    
    fig.update_yaxes(title_text="Source Power (mW)", secondary_y=False, color='#3498db')
    fig.update_yaxes(title_text="Response Power (mW)", secondary_y=True, color='#e74c3c')
    
    return fig


def create_analysis_plot(t_src, v_src, t_cal, v_cal, results: AnalysisResults, params: AnalysisParams):
    """Create the detailed analysis plot with amplitude markers."""
    fig = go.Figure()
    
    # Filter to selected range
    mask_s = (t_src >= results.t_min) & (t_src <= results.t_max)
    mask_c = (t_cal >= results.t_min) & (t_cal <= results.t_max)
    
    t_s_sel = t_src[mask_s]
    v_s_sel = v_src[mask_s]
    t_c_sel = t_cal[mask_c]
    v_c_sel = v_cal[mask_c]
    
    # Source signal
    fig.add_trace(go.Scatter(
        x=t_s_sel, y=v_s_sel,
        name='Source',
        line=dict(color='#3498db', width=2),
        hovertemplate='Time: %{x:.1f}s<br>Power: %{y:.3f} mW<extra>Source</extra>'
    ))
    
    # Response signal
    fig.add_trace(go.Scatter(
        x=t_c_sel, y=v_c_sel,
        name='Response',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='Time: %{x:.1f}s<br>Power: %{y:.3f} mW<extra>Response</extra>'
    ))
    
    # Mean lines
    mid_s = (v_s_sel.max() + v_s_sel.min()) / 2
    mid_c = (v_c_sel.max() + v_c_sel.min()) / 2
    
    fig.add_hline(y=mid_s, line_dash="dash", line_color='#3498db', opacity=0.6,
                  annotation_text=f"Mean S = {mid_s:.2f}")
    fig.add_hline(y=mid_c, line_dash="dash", line_color='#e74c3c', opacity=0.6,
                  annotation_text=f"Mean R = {mid_c:.2f}")
    
    # Amplitude markers
    fig.add_hline(y=mid_s + results.amplitude_a1, line_dash="dot", line_color='#3498db', opacity=0.4)
    fig.add_hline(y=mid_c + results.amplitude_a2, line_dash="dot", line_color='#e74c3c', opacity=0.4)
    
    # Time markers for delta t
    fig.add_vline(x=results.marker_src_time, line_dash="dashdot", line_color='blue', opacity=0.5,
                  annotation_text="Src Peak")
    fig.add_vline(x=results.marker_cal_time, line_dash="dashdot", line_color='red', opacity=0.5,
                  annotation_text="Resp Peak")
    
    # Annotations
    fig.add_annotation(
        x=t_s_sel.iloc[len(t_s_sel)//4], y=mid_s + results.amplitude_a1,
        text=f"A1 = {results.amplitude_a1:.3f} mW",
        showarrow=True, arrowhead=2,
        font=dict(color='#3498db', size=12)
    )
    
    fig.add_annotation(
        x=t_c_sel.iloc[len(t_c_sel)//4], y=mid_c + results.amplitude_a2,
        text=f"A2 = {results.amplitude_a2:.3f} mW",
        showarrow=True, arrowhead=2,
        font=dict(color='#e74c3c', size=12)
    )
    
    fig.update_layout(
        title=f'Stable Zone Analysis - Δt = {results.raw_lag_dt:.2f}s',
        xaxis_title='Time (s)',
        yaxis_title='Power (mW)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_dashboard_figure(t_src, v_src, t_cal, v_cal, results: AnalysisResults, params: AnalysisParams):
    """Create the complete dashboard figure for saving."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"secondary_y": True}, {"type": "table"}],
            [{"colspan": 2}, None]
        ],
        subplot_titles=("Full Overview", "Results Summary", "Selected Region Analysis"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Plot 1: Full overview
    fig.add_trace(
        go.Scatter(x=t_src, y=v_src, name='Source', line=dict(color='#3498db', width=1)),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=t_cal, y=v_cal, name='Response', line=dict(color='#e74c3c', width=1), opacity=0.5),
        row=1, col=1, secondary_y=True
    )
    
    # Highlight selection
    fig.add_vrect(
        x0=results.t_min, x1=results.t_max,
        fillcolor="green", opacity=0.15,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Plot 2: Results table
    cal_info = f"Lag: {params.system_lag}s" if params.use_calibration else "Disabled"
    
    table_data = [
        ["Parameter", "Value"],
        ["Model", params.model_name],
        ["Date", params.test_date],
        ["r₁ (mm)", f"{params.r1_mm:.2f}"],
        ["r₂ (mm)", f"{params.r2_mm:.2f}"],
        ["A₁ (mW)", f"{results.amplitude_a1:.3f}"],
        ["A₂ (mW)", f"{results.amplitude_a2:.3f}"],
        ["Period T (s)", f"{results.period_t:.2f}"],
        ["Frequency (Hz)", f"{results.frequency_f:.5f}"],
        ["ω (rad/s)", f"{results.angular_freq_w:.5f}"],
        ["Raw Δt (s)", f"{results.raw_lag_dt:.2f}"],
        ["Raw φ (rad)", f"{results.raw_phase_phi:.4f}"],
        ["ln term", f"{results.ln_term:.4f}"],
        ["α_comb (raw)", format_scientific(results.alpha_combined_raw)],
        ["α_phase (raw)", format_scientific(results.alpha_phase_raw)],
        ["Calibration", cal_info],
        ["α_comb (cal)", format_scientific(results.alpha_combined_cal)],
        ["α_phase (cal)", format_scientific(results.alpha_phase_cal)],
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Parameter</b>', '<b>Value</b>'],
                font=dict(size=11),
                align='left',
                fill_color='#2c3e50',
                font_color='white'
            ),
            cells=dict(
                values=[[row[0] for row in table_data[1:]], [row[1] for row in table_data[1:]]],
                font=dict(size=10),
                align='left',
                fill_color=[['#f8f9fa', '#ffffff'] * 9]
            )
        ),
        row=1, col=2
    )
    
    # Plot 3: Selected region analysis
    mask_s = (t_src >= results.t_min) & (t_src <= results.t_max)
    mask_c = (t_cal >= results.t_min) & (t_cal <= results.t_max)
    
    fig.add_trace(
        go.Scatter(x=t_src[mask_s], y=v_src[mask_s], name='Src (selected)',
                   line=dict(color='#3498db', width=2), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_cal[mask_c], y=v_cal[mask_c], name='Resp (selected)',
                   line=dict(color='#e74c3c', width=2), showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Radial Heat Wave Analysis Report - {params.model_name}",
        showlegend=True
    )
    
    return fig


def fig_to_json(fig):
    """Convert a Plotly figure to JSON string for storage."""
    import json
    return fig.to_json()


def json_to_fig(json_str):
    """Recreate a Plotly figure from JSON string."""
    import plotly.io as pio
    return pio.from_json(json_str)


def results_to_dataframe(results: AnalysisResults, params: AnalysisParams) -> pd.DataFrame:
    """Convert results to a DataFrame for display/export."""
    data = {
        'Parameter': [
            'Model Name', 'Test Date', 'Inner Radius r₁ (mm)', 'Outer Radius r₂ (mm)',
            'Amplitude A₁ (mW)', 'Amplitude A₂ (mW)', 'Period T (s)', 'Frequency f (Hz)',
            'Angular Frequency ω (rad/s)', 'Raw Time Lag Δt (s)', 'Raw Phase φ (rad)',
            'Log Term ln(A₁/A₂·√(r₁/r₂))', 'α Combined Method (raw) (m²/s)',
            'α Phase Method (raw) (m²/s)', 'Calibration Enabled', 'System Lag (s)',
            'Net Time Lag (s)', 'Net Phase (rad)',
            'α Combined Method (calibrated) (m²/s)', 'α Phase Method (calibrated) (m²/s)'
        ],
        'Value': [
            params.model_name, params.test_date, params.r1_mm, params.r2_mm,
            f"{results.amplitude_a1:.4f}", f"{results.amplitude_a2:.4f}",
            f"{results.period_t:.2f}", f"{results.frequency_f:.6f}",
            f"{results.angular_freq_w:.6f}", f"{results.raw_lag_dt:.2f}",
            f"{results.raw_phase_phi:.4f}", f"{results.ln_term:.4f}",
            format_scientific(results.alpha_combined_raw),
            format_scientific(results.alpha_phase_raw),
            'Yes' if params.use_calibration else 'No',
            params.system_lag,
            f"{results.net_lag_dt:.2f}",
            f"{results.net_phase_phi:.4f}",
            format_scientific(results.alpha_combined_cal),
            format_scientific(results.alpha_phase_cal)
        ]
    }
    return pd.DataFrame(data)


# ==================== MAIN APPLICATION ====================

def main():
    init_session_state()

    # Global CSS — larger sidebar title, larger nav font, pinned credit
    st.markdown("""
    <style>
    /* Bigger sidebar app title (h2) */
    [data-testid="stSidebar"] h2 {
        font-size: 2.1rem !important;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.1rem;
    }
    /* Larger font for sidebar radio nav options */
    [data-testid="stSidebar"] .stRadio label p {
        font-size: 1.3rem !important;
        font-weight: 500;
        line-height: 1.75;
    }
    /* Give sidebar content breathing room above the pinned credit */
    [data-testid="stSidebarContent"] {
        padding-bottom: 72px !important;
    }
    /* Pinned credit — no custom border; st.divider() above it provides the line */
    .sidebar-credit {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 21rem;
        padding: 6px 1.1rem 10px 1.1rem;
        font-size: 0.69rem;
        color: #888;
        background: var(--secondary-background-color);
        z-index: 99999;
        line-height: 1.6;
        box-sizing: border-box;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🌡️ Radial Heat Wave Analysis")
        st.caption("Dynamic radial system wave based method for thermal diffusivity measurement")
        st.divider()
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["📊 New Analysis", "📷 Upload Results Image", "📋 Results Summary", "📁 Results History", "📈 Statistics", "📐 Theory & Mathematical Evolution"],
            index=0
        )

        st.divider()
        st.caption(f"Total analyses saved: {db.get_analysis_count()}")

        # Native divider — same style as all other sidebar dividers
        st.divider()

        # Credit pinned to sidebar bottom
        st.markdown("""
        <div class="sidebar-credit">
            © Created by Omer Vered<br>
            Hayun Group, BGU University, Israel
        </div>
        """, unsafe_allow_html=True)
    
    if page == "📊 New Analysis":
        render_analysis_page()
    elif page == "📷 Upload Results Image":
        render_upload_image_page()
    elif page == "📋 Results Summary":
        render_results_summary_page()
    elif page == "📁 Results History":
        render_history_page()
    elif page == "📈 Statistics":
        render_statistics_page()
    else:
        render_theory_page()


def render_analysis_page():
    """Render the main analysis page."""
    
    # Step 1: File Upload & Parameters
    st.header("1️⃣ File Upload & Parameters")
    
    col1, col2 = st.columns(2)
    
    # Variables to store auto-detected values from C80 filename
    detected_date = None
    detected_temp = None
    detected_sample = None
    
    with col1:
        st.subheader("Power Response Data File")
        c80_file = st.file_uploader("Upload power response data file", type=['csv', 'txt', 'dat', 'xls', 'xlsx'], key='c80')
        
        # Auto-detect start time from C80 file
        c80_default_time = "12:00:00"
        if c80_file is not None:
            c80_content = c80_file.read()
            c80_file.seek(0)  # Reset for later use
            detected_c80_time = extract_start_time(c80_content, c80_file.name)
            if detected_c80_time:
                c80_default_time = detected_c80_time
                st.success(f"✓ Detected start time: {detected_c80_time}")
            
            # Auto-detect date, temperature, and sample name from filename
            detected_date = extract_date_from_filename(c80_file.name)
            detected_temp = extract_temperature_from_filename(c80_file.name)
            detected_sample = extract_sample_name_from_filename(c80_file.name)
            
            # Show what was detected from filename
            detections = []
            if detected_date:
                detections.append(f"Date: {detected_date}")
            if detected_temp:
                detections.append(f"Temp: {detected_temp}°C")
            if detected_sample:
                detections.append(f"Sample: {detected_sample}")
            if detections:
                st.info(f"✓ From filename: {', '.join(detections)}")
        
        c80_time_unit = st.selectbox("Time Unit", ["Seconds", "Minutes", "Hours", "ms"], key='c80_time')
        c80_pwr_unit = st.selectbox("Power Unit", ["mW", "Watts", "uW"], key='c80_pwr')
        t_cal = st.text_input("Response Start Time (HH:MM:SS)", c80_default_time)

    with col2:
        st.subheader("Power Source Data File")
        src_file = st.file_uploader("Upload power source data file", type=['csv', 'txt', 'dat', 'xls', 'xlsx'], key='src')

        # Auto-detect start time from source file
        src_default_time = "12:05:00"
        if src_file is not None:
            src_content = src_file.read()
            src_file.seek(0)  # Reset for later use
            detected_src_time = extract_start_time(src_content, src_file.name)
            if detected_src_time:
                src_default_time = detected_src_time
                st.success(f"✓ Detected start time: {detected_src_time}")

        src_time_unit = st.selectbox("Time Unit", ["Seconds", "Minutes", "Hours", "ms"], key='src_time')
        src_pwr_unit = st.selectbox("Power Unit", ["Watts", "mW", "uW"], key='src_pwr')
        t_src = st.text_input("Source Start Time (HH:MM:SS)", src_default_time)
    
    st.divider()
    
    # Metadata - use auto-detected values from C80 filename as defaults
    st.subheader("Experiment Metadata")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        default_sample = detected_sample if detected_sample else "Sample-01"
        model_name = st.text_input("Model/Sample Name", default_sample)
    with col2:
        default_date = detected_date if detected_date else datetime.now().strftime("%d/%m/%Y")
        test_date = st.text_input("Test Date (DD/MM/YYYY)", default_date)
    with col3:
        test_time = st.text_input("Test Time (HH:MM)", datetime.now().strftime("%H:%M"))
    with col4:
        # Only use detected_temp if it's in valid range
        if detected_temp is not None and -50.0 <= detected_temp <= 500.0:
            default_temp = detected_temp
        else:
            default_temp = 25.0
        temperature_c = st.number_input("Temperature (°C)", value=default_temp, min_value=-50.0, max_value=500.0, step=0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        r1_mm = st.number_input("Inner Radius r₁ (mm)", value=6.72, min_value=0.1, step=0.01)
    with col2:
        r2_mm = st.number_input("Outer Radius r₂ (mm)", value=14.86, min_value=0.1, step=0.01)
    
    st.divider()
    
    # Calibration settings
    st.subheader("Calibration Settings")
    col1, col2 = st.columns(2)
    with col1:
        use_calibration = st.checkbox("Enable System Calibration", value=True)
    with col2:
        system_lag = st.number_input("System Lag (seconds)", value=105.0, min_value=0.0, step=1.0,
                                      disabled=not use_calibration)
    
    # Analysis mode
    analysis_mode = st.radio("Analysis Mode", ["Auto", "Manual"], horizontal=True,
                             help="Auto: Uses peak detection. Manual: Click to select peaks.")
    
    st.divider()
    
    # Process files button
    if st.button("📤 Load & Process Files", type="primary", use_container_width=True):
        if c80_file is None or src_file is None:
            st.error("Please upload both C80 and Keithley files.")
        else:
            with st.spinner("Processing files..."):
                # Read files
                df_cal = clean_read(c80_file.read(), c80_file.name)
                c80_file.seek(0)  # Reset file pointer
                df_src = clean_read(src_file.read(), src_file.name)
                src_file.seek(0)
                
                if df_cal is None or df_src is None:
                    st.error("Failed to parse one or both files. Check file format.")
                else:
                    # Store in session state
                    st.session_state['df_cal'] = df_cal
                    st.session_state['df_src'] = df_src
                    st.session_state['cal_filename'] = c80_file.name
                    st.session_state['src_filename'] = src_file.name
                    
                    # Create params object
                    params = AnalysisParams(
                        model_name=model_name,
                        test_date=test_date,
                        test_time=test_time,
                        t_cal=t_cal,
                        t_src=t_src,
                        r1_mm=r1_mm,
                        r2_mm=r2_mm,
                        c80_time_unit=c80_time_unit,
                        c80_pwr_unit=c80_pwr_unit,
                        src_time_unit=src_time_unit,
                        src_pwr_unit=src_pwr_unit,
                        use_calibration=use_calibration,
                        system_lag=system_lag,
                        analysis_mode=analysis_mode
                    )
                    st.session_state['params'] = params
                    
                    # Sync data
                    t_src_sync, v_src, t_cal_sync, v_cal = sync_and_filter_data(df_cal, df_src, params)
                    st.session_state['t_src_sync'] = t_src_sync
                    st.session_state['v_src'] = v_src
                    st.session_state['t_cal_sync'] = t_cal_sync
                    st.session_state['v_cal'] = v_cal
                    st.session_state['step'] = 2
                    
                    st.success(f"✅ Files loaded! Response: {len(df_cal)} points, Source: {len(df_src)} points")
                    st.rerun()
    
    # Step 2: Range Selection
    if st.session_state['step'] >= 2 and st.session_state['t_src_sync'] is not None:
        st.header("2️⃣ Select Analysis Region")
        st.info("🎯 Use the slider below to select a stable region for analysis.")
        
        t_src = st.session_state['t_src_sync']
        v_src = st.session_state['v_src']
        t_cal = st.session_state['t_cal_sync']
        v_cal = st.session_state['v_cal']
        
        # Time range slider
        t_min_data = float(min(t_src.min(), t_cal.min()))
        t_max_data = float(max(t_src.max(), t_cal.max()))
        
        selection = st.slider(
            "Select time range (seconds)",
            min_value=t_min_data,
            max_value=t_max_data,
            value=(t_min_data + (t_max_data - t_min_data) * 0.3, 
                   t_min_data + (t_max_data - t_min_data) * 0.7),
            format="%.1f"
        )
        
        st.session_state['selection_range'] = selection
        
        # Show overview plot with selection
        fig = create_overview_plot(t_src, v_src, t_cal, v_cal, selection)
        st.plotly_chart(fig, width="stretch")
        
        # Run analysis button
        if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                params = st.session_state['params']
                t_min, t_max = selection
                
                # Filter data to selection
                mask_s = (t_src >= t_min) & (t_src <= t_max)
                mask_c = (t_cal >= t_min) & (t_cal <= t_max)
                
                try:
                    if params.analysis_mode == "Auto":
                        results = run_auto_analysis(
                            t_cal[mask_c].reset_index(drop=True),
                            v_cal[mask_c].reset_index(drop=True),
                            t_src[mask_s].reset_index(drop=True),
                            v_src[mask_s].reset_index(drop=True),
                            params, t_min, t_max
                        )
                    else:
                        # For manual mode, we'll use auto for now
                        # (manual click selection would require JavaScript interop)
                        st.warning("Manual mode uses automatic peak detection in web version.")
                        results = run_auto_analysis(
                            t_cal[mask_c].reset_index(drop=True),
                            v_cal[mask_c].reset_index(drop=True),
                            t_src[mask_s].reset_index(drop=True),
                            v_src[mask_s].reset_index(drop=True),
                            params, t_min, t_max
                        )
                    
                    st.session_state['analysis_results'] = results
                    st.session_state['step'] = 3
                    st.success("✅ Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Step 3: Results Display
    if st.session_state['step'] >= 3 and st.session_state['analysis_results'] is not None:
        st.header("3️⃣ Analysis Results")
        
        results = st.session_state['analysis_results']
        params = st.session_state['params']
        t_src = st.session_state['t_src_sync']
        v_src = st.session_state['v_src']
        t_cal = st.session_state['t_cal_sync']
        v_cal = st.session_state['v_cal']
        
        # Main results in prominent boxes
        st.subheader("🎯 Thermal Diffusivity Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "α (Combined Method - Raw)",
                format_scientific(results.alpha_combined_raw) + " m²/s"
            )
            if params.use_calibration:
                st.metric(
                    "α (Combined Method - Calibrated)",
                    format_scientific(results.alpha_combined_cal) + " m²/s"
                )
        
        with col2:
            st.metric(
                "α (Phase Method - Raw)",
                format_scientific(results.alpha_phase_raw) + " m²/s"
            )
            if params.use_calibration:
                st.metric(
                    "α (Phase Method - Calibrated)",
                    format_scientific(results.alpha_phase_cal) + " m²/s"
                )
        
        st.divider()
        
        # Detailed analysis plot
        st.subheader("📈 Analysis Visualization")
        analysis_fig = create_analysis_plot(t_src, v_src, t_cal, v_cal, results, params)
        st.plotly_chart(analysis_fig, width="stretch")
        
        # Full results table
        st.subheader("📋 Complete Results Table")
        results_df = results_to_dataframe(results, params)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Save & Export
        st.subheader("💾 Save & Export")
        
        # Generate dashboard figure (do this BEFORE columns so it's available for save)
        dashboard_fig = create_dashboard_figure(t_src, v_src, t_cal, v_cal, results, params)
        
        # Generate graph JSON and cache in session state
        if 'cached_graph_json' not in st.session_state or st.session_state.get('cached_analysis_id') != id(results):
            try:
                graph_json = fig_to_json(dashboard_fig)
                st.session_state['cached_graph_json'] = graph_json
                st.session_state['cached_analysis_id'] = id(results)
                st.session_state['graph_save_status'] = 'success'
            except Exception as e:
                st.session_state['cached_graph_json'] = None
                st.session_state['graph_save_status'] = f'error: {str(e)}'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as CSV
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv_data,
                file_name=f"{params.model_name}_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export graph as HTML (interactive)
            html_data = dashboard_fig.to_html(include_plotlyjs=True, full_html=True)
            st.download_button(
                "📥 Download Report (HTML)",
                data=html_data,
                file_name=f"{params.model_name}_report.html",
                mime="text/html"
            )
        
        with col3:
            # Save to database
            if st.button("💾 Save to Database", type="primary"):
                try:
                    # Use cached graph JSON
                    graph_json = st.session_state.get('cached_graph_json')
                    
                    analysis_id = db.save_analysis(
                        model_name=params.model_name,
                        test_date=params.test_date,
                        test_time=params.test_time,
                        c80_filename=st.session_state['cal_filename'],
                        keithley_filename=st.session_state['src_filename'],
                        r1_mm=params.r1_mm,
                        r2_mm=params.r2_mm,
                        t_cal=params.t_cal,
                        t_src=params.t_src,
                        c80_time_unit=params.c80_time_unit,
                        c80_pwr_unit=params.c80_pwr_unit,
                        src_time_unit=params.src_time_unit,
                        src_pwr_unit=params.src_pwr_unit,
                        analysis_mode=params.analysis_mode,
                        use_calibration=params.use_calibration,
                        system_lag=params.system_lag,
                        t_min=results.t_min,
                        t_max=results.t_max,
                        amplitude_a1=results.amplitude_a1,
                        amplitude_a2=results.amplitude_a2,
                        period_t=results.period_t,
                        frequency_f=results.frequency_f,
                        angular_freq_w=results.angular_freq_w,
                        raw_lag_dt=results.raw_lag_dt,
                        raw_phase_phi=results.raw_phase_phi,
                        ln_term=results.ln_term,
                        alpha_combined_raw=results.alpha_combined_raw,
                        alpha_combined_cal=results.alpha_combined_cal if params.use_calibration else 0.0,
                        alpha_phase_raw=results.alpha_phase_raw,
                        alpha_phase_cal=results.alpha_phase_cal if params.use_calibration else 0.0,
                        net_lag_dt=results.net_lag_dt if params.use_calibration else 0.0,
                        net_phase_phi=results.net_phase_phi if params.use_calibration else 0.0,
                        temperature_c=temperature_c,
                        graph_json=graph_json
                    )
                    st.success(f"✅ Saved to database! (ID: {analysis_id})")
                except Exception as e:
                    st.error(f"Failed to save: {str(e)}")


def render_upload_image_page():
    """Render the page for uploading result images and extracting data."""
    st.header("📷 Upload Results Image")
    st.info("Upload a screenshot or photo of analysis results. The system will attempt to extract values automatically using OCR.")
    
    uploaded_image = st.file_uploader(
        "Upload result image (JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        key='result_image'
    )
    
    if uploaded_image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(uploaded_image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Extract Data")
            
            if st.button("🔬 Run OCR Extraction", type="primary"):
                with st.spinner("Extracting text from image..."):
                    try:
                        image_bytes = uploaded_image.read()
                        uploaded_image.seek(0)
                        
                        extracted, raw_text = process_image(image_bytes)
                        
                        st.session_state['extracted_results'] = extracted
                        st.session_state['raw_ocr_text'] = raw_text
                        st.session_state['uploaded_image_bytes'] = image_bytes
                        
                        if extracted.confidence > 0:
                            st.success(f"✅ Extraction complete! Confidence: {extracted.confidence:.0f}%")
                        else:
                            st.warning("⚠️ Could not extract data automatically. Please enter values manually below.")
                    except Exception as e:
                        st.error(f"OCR failed: {str(e)}. Please enter values manually.")
                        st.session_state['extracted_results'] = ExtractedResults()
    
    st.divider()
    
    # Manual entry / correction form
    st.subheader("📝 Enter or Correct Values")
    st.caption("Fill in or correct the extracted values, then save to database.")
    
    # Get extracted values or defaults
    extracted = st.session_state.get('extracted_results', ExtractedResults())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_name = st.text_input("Model/Sample Name", value=extracted.model_name or "")
        test_date = st.text_input("Test Date", value=extracted.test_date or datetime.now().strftime("%d/%m/%Y"))
        r1_mm = st.number_input("r₁ (mm)", value=extracted.r1_mm or 6.72, min_value=0.1)
        r2_mm = st.number_input("r₂ (mm)", value=extracted.r2_mm or 14.86, min_value=0.1)
        analysis_mode = st.selectbox("Analysis Mode", ["Auto", "Manual"], index=0)
    
    with col2:
        amplitude_a1 = st.number_input("A₁ (mW)", value=extracted.amplitude_a1 or 0.0, format="%.4f")
        amplitude_a2 = st.number_input("A₂ (mW)", value=extracted.amplitude_a2 or 0.0, format="%.4f")
        period_t = st.number_input("Period T (s)", value=extracted.period_t or 0.0, format="%.2f")
        raw_lag_dt = st.number_input("Raw Δt (s)", value=extracted.raw_lag_dt or 0.0, format="%.2f")
        raw_phase_phi = st.number_input("Raw Phase φ (rad)", value=extracted.raw_phase_phi or 0.0, format="%.4f")
    
    with col3:
        # Auto-calculate frequency if period is provided
        default_freq = extracted.frequency_f or (1.0 / extracted.period_t if extracted.period_t and extracted.period_t > 0 else 0.0)
        default_omega = extracted.angular_freq_w or (2 * 3.14159 / extracted.period_t if extracted.period_t and extracted.period_t > 0 else 0.0)
        
        frequency_f = st.number_input("Frequency f (Hz)", value=default_freq, format="%.6f")
        angular_freq_w = st.number_input("Angular Freq ω (rad/s)", value=default_omega, format="%.6f")
        ln_term = st.number_input("Log Term ln(...)", value=extracted.ln_term or 0.0, format="%.4f")
        net_lag_dt = st.number_input("Net Δt (s)", value=extracted.net_lag_dt or 0.0, format="%.2f")
    
    with col4:
        alpha_combined_raw = st.text_input(
            "α Combined Raw (m²/s)",
            value=f"{extracted.alpha_combined_raw:.2e}" if extracted.alpha_combined_raw else ""
        )
        alpha_phase_raw = st.text_input(
            "α Phase Raw (m²/s)",
            value=f"{extracted.alpha_phase_raw:.2e}" if extracted.alpha_phase_raw else ""
        )
        alpha_combined_cal = st.text_input(
            "α Combined Cal (m²/s)",
            value=f"{extracted.alpha_combined_cal:.2e}" if extracted.alpha_combined_cal else ""
        )
        alpha_phase_cal = st.text_input(
            "α Phase Cal (m²/s)",
            value=f"{extracted.alpha_phase_cal:.2e}" if extracted.alpha_phase_cal else ""
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        use_calibration = st.checkbox("Used Calibration", value=extracted.use_calibration)
    with col2:
        system_lag = st.number_input("System Lag (s)", value=extracted.system_lag or 105.0)
    with col3:
        temperature_c = st.number_input("Temperature (°C)", value=25.0, min_value=-50.0, max_value=500.0, step=0.1)
    
    st.divider()
    
    # Save button
    if st.button("💾 Save Results to Database", type="primary", use_container_width=True):
        if not model_name:
            st.error("Model name is required!")
        else:
            try:
                # Parse scientific notation inputs
                def parse_alpha(val):
                    if not val or val.strip() == "":
                        return 0.0
                    try:
                        return float(val)
                    except:
                        return 0.0
                
                # Use user-provided values or calculate if not provided
                freq_f = frequency_f if frequency_f > 0 else (1 / period_t if period_t > 0 else 0)
                omega_w = angular_freq_w if angular_freq_w > 0 else (2 * 3.14159 / period_t if period_t > 0 else 0)
                phi = raw_phase_phi if raw_phase_phi > 0 else (omega_w * raw_lag_dt if omega_w > 0 else 0)
                
                analysis_id = db.save_analysis(
                    model_name=model_name,
                    test_date=test_date,
                    test_time=datetime.now().strftime("%H:%M"),
                    c80_filename="uploaded_image",
                    keithley_filename="uploaded_image",
                    r1_mm=r1_mm,
                    r2_mm=r2_mm,
                    t_cal="00:00:00",
                    t_src="00:00:00",
                    c80_time_unit="Seconds",
                    c80_pwr_unit="mW",
                    src_time_unit="Seconds",
                    src_pwr_unit="mW",
                    analysis_mode=analysis_mode,
                    use_calibration=use_calibration,
                    system_lag=system_lag,
                    t_min=0,
                    t_max=0,
                    amplitude_a1=amplitude_a1,
                    amplitude_a2=amplitude_a2,
                    period_t=period_t,
                    frequency_f=freq_f,
                    angular_freq_w=omega_w,
                    raw_lag_dt=raw_lag_dt,
                    raw_phase_phi=phi,
                    ln_term=ln_term,
                    alpha_combined_raw=parse_alpha(alpha_combined_raw),
                    alpha_combined_cal=parse_alpha(alpha_combined_cal),
                    alpha_phase_raw=parse_alpha(alpha_phase_raw),
                    alpha_phase_cal=parse_alpha(alpha_phase_cal),
                    net_lag_dt=net_lag_dt,
                    net_phase_phi=0,
                    temperature_c=temperature_c,
                    graph_json=None  # No graph for manually uploaded results
                )
                st.success(f"✅ Saved to database! (ID: {analysis_id})")
                st.balloons()
            except Exception as e:
                st.error(f"Failed to save: {str(e)}")
    
    # Show raw OCR text for debugging
    with st.expander("🔧 Debug: Raw OCR Text"):
        raw_text = st.session_state.get('raw_ocr_text', '')
        st.text_area("Extracted text", value=raw_text, height=200)


def render_results_summary_page():
    """Render a comprehensive results summary table."""
    st.header("📋 Results Summary")
    st.caption("Comprehensive overview of all saved analysis results")
    
    analyses = db.get_all_analyses()
    
    if not analyses:
        st.info("No analyses saved yet. Run an analysis or upload results to see them here.")
        return
    
    # Fetch full data for all analyses
    full_analyses = []
    for a in analyses:
        full_data = db.get_analysis_by_id(a['id'])
        if full_data:
            full_analyses.append(full_data)
    
    # Create comprehensive DataFrame with raw values for editing
    summary_data = []
    for a in full_analyses:
        summary_data.append({
            'ID': a['id'],
            'Model': a['model_name'],
            'Date': a['test_date'],
            'T (°C)': a.get('temperature_c') or 25.0,
            'Mode': a['analysis_mode'] or 'Auto',
            'r₁ (mm)': a['r1_mm'] or 0.0,
            'r₂ (mm)': a['r2_mm'] or 0.0,
            'A₁ (mW)': a['amplitude_a1'] or 0.0,
            'A₂ (mW)': a['amplitude_a2'] or 0.0,
            'Period (s)': a['period_t'] or 0.0,
            'f (Hz)': a['frequency_f'] or 0.0,
            'ω (rad/s)': a['angular_freq_w'] or 0.0,
            'Δt (s)': a['raw_lag_dt'] or 0.0,
            'φ (rad)': a['raw_phase_phi'] or 0.0,
            'ln term': a['ln_term'] or 0.0,
            'α_comb (raw)': a['alpha_combined_raw'] or 0.0,
            'α_phase (raw)': a['alpha_phase_raw'] or 0.0,
            'Calibrated': bool(a['use_calibration']),
            'Lag (s)': a['system_lag'] or 0.0,
            'Net Δt (s)': (a['net_lag_dt'] or 0.0) if a['use_calibration'] else 0.0,
            'α_comb (cal)': (a['alpha_combined_cal'] or 0.0) if a['use_calibration'] else 0.0,
            'α_phase (cal)': (a['alpha_phase_cal'] or 0.0) if a['use_calibration'] else 0.0,
        })
    
    df = pd.DataFrame(summary_data)
    
    # Display options
    st.subheader("📊 Editable Summary Table")
    st.info("💡 Edit any cell directly in the table below, then click 'Save Changes' to update the database.")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        models = ['All'] + list(df['Model'].unique())
        selected_model = st.selectbox("Filter by Model", models)
    with col2:
        modes = ['All'] + list(df['Mode'].unique())
        selected_mode = st.selectbox("Filter by Mode", modes)
    with col3:
        cal_filter = st.selectbox("Filter by Calibration", ['All', 'Calibrated Only', 'Non-Calibrated Only'])
    
    # Apply filters
    filtered_df = df.copy()
    if selected_model != 'All':
        filtered_df = filtered_df[filtered_df['Model'] == selected_model]
    if selected_mode != 'All':
        filtered_df = filtered_df[filtered_df['Mode'] == selected_mode]
    if cal_filter == 'Calibrated Only':
        filtered_df = filtered_df[filtered_df['Calibrated'] == True]
    elif cal_filter == 'Non-Calibrated Only':
        filtered_df = filtered_df[filtered_df['Calibrated'] == False]
    
    # Column config for formatting
    column_config = {
        'ID': st.column_config.NumberColumn('ID', disabled=True),
        'Model': st.column_config.TextColumn('Model'),
        'Date': st.column_config.TextColumn('Date'),
        'T (°C)': st.column_config.NumberColumn('T (°C)', format="%.1f"),
        'Mode': st.column_config.SelectboxColumn('Mode', options=['Auto', 'Manual']),
        'r₁ (mm)': st.column_config.NumberColumn('r₁ (mm)', format="%.2f"),
        'r₂ (mm)': st.column_config.NumberColumn('r₂ (mm)', format="%.2f"),
        'A₁ (mW)': st.column_config.NumberColumn('A₁ (mW)', format="%.4f"),
        'A₂ (mW)': st.column_config.NumberColumn('A₂ (mW)', format="%.4f"),
        'Period (s)': st.column_config.NumberColumn('Period (s)', format="%.2f"),
        'f (Hz)': st.column_config.NumberColumn('f (Hz)', format="%.6f"),
        'ω (rad/s)': st.column_config.NumberColumn('ω (rad/s)', format="%.5f"),
        'Δt (s)': st.column_config.NumberColumn('Δt (s)', format="%.2f"),
        'φ (rad)': st.column_config.NumberColumn('φ (rad)', format="%.4f"),
        'ln term': st.column_config.NumberColumn('ln term', format="%.4f"),
        'α_comb (raw)': st.column_config.NumberColumn('α_comb (raw)', format="%.2e"),
        'α_phase (raw)': st.column_config.NumberColumn('α_phase (raw)', format="%.2e"),
        'Calibrated': st.column_config.CheckboxColumn('Cal'),
        'Lag (s)': st.column_config.NumberColumn('Lag (s)', format="%.1f"),
        'Net Δt (s)': st.column_config.NumberColumn('Net Δt (s)', format="%.2f"),
        'α_comb (cal)': st.column_config.NumberColumn('α_comb (cal)', format="%.2e"),
        'α_phase (cal)': st.column_config.NumberColumn('α_phase (cal)', format="%.2e"),
    }
    
    # Editable table
    edited_df = st.data_editor(
        filtered_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=400,
        num_rows="fixed",
        key="editable_summary_table"
    )
    
    st.caption(f"Showing {len(filtered_df)} of {len(df)} results")
    
    # Save changes button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            changes_made = 0
            for idx, row in edited_df.iterrows():
                original_row = filtered_df.loc[idx]
                
                # Check if row changed
                if not row.equals(original_row):
                    analysis_id = int(row['ID'])
                    update_data = {
                        'model_name': row['Model'],
                        'test_date': row['Date'],
                        'temperature_c': row['T (°C)'],
                        'analysis_mode': row['Mode'],
                        'r1_mm': row['r₁ (mm)'],
                        'r2_mm': row['r₂ (mm)'],
                        'amplitude_a1': row['A₁ (mW)'],
                        'amplitude_a2': row['A₂ (mW)'],
                        'period_t': row['Period (s)'],
                        'frequency_f': row['f (Hz)'],
                        'angular_freq_w': row['ω (rad/s)'],
                        'raw_lag_dt': row['Δt (s)'],
                        'raw_phase_phi': row['φ (rad)'],
                        'ln_term': row['ln term'],
                        'alpha_combined_raw': row['α_comb (raw)'],
                        'alpha_phase_raw': row['α_phase (raw)'],
                        'use_calibration': row['Calibrated'],
                        'system_lag': row['Lag (s)'],
                        'net_lag_dt': row['Net Δt (s)'],
                        'alpha_combined_cal': row['α_comb (cal)'],
                        'alpha_phase_cal': row['α_phase (cal)'],
                    }
                    
                    if db.update_analysis(analysis_id, update_data):
                        changes_made += 1
            
            if changes_made > 0:
                st.success(f"✅ Saved {changes_made} change(s)!")
                st.rerun()
            else:
                st.info("No changes detected")
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            data=csv_data,
            file_name="angstrom_results_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Create detailed Excel-like view
        detailed_data = []
        for a in full_analyses:
            if selected_model != 'All' and a['model_name'] != selected_model:
                continue
            detailed_data.append({
                'id': a['id'],
                'model_name': a['model_name'],
                'test_date': a['test_date'],
                'temperature_c': a.get('temperature_c'),
                'analysis_mode': a['analysis_mode'],
                'r1_mm': a['r1_mm'],
                'r2_mm': a['r2_mm'],
                'amplitude_a1': a['amplitude_a1'],
                'amplitude_a2': a['amplitude_a2'],
                'period_t': a['period_t'],
                'frequency_f': a['frequency_f'],
                'angular_freq_w': a['angular_freq_w'],
                'raw_lag_dt': a['raw_lag_dt'],
                'raw_phase_phi': a['raw_phase_phi'],
                'ln_term': a['ln_term'],
                'alpha_combined_raw': a['alpha_combined_raw'],
                'alpha_phase_raw': a['alpha_phase_raw'],
                'use_calibration': a['use_calibration'],
                'system_lag': a['system_lag'],
                'net_lag_dt': a['net_lag_dt'],
                'alpha_combined_cal': a['alpha_combined_cal'],
                'alpha_phase_cal': a['alpha_phase_cal'],
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv = detailed_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Data (CSV)",
            data=detailed_csv,
            file_name="angstrom_results_full.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Statistics")
    
    # Extract numeric alpha values for statistics
    alpha_comb_raw_vals = [a['alpha_combined_raw'] for a in full_analyses if a['alpha_combined_raw'] and a['alpha_combined_raw'] > 0]
    alpha_comb_cal_vals = [a['alpha_combined_cal'] for a in full_analyses if a['alpha_combined_cal'] and a['alpha_combined_cal'] > 0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(full_analyses))
    
    with col2:
        st.metric("Unique Models", len(set(a['model_name'] for a in full_analyses)))
    
    with col3:
        if alpha_comb_raw_vals:
            avg_alpha = np.mean(alpha_comb_raw_vals)
            st.metric("Avg α (raw)", format_scientific(avg_alpha))
        else:
            st.metric("Avg α (raw)", "-")
    
    with col4:
        if alpha_comb_cal_vals:
            avg_alpha_cal = np.mean(alpha_comb_cal_vals)
            st.metric("Avg α (cal)", format_scientific(avg_alpha_cal))
        else:
            st.metric("Avg α (cal)", "-")


def render_history_page():
    """Render the results history page."""
    st.header("📁 Analysis History")
    
    analyses = db.get_all_analyses()
    
    if not analyses:
        st.info("No analyses saved yet. Run an analysis and save it to see it here.")
        return
    
    # Summary table
    st.subheader(f"📊 All Analyses ({len(analyses)} total)")
    
    # Create DataFrame for display
    df = pd.DataFrame(analyses)
    df['use_calibration'] = df['use_calibration'].apply(lambda x: '✓' if x else '✗')
    df = df.rename(columns={
        'id': 'ID',
        'created_at': 'Date/Time',
        'model_name': 'Model',
        'test_date': 'Test Date',
        'analysis_mode': 'Mode',
        'r1_mm': 'r₁ (mm)',
        'r2_mm': 'r₂ (mm)',
        'alpha_combined_raw': 'α_comb (raw)',
        'alpha_combined_cal': 'α_comb (cal)',
        'alpha_phase_raw': 'α_phase (raw)',
        'alpha_phase_cal': 'α_phase (cal)',
        'use_calibration': 'Cal'
    })
    
    # Format scientific notation
    for col in ['α_comb (raw)', 'α_comb (cal)', 'α_phase (raw)', 'α_phase (cal)']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_scientific(x) if x and x > 0 else 'N/A')
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Detail view
    st.subheader("🔍 View Details")
    
    analysis_ids = [a['id'] for a in analyses]
    
    def format_analysis_label(aid):
        a = next((a for a in analyses if a['id'] == aid), None)
        if not a:
            return f"ID {aid}: Unknown"
        label = f"{a['model_name']} | {a['test_date']} | {a['analysis_mode']}"
        if a['use_calibration']:
            label += f" | Cal (Lag: {a['system_lag']}s)"
        else:
            label += " | No Cal"
        return label
    
    selected_id = st.selectbox(
        "Select analysis to view",
        options=analysis_ids,
        format_func=format_analysis_label
    )
    
    if selected_id:
        analysis = db.get_analysis_by_id(selected_id)
        
        if analysis:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Full Details:**")
                
                # Helper function to safely format values
                def safe_format(val, fmt=".2f", suffix=""):
                    if val is None or val == 0:
                        return "N/A"
                    try:
                        return f"{val:{fmt}}{suffix}"
                    except:
                        return str(val)
                
                # Create detailed view
                details = {
                    'Created': analysis.get('created_at', 'N/A'),
                    'Model': analysis.get('model_name', 'N/A'),
                    'Test Date': analysis.get('test_date', 'N/A'),
                    'Radii': f"r₁={analysis.get('r1_mm', 'N/A')}mm, r₂={analysis.get('r2_mm', 'N/A')}mm",
                    'Period T': safe_format(analysis.get('period_t'), ".2f", "s"),
                    'Frequency': safe_format(analysis.get('frequency_f'), ".5f", "Hz"),
                    'Raw Δt': safe_format(analysis.get('raw_lag_dt'), ".2f", "s"),
                    'α Combined (raw)': format_scientific(analysis.get('alpha_combined_raw')),
                    'α Phase (raw)': format_scientific(analysis.get('alpha_phase_raw')),
                }
                
                if analysis.get('use_calibration'):
                    details['System Lag'] = f"{analysis.get('system_lag', 0)}s"
                    details['Net Δt'] = safe_format(analysis.get('net_lag_dt'), ".2f", "s")
                    details['α Combined (cal)'] = format_scientific(analysis.get('alpha_combined_cal'))
                    details['α Phase (cal)'] = format_scientific(analysis.get('alpha_phase_cal'))
                
                for k, v in details.items():
                    st.write(f"**{k}:** {v}")
            
            with col2:
                # Show saved graph if available (now stored as JSON)
                graph_json = analysis.get('graph_json')
                if graph_json and isinstance(graph_json, str) and len(graph_json) > 10:
                    try:
                        saved_fig = json_to_fig(graph_json)
                        st.plotly_chart(saved_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display graph: {str(e)}")
                else:
                    st.info("No graph saved for this analysis (uploaded from image)")
                
                # Delete button
                if st.button("🗑️ Delete this analysis", type="secondary"):
                    if db.delete_analysis(selected_id):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")


def render_statistics_page():
    """Render the Statistics page with a flexible chart builder and fixed analytical sections."""
    st.title("📈 Statistics")
    st.caption("Build custom charts and explore distributions, trends, and correlations")

    # ── Load data ──────────────────────────────────────────────────────────────
    analyses = db.get_all_analyses()
    if not analyses:
        st.info("No analyses saved yet. Run some analyses first to see statistics.")
        return

    rows = []
    for a in analyses:
        period = a.get('period_t') or 0.0
        freq = a.get('frequency_f') or (1.0 / period if period > 0 else 0.0)
        rows.append({
            'id': a['id'],
            'Model': a.get('model_name') or 'Unknown',
            'Date': a.get('test_date') or '',
            'T (°C)': a.get('temperature_c') or 25.0,
            'Calibrated': bool(a.get('use_calibration')),
            'Cal Label': 'Calibrated' if a.get('use_calibration') else 'Uncalibrated',
            'Analysis Mode': a.get('analysis_mode') or 'Auto',
            'System Lag (s)': a.get('system_lag') or 0.0,
            'r₁ (mm)': a.get('r1_mm') or 0.0,
            'r₂ (mm)': a.get('r2_mm') or 0.0,
            'A₁ (mW)': a.get('amplitude_a1') or 0.0,
            'A₂ (mW)': a.get('amplitude_a2') or 0.0,
            'Period (s)': period,
            'Frequency (Hz)': freq,
            'Raw Δt (s)': a.get('raw_lag_dt') or 0.0,
            'Raw φ (rad)': a.get('raw_phase_phi') or 0.0,
            'ln term': a.get('ln_term') or 0.0,
            'α_comb_raw': a.get('alpha_combined_raw') or 0.0,
            'α_phase_raw': a.get('alpha_phase_raw') or 0.0,
            'Net Δt (s)': a.get('net_lag_dt') or 0.0,
            'α_comb_cal': a.get('alpha_combined_cal') or 0.0,
            'α_phase_cal': a.get('alpha_phase_cal') or 0.0,
        })
    df_all = pd.DataFrame(rows)

    # ── Derived columns ────────────────────────────────────────────────────────
    df_all['A₁/A₂ ratio'] = df_all.apply(
        lambda r: r['A₁ (mW)'] / r['A₂ (mW)'] if r['A₂ (mW)'] > 0 else np.nan, axis=1)
    df_all['α_phase/α_comb'] = df_all.apply(
        lambda r: r['α_phase_raw'] / r['α_comb_raw']
        if r['α_comb_raw'] > 0 and r['α_phase_raw'] > 0 else np.nan, axis=1)

    # Period exact label — rounded to nearest 10s for cleaner grouping
    def period_label(p):
        if p <= 0:
            return 'Unknown'
        rounded = round(p / 10) * 10
        return f"{rounded:.0f} s"

    def period_band(p):
        if p <= 0:
            return 'Unknown'
        elif p < 200:
            return '< 200 s'
        elif p < 400:
            return '200–400 s'
        elif p < 700:
            return '400–700 s'
        elif p < 1200:
            return '700–1200 s'
        else:
            return '> 1200 s'

    def temp_band(t):
        if t < 50:
            return '< 50 °C'
        elif t < 100:
            return '50–100 °C'
        elif t < 200:
            return '100–200 °C'
        elif t < 300:
            return '200–300 °C'
        else:
            return '> 300 °C'

    df_all['Period (exact)'] = df_all['Period (s)'].apply(period_label)
    df_all['Period Band'] = df_all['Period (s)'].apply(period_band)
    df_all['Temp Band'] = df_all['T (°C)'].apply(temp_band)
    df_all['Model + Period'] = df_all['Model'] + ' / ' + df_all['Period (exact)']

    band_order = ['< 200 s', '200–400 s', '400–700 s', '700–1200 s', '> 1200 s']
    temp_band_order = ['< 50 °C', '50–100 °C', '100–200 °C', '200–300 °C', '> 300 °C']

    # ── Sidebar global filters ─────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Statistics Filters")
        all_models = sorted(df_all['Model'].unique().tolist())
        sel_models = st.multiselect("Models", all_models, default=all_models, key='stats_models')
        cal_choice = st.radio("Calibration", ["All", "Calibrated", "Uncalibrated"], key='stats_cal')
        all_periods = sorted(df_all['Period (exact)'].unique().tolist(),
                             key=lambda x: float(x.replace(' s', '')) if x != 'Unknown' else 0)
        sel_periods = st.multiselect("Periods", all_periods, default=all_periods, key='stats_periods')

    df = df_all[df_all['Model'].isin(sel_models)].copy()
    if sel_periods:
        df = df[df['Period (exact)'].isin(sel_periods)]
    if cal_choice == "Calibrated":
        df = df[df['Calibrated']]
    elif cal_choice == "Uncalibrated":
        df = df[~df['Calibrated']]

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    st.caption(f"Showing **{len(df)}** of {len(df_all)} analyses")

    # ══════════════════════════════════════════════════════════════════════════
    # ── CHART BUILDER ─────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    st.header("🛠️ Chart Builder")

    # Metric definitions: label → (df column, y-axis label, filter_nonzero)
    METRICS = {
        'α Combined (raw)':   ('α_comb_raw',   'α (m²/s)', True),
        'α Phase (raw)':      ('α_phase_raw',  'α (m²/s)', True),
        'α Combined (cal)':   ('α_comb_cal',   'α (m²/s)', True),
        'α Phase (cal)':      ('α_phase_cal',  'α (m²/s)', True),
        'Raw Time Lag Δt':    ('Raw Δt (s)',   'Δt (s)',   False),
        'Raw Phase φ':        ('Raw φ (rad)',  'φ (rad)',  False),
        'ln term':            ('ln term',      'ln(A₁√r₁/A₂√r₂)', False),
        'A₁ (mW)':            ('A₁ (mW)',      'mW',       False),
        'A₂ (mW)':            ('A₂ (mW)',      'mW',       False),
        'A₁/A₂ ratio':        ('A₁/A₂ ratio',  'A₁/A₂',   False),
        'α_phase / α_comb':   ('α_phase/α_comb', 'ratio',  False),
        'Net Δt (s)':         ('Net Δt (s)',   's',        False),
        'Period (s)':         ('Period (s)',   's',        True),
        'Temperature (°C)':   ('T (°C)',       '°C',       False),
    }

    # Group-by options: label → (df column, ordered categories or None)
    GROUP_OPTIONS = {
        'Model':              ('Model',         None),
        'Period (exact)':     ('Period (exact)', None),   # sorted numerically below
        'Period Band':        ('Period Band',    band_order),
        'Temperature Band':   ('Temp Band',      temp_band_order),
        'Calibration':        ('Cal Label',      ['Calibrated', 'Uncalibrated']),
        'Analysis Mode':      ('Analysis Mode',  None),
        'Model + Period':     ('Model + Period', None),
    }

    # X axis options for scatter
    X_AXIS_OPTIONS = {
        'Period (s)':       'Period (s)',
        'Frequency (Hz)':   'Frequency (Hz)',
        'Temperature (°C)': 'T (°C)',
        'Raw Δt (s)':       'Raw Δt (s)',
        'ln term':          'ln term',
        'A₁/A₂ ratio':      'A₁/A₂ ratio',
        'System Lag (s)':   'System Lag (s)',
        'A₁ (mW)':          'A₁ (mW)',
        'A₂ (mW)':          'A₂ (mW)',
    }

    CHART_TYPES = ['Box', 'Violin', 'Bar (mean ± std)', 'Scatter']
    PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
               '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#16a085']

    # Controls row 1: Y axis + Chart type
    c1, c2 = st.columns(2)
    with c1:
        y_label = st.selectbox("Y axis (metric)", list(METRICS.keys()), key='cb_y')
    with c2:
        chart_type = st.selectbox("Chart type", CHART_TYPES, key='cb_type')

    y_col, y_unit, filter_nonzero = METRICS[y_label]

    # Controls row 2: Group by / X axis + Color by
    c3, c4 = st.columns(2)
    if chart_type == 'Scatter':
        with c3:
            x_label = st.selectbox("X axis", list(X_AXIS_OPTIONS.keys()), key='cb_x')
        with c4:
            color_label = st.selectbox(
                "Color by", ['None'] + list(GROUP_OPTIONS.keys()), key='cb_color')
        group_label = None
    else:
        with c3:
            group_label = st.selectbox("Group by", list(GROUP_OPTIONS.keys()), key='cb_group')
        with c4:
            split_label = st.selectbox(
                "Split / color by (optional)", ['None'] + list(GROUP_OPTIONS.keys()),
                key='cb_split')
        color_label = split_label if split_label != 'None' else None

    # Prepare data
    df_cb = df.copy()
    if filter_nonzero:
        df_cb = df_cb[df_cb[y_col] > 0]

    if df_cb.empty:
        st.info(f"No valid (non-zero) data for **{y_label}** in the current filter selection.")
    else:
        def get_cat_order(glabel):
            gcol, gorder = GROUP_OPTIONS[glabel]
            if gorder:
                return gcol, [c for c in gorder if c in df_cb[gcol].values]
            elif glabel == 'Period (exact)':
                # Sort numerically
                unique_vals = df_cb[gcol].unique().tolist()
                return gcol, sorted(
                    [v for v in unique_vals if v != 'Unknown'],
                    key=lambda x: float(x.replace(' s', ''))
                ) + (['Unknown'] if 'Unknown' in unique_vals else [])
            elif glabel == 'Model + Period':
                # Sort by period first, then model
                unique_vals = df_cb[gcol].unique().tolist()
                def sort_key(v):
                    parts = v.split(' / ')
                    try:
                        return (float(parts[1].replace(' s', '')), parts[0])
                    except Exception:
                        return (0, v)
                return gcol, sorted(unique_vals, key=sort_key)
            else:
                return gcol, sorted(df_cb[gcol].unique().tolist())

        if chart_type == 'Scatter':
            x_col = X_AXIS_OPTIONS[x_label]
            fig_cb = go.Figure()
            if color_label and color_label != 'None':
                color_col, color_order = get_cat_order(color_label)
                for i, cat in enumerate(color_order):
                    sub = df_cb[df_cb[color_col] == cat]
                    if sub.empty:
                        continue
                    hover_text = sub.apply(
                        lambda r: f"{r['Model']} | {r['Period (exact)']} | {r['Date']}", axis=1)
                    fig_cb.add_trace(go.Scatter(
                        x=sub[x_col], y=sub[y_col],
                        mode='markers',
                        name=str(cat),
                        text=hover_text,
                        hovertemplate=f'%{{text}}<br>{x_label}=%{{x}}<br>{y_label}=%{{y:.3g}}<extra>%{{fullData.name}}</extra>',
                        marker=dict(size=10, color=PALETTE[i % len(PALETTE)], opacity=0.85),
                    ))
                    # Trend line if ≥ 3 numeric X points
                    if len(sub) >= 3 and pd.api.types.is_numeric_dtype(sub[x_col]):
                        try:
                            coeffs = np.polyfit(sub[x_col], sub[y_col], 1)
                            xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 80)
                            fig_cb.add_trace(go.Scatter(
                                x=xr, y=np.polyval(coeffs, xr),
                                mode='lines', name=f'{cat} trend',
                                line=dict(color=PALETTE[i % len(PALETTE)], dash='dash', width=1),
                                showlegend=False,
                            ))
                        except Exception:
                            pass
            else:
                hover_text = df_cb.apply(
                    lambda r: f"{r['Model']} | {r['Period (exact)']} | {r['Date']}", axis=1)
                fig_cb.add_trace(go.Scatter(
                    x=df_cb[x_col], y=df_cb[y_col],
                    mode='markers',
                    text=hover_text,
                    hovertemplate=f'%{{text}}<br>{x_label}=%{{x}}<br>{y_label}=%{{y:.3g}}<extra></extra>',
                    marker=dict(size=10, color='#3498db', opacity=0.85),
                ))
            fig_cb.update_layout(
                title=f"{y_label} vs {x_label}",
                xaxis_title=x_label, yaxis_title=y_unit,
                height=480, hovermode='closest',
            )

        else:
            # Box / Violin / Bar
            group_col, cat_order = get_cat_order(group_label)
            if color_label and color_label != 'None':
                color_col, color_order = get_cat_order(color_label)
            else:
                color_col, color_order = None, None

            fig_cb = go.Figure()

            if color_col:
                # Grouped chart: for each color category, one trace per group value
                for ci, ccat in enumerate(color_order):
                    sub_c = df_cb[df_cb[color_col] == ccat]
                    y_vals_by_group = [sub_c[sub_c[group_col] == g][y_col].tolist() for g in cat_order]

                    if chart_type == 'Box':
                        for gi, (g, yv) in enumerate(zip(cat_order, y_vals_by_group)):
                            fig_cb.add_trace(go.Box(
                                y=yv, name=str(ccat), x=[str(g)] * len(yv),
                                boxpoints='all', jitter=0.4, pointpos=0,
                                marker_color=PALETTE[ci % len(PALETTE)],
                                legendgroup=str(ccat),
                                showlegend=(gi == 0),
                            ))
                    elif chart_type == 'Violin':
                        for gi, (g, yv) in enumerate(zip(cat_order, y_vals_by_group)):
                            fig_cb.add_trace(go.Violin(
                                y=yv, name=str(ccat), x=[str(g)] * len(yv),
                                box_visible=True, meanline_visible=True, points='all',
                                line_color=PALETTE[ci % len(PALETTE)],
                                legendgroup=str(ccat),
                                showlegend=(gi == 0),
                            ))
                    else:  # Bar mean±std
                        means = [np.mean(yv) if yv else 0 for yv in y_vals_by_group]
                        stds  = [np.std(yv)  if len(yv) > 1 else 0 for yv in y_vals_by_group]
                        fig_cb.add_trace(go.Bar(
                            x=[str(g) for g in cat_order], y=means,
                            error_y=dict(type='data', array=stds, visible=True),
                            name=str(ccat),
                            marker_color=PALETTE[ci % len(PALETTE)],
                        ))
                fig_cb.update_layout(boxmode='group', violinmode='group', barmode='group')
            else:
                # Single dimension
                for i, cat in enumerate(cat_order):
                    sub_c = df_cb[df_cb[group_col] == cat]
                    yv = sub_c[y_col].tolist()
                    color = PALETTE[i % len(PALETTE)]
                    hover = sub_c.apply(
                        lambda r: f"{r['Model']} | {r['Period (exact)']} | {r['Date']}", axis=1).tolist()

                    if chart_type == 'Box':
                        fig_cb.add_trace(go.Box(
                            y=yv, name=str(cat),
                            boxpoints='all', jitter=0.4, pointpos=0,
                            marker=dict(size=7, color=color),
                            line_color=color,
                            text=hover,
                            hovertemplate='%{text}<br>%{y:.3g}<extra>%{fullData.name}</extra>',
                        ))
                    elif chart_type == 'Violin':
                        fig_cb.add_trace(go.Violin(
                            y=yv, name=str(cat),
                            box_visible=True, meanline_visible=True, points='all',
                            line_color=color,
                            text=hover,
                            hovertemplate='%{text}<br>%{y:.3g}<extra>%{fullData.name}</extra>',
                        ))
                    else:  # Bar mean±std
                        mean_v = np.mean(yv) if yv else 0
                        std_v  = np.std(yv)  if len(yv) > 1 else 0
                        fig_cb.add_trace(go.Bar(
                            x=[str(cat)], y=[mean_v],
                            error_y=dict(type='data', array=[std_v], visible=True),
                            name=str(cat),
                            marker_color=color,
                            text=[f"n={len(yv)}"],
                            textposition='outside',
                        ))

            title_str = f"{y_label}  ·  grouped by {group_label}"
            if color_label and color_label != 'None':
                title_str += f"  ·  colored by {color_label}"
            fig_cb.update_layout(
                title=title_str,
                yaxis_title=y_unit,
                height=480,
                showlegend=True,
            )

        st.plotly_chart(fig_cb, use_container_width=True)

        # Quick stats under the chart
        with st.expander("📊 Quick statistics for this chart"):
            if chart_type != 'Scatter' and group_label:
                group_col, cat_order = get_cat_order(group_label)
                stat_rows_cb = []
                for cat in cat_order:
                    yv = df_cb[df_cb[group_col] == cat][y_col].dropna()
                    if len(yv) == 0:
                        continue
                    stat_rows_cb.append({
                        group_label: cat, 'N': len(yv),
                        'Mean': yv.mean(), 'Std': yv.std(),
                        'Min': yv.min(), 'Median': yv.median(), 'Max': yv.max(),
                    })
                if stat_rows_cb:
                    st.dataframe(pd.DataFrame(stat_rows_cb), hide_index=True, use_container_width=True)
            else:
                yv = df_cb[y_col].dropna()
                st.dataframe(pd.DataFrame([{
                    'N': len(yv), 'Mean': yv.mean(), 'Std': yv.std(),
                    'Min': yv.min(), 'Median': yv.median(), 'Max': yv.max(),
                }]), hide_index=True, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fixed analytical sections ─────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # ── Section 2: Alpha vs Period (scatter + trend per model) ────────────────
    st.header("2. α vs Period — by Model")
    st.caption("Ideal: α constant across periods. Drift indicates frequency-dependent artifacts.")

    alpha_cols = {'α Combined (raw)': 'α_comb_raw', 'α Phase (raw)': 'α_phase_raw'}
    if df['Calibrated'].any():
        alpha_cols['α Combined (cal)'] = 'α_comb_cal'
        alpha_cols['α Phase (cal)'] = 'α_phase_cal'

    sel_alphas = st.multiselect("Series", list(alpha_cols.keys()),
                                default=list(alpha_cols.keys())[:2], key='stats_alpha_series')

    fig_avp = go.Figure()
    pal2 = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, label in enumerate(sel_alphas):
        col_key = alpha_cols[label]
        sub = df[df[col_key] > 0][['Period (s)', col_key, 'Model', 'Period (exact)']].dropna()
        if sub.empty:
            continue
        color = pal2[i % len(pal2)]
        hover = sub.apply(lambda r: f"{r['Model']} | {r['Period (exact)']}", axis=1)
        fig_avp.add_trace(go.Scatter(
            x=sub['Period (s)'], y=sub[col_key], mode='markers', name=label,
            text=hover,
            hovertemplate='%{text}<br>T=%{x:.1f}s<br>α=%{y:.2e}<extra>' + label + '</extra>',
            marker=dict(size=9, color=color, opacity=0.8),
        ))
        if len(sub) >= 3:
            try:
                log_x = np.log(sub['Period (s)'].clip(lower=1))
                coeffs = np.polyfit(log_x, sub[col_key], 1)
                xr = np.linspace(sub['Period (s)'].min(), sub['Period (s)'].max(), 100)
                fig_avp.add_trace(go.Scatter(
                    x=xr, y=np.polyval(coeffs, np.log(xr.clip(min=1))),
                    mode='lines', name=f'{label} trend',
                    line=dict(color=color, dash='dash', width=1), showlegend=True,
                ))
            except Exception:
                pass
    fig_avp.update_layout(title="α vs Oscillation Period", xaxis_title="Period T (s)",
                          yaxis_title="α (m²/s)", height=420, hovermode='closest')
    st.plotly_chart(fig_avp, use_container_width=True)

    st.divider()

    # ── Section 3: Heat-loss indicators ──────────────────────────────────────
    st.header("3. Amplitude Attenuation & Heat-Loss Indicators")
    st.caption("Large A₁/A₂ or ln-term → strong heat loss. α_phase/α_comb → 1 in adiabatic case.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sub_ln = df[df['ln term'] != 0]
        fig_ln = go.Figure()
        for i, cat in enumerate(sorted(sub_ln['Model'].unique())):
            fig_ln.add_trace(go.Box(y=sub_ln[sub_ln['Model'] == cat]['ln term'].tolist(),
                                    name=cat, boxpoints='all', jitter=0.4, pointpos=0,
                                    marker_color=PALETTE[i % len(PALETTE)]))
        fig_ln.update_layout(title="ln term by Model", yaxis_title="ln(A₁√r₁/A₂√r₂)",
                              height=350, showlegend=False)
        st.plotly_chart(fig_ln, use_container_width=True)

    with col_b:
        sub_r = df[df['A₁/A₂ ratio'].notna() & (df['A₁/A₂ ratio'] > 0)]
        fig_r = go.Figure()
        for i, cat in enumerate(sorted(sub_r['Model'].unique())):
            fig_r.add_trace(go.Box(y=sub_r[sub_r['Model'] == cat]['A₁/A₂ ratio'].tolist(),
                                   name=cat, boxpoints='all', jitter=0.4, pointpos=0,
                                   marker_color=PALETTE[i % len(PALETTE)]))
        fig_r.update_layout(title="Amplitude Ratio A₁/A₂ by Model", yaxis_title="A₁/A₂",
                             height=350, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_c:
        sub_r2 = df[df['α_phase/α_comb'].notna()]
        fig_r2 = go.Figure()
        for i, cat in enumerate(sorted(sub_r2['Model'].unique())):
            fig_r2.add_trace(go.Box(y=sub_r2[sub_r2['Model'] == cat]['α_phase/α_comb'].tolist(),
                                    name=cat, boxpoints='all', jitter=0.4, pointpos=0,
                                    marker_color=PALETTE[i % len(PALETTE)]))
        fig_r2.add_hline(y=1, line_dash='dash', line_color='gray',
                          annotation_text='Ideal (adiabatic)')
        fig_r2.update_layout(title="α_phase / α_comb", yaxis_title="ratio",
                              height=350, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    # ── Section 4: Phase lag analysis ─────────────────────────────────────────
    st.header("4. Phase Lag Analysis")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        sub_phi = df[df['Raw φ (rad)'] > 0]
        fig_phi = go.Figure()
        for i, model in enumerate(sorted(sub_phi['Model'].unique())):
            sm = sub_phi[sub_phi['Model'] == model]
            fig_phi.add_trace(go.Scatter(
                x=sm['Period (s)'], y=sm['Raw φ (rad)'],
                mode='markers+lines', name=model, marker=dict(size=8, color=PALETTE[i % len(PALETTE)]),
                hovertemplate='%{x:.1f}s → φ=%{y:.3f} rad<extra>' + model + '</extra>'))
        fig_phi.update_layout(title="Raw Phase φ vs Period", xaxis_title="Period (s)",
                               yaxis_title="φ (rad)", height=360)
        st.plotly_chart(fig_phi, use_container_width=True)
    with col_p2:
        sub_dt = df[df['Raw Δt (s)'] > 0]
        fig_dt = go.Figure()
        for i, model in enumerate(sorted(sub_dt['Model'].unique())):
            sm = sub_dt[sub_dt['Model'] == model]
            fig_dt.add_trace(go.Scatter(
                x=sm['Period (s)'], y=sm['Raw Δt (s)'],
                mode='markers+lines', name=model, marker=dict(size=8, color=PALETTE[i % len(PALETTE)]),
                hovertemplate='%{x:.1f}s → Δt=%{y:.1f}s<extra>' + model + '</extra>'))
        fig_dt.update_layout(title="Raw Time Lag Δt vs Period", xaxis_title="Period (s)",
                              yaxis_title="Δt (s)", height=360)
        st.plotly_chart(fig_dt, use_container_width=True)

    st.divider()

    # ── Section 5: Temperature dependence (only if T varies) ──────────────────
    if df['T (°C)'].nunique() > 1:
        st.header("5. Temperature Dependence")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            sub_t = df[df['α_comb_raw'] > 0]
            fig_temp = go.Figure()
            for i, model in enumerate(sorted(sub_t['Model'].unique())):
                sm = sub_t[sub_t['Model'] == model]
                fig_temp.add_trace(go.Scatter(
                    x=sm['T (°C)'], y=sm['α_comb_raw'], mode='markers', name=model,
                    marker=dict(size=10, color=PALETTE[i % len(PALETTE)]),
                    hovertemplate='T=%{x}°C → α=%{y:.2e}<extra>' + model + '</extra>'))
            fig_temp.update_layout(title="α Combined (raw) vs T", xaxis_title="T (°C)",
                                    yaxis_title="α (m²/s)", height=360)
            st.plotly_chart(fig_temp, use_container_width=True)
        with col_t2:
            sub_t2 = df[df['α_phase_raw'] > 0]
            fig_temp2 = go.Figure()
            for i, model in enumerate(sorted(sub_t2['Model'].unique())):
                sm = sub_t2[sub_t2['Model'] == model]
                fig_temp2.add_trace(go.Scatter(
                    x=sm['T (°C)'], y=sm['α_phase_raw'], mode='markers', name=model,
                    marker=dict(size=10, color=PALETTE[i % len(PALETTE)]),
                    hovertemplate='T=%{x}°C → α=%{y:.2e}<extra>' + model + '</extra>'))
            fig_temp2.update_layout(title="α Phase (raw) vs T", xaxis_title="T (°C)",
                                     yaxis_title="α (m²/s)", height=360)
            st.plotly_chart(fig_temp2, use_container_width=True)
        st.divider()

    # ── Section 6: Correlation heatmap ────────────────────────────────────────
    st.header("6. Correlation Matrix")
    st.caption("Pearson r between key numeric parameters")

    numeric_cols = ['A₁ (mW)', 'A₂ (mW)', 'A₁/A₂ ratio', 'Period (s)',
                    'Raw Δt (s)', 'Raw φ (rad)', 'ln term',
                    'α_comb_raw', 'α_phase_raw', 'T (°C)']
    if df['Calibrated'].any():
        numeric_cols += ['α_comb_cal', 'α_phase_cal', 'Net Δt (s)']

    corr_df = df[numeric_cols].replace(0, np.nan).dropna()
    corr_df = corr_df.loc[:, corr_df.std() > 0]

    if len(corr_df) >= 3 and corr_df.shape[1] >= 2:
        cm = corr_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=np.round(cm.values, 2), texttemplate='%{text}',
            hovertemplate='%{y} × %{x}<br>r = %{z:.2f}<extra></extra>',
        ))
        fig_corr.update_layout(title="Correlation Matrix (Pearson r)", height=500,
                                xaxis=dict(tickangle=-35))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need ≥ 3 analyses with valid values for correlation matrix.")

    st.divider()

    # ── Section 7: Summary statistics table ───────────────────────────────────
    st.header("7. Summary Statistics by Model")

    stat_cols_map = {
        'α_comb_raw': 'α comb (raw)', 'α_phase_raw': 'α phase (raw)',
        'α_comb_cal': 'α comb (cal)', 'α_phase_cal': 'α phase (cal)',
        'Raw φ (rad)': 'φ (rad)', 'A₁/A₂ ratio': 'A₁/A₂',
    }
    stat_rows = []
    for model, grp in df.groupby('Model'):
        row = {'Model': model, 'N': len(grp)}
        for col, lbl in stat_cols_map.items():
            vals = grp[col].replace(0, np.nan).dropna()
            row[f'{lbl} mean'] = vals.mean() if len(vals) else np.nan
            row[f'{lbl} std']  = vals.std()  if len(vals) else np.nan
            row[f'{lbl} min']  = vals.min()  if len(vals) else np.nan
            row[f'{lbl} max']  = vals.max()  if len(vals) else np.nan
        stat_rows.append(row)

    stat_df = pd.DataFrame(stat_rows)
    st.dataframe(stat_df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download Statistics (CSV)", data=stat_df.to_csv(index=False),
                       file_name="angstrom_statistics.csv", mime="text/csv")


def render_theory_page():
    """Render the Theory & Mathematical Evolution page."""
    st.title("📐 Theory & Mathematical Evolution")
    st.caption("Comprehensive Mathematical Evolution of Radial Thermal Diffusivity")

    # ── Literature Review ──────────────────────────────────────────────────────
    st.header("Literature Review: The Evolution of Periodic Heat Methods")
    st.markdown(
        """
The determination of thermal diffusivity through periodic heat flow, often referred to as the
**"wave method"**, has its roots in the 19th century and has evolved significantly to accommodate
modern materials and cylindrical geometries.
"""
    )

    with st.expander("📜 The Ångström Method (1861) **[1]**", expanded=True):
        st.markdown(
            """
The foundational work for periodic thermal analysis was established by **Anders Jonas Ångström**
in his 1861 paper, *"Neue Methode das Wärmeleitungsvermögen der Körper zu bestimmen"* **[1]**.

#### Historical Context

Prior to Ångström's contribution, thermal diffusivity was measured through *static* methods —
imposing a steady-state temperature gradient across a sample and measuring the resulting heat flux.
These approaches required precise knowledge of boundary conditions and suffered systematic errors
from contact resistances and unmeasured environmental heat losses. Ångström's insight was to
exploit *dynamic* periodic oscillations, making the result self-referencing and independent of the
absolute magnitude of the heat input.

#### Experimental Setup

Ångström's apparatus consisted of a long metallic rod, one end of which was alternately brought
into contact with a hot and a cold reservoir to produce a near-sinusoidal temperature oscillation
at angular frequency ω = 2π/T. Two thermometers placed at axial positions x₁ and x₂ recorded
the resulting oscillations at each location:

T(xᵢ, t) = T∞ + Aᵢ cos(ωt + ψᵢ)

where Aᵢ is the local amplitude and ψᵢ the phase. The amplitude ratio A₁/A₂ and the phase
difference φ = ψ₁ − ψ₂ are the two key observables.

#### Governing Equation for a Lossy Rod

For a thin rod exchanging heat with the environment through a linear (Newtonian) loss term,
the one-dimensional temperature field satisfies:
"""
        )
        st.latex(
            r"\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}"
            r"- \mu^2(T - T_\infty)"
        )
        st.markdown(
            r"""
where μ² = hP/kS encodes the geometry (h = surface heat-transfer coefficient, P = rod
perimeter, k = thermal conductivity, S = cross-sectional area). Substituting the steady-state
harmonic ansatz T − T∞ = θ(x)eⁱωᵗ and solving the resulting ODE yields a travelling wave:
"""
        )
        st.latex(
            r"T(x,t) - T_\infty = A_0\, e^{-m_A x}\cos(\omega t - m_\varphi x)"
        )
        st.markdown(
            r"""
where m_A is the **amplitude attenuation coefficient** and m_φ the **spatial phase coefficient**.

#### Ångström's Key Derivation

Comparing the wave at x₁ and x₂ (separation Δx = x₂ − x₁):
"""
        )
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"m_A = \frac{1}{\Delta x}\ln\!\frac{A_1}{A_2}")
        with col2:
            st.latex(r"m_\varphi = \frac{\varphi}{\Delta x}")
        st.markdown(
            r"""
The complex propagation constant λ = m_A + im_φ satisfies λ² = μ² + iω/α. Equating the
imaginary parts of both sides:
"""
        )
        st.latex(r"\frac{\omega}{\alpha} = 2\,m_A m_\varphi")
        st.markdown("Substituting the measured quantities gives **Ångström's celebrated formula**:")
        st.latex(
            r"\alpha = \frac{\omega\,(\Delta x)^2}{2\,\varphi\,\ln\!\left(\dfrac{A_1}{A_2}\right)}"
        )
        st.markdown(
            r"""
The heat-loss term μ² cancels entirely from this combined expression — it appears only in the
real part (m_A² − m_φ²), leaving α determined by amplitude and phase alone, with no knowledge
of absolute heat flux required.

#### Significance & Limitations

The elegance of the method lies in its self-referencing nature: systematic errors in absolute
power measurement are irrelevant. This made the technique far more reliable than any steady-state
method available in the 19th century.

However, the derivation assumes a **one-dimensional, semi-infinite rod** with uniform cross-section
and isotropic properties. Extension to disc, cylindrical, and annular geometries — necessary for
bulk samples and modern materials science — required the subsequent Bessel-function formulation
described in the Mathematical Derivation section. Notably, the linear formula above becomes the
radial formula upon replacing ln(A₁/A₂) with ln(A₁√r₁ / A₂√r₂) — the geometric correction
accounting for the cylindrical spreading of the wavefront.
"""
        )

    with st.expander("🔬 Cowan's Advancements (1961–1963) **[2][3][4]**", expanded=True):
        st.markdown(
            """
In the mid-20th century, **R. D. Cowan** systematically extended periodic and pulse heat methods
to address two central challenges: **high-temperature radiative losses** and **non-rod sample
geometries** **[2][3]**.

#### The Flash Method and Heat-Loss Corrections (1961)

Parker et al. **[4]** introduced the laser flash method in 1961, in which a short pulse heats
the front face of a disc and the rear-face temperature rise is monitored. For an ideal adiabatic
disc of thickness L the thermal diffusivity follows directly from the half-rise time t½:
"""
        )
        st.latex(r"\alpha = \frac{0.1388\, L^2}{t_{1/2}}")
        st.markdown(
            r"""
Cowan **[2]** recognised that at high temperatures (> 500 °C) radiative cooling from the disc
surfaces significantly distorts the temperature-rise curve, causing the simple formula above to
overestimate α. He derived a correction factor ξ based on the dimensionless ratio of the
temperature at five half-rise times to the peak temperature, T(5t½)/T_max:
"""
        )
        st.latex(r"\alpha_\text{corrected} = \frac{0.1388\, L^2}{t_{1/2}\cdot \xi\!\left(\frac{T(5t_{1/2})}{T_\text{max}}\right)}")
        st.markdown(
            r"""
This was the first systematic treatment of how surface emissivity and sample aspect ratio distort
flash measurements — critical for ceramics, refractory metals, and carbides tested above 1000 °C.

#### Phase Lag as a Robust Observable under Radiation (1963)

In his 1963 extension **[3]**, Cowan considered a plate driven sinusoidally from one face with
Newtonian heat losses from both surfaces. His central finding: the **phase lag** φ between the
front and rear oscillations is far less sensitive to surface heat losses than the **amplitude
ratio** A₁/A₂.

For an adiabatic plate of thickness L, the phase-only formula gives:
"""
        )
        st.latex(r"\alpha_\text{phase} = \frac{\omega\, L^2}{\varphi^2}")
        st.markdown(
            r"""
Under non-adiabatic conditions, the amplitude decays rapidly due to surface emission — but
Cowan showed that φ shifts by only a **second-order correction** in the heat-loss parameter.
Quantitatively: for materials with emissivity up to ε ≈ 0.9 tested below 2000 °C, the
phase-derived α carries an error below ~5 %, while amplitude-only values can deviate by 30–50 %.

This result provides the theoretical justification for treating α_phase as the preferred
observable in demanding experimental conditions, and directly motivates the phase formula used
in this application:
"""
        )
        st.latex(
            r"\alpha_\text{phase} = \frac{\omega\,(\Delta r)^2}{2\,\varphi^2}"
        )
        st.markdown(
            r"""
#### Bridge to Radial Geometry

Cowan also demonstrated that the wave-based framework generalises beyond linear and plate
geometries. A periodic heat source on the axis of a cylindrical sample produces a **radially
propagating thermal wave** governed by the modified Bessel equation of order zero:
"""
        )
        st.latex(
            r"\frac{d^2\theta}{dr^2} + \frac{1}{r}\frac{d\theta}{dr} - \lambda^2\theta = 0"
            r"\qquad \lambda = \sqrt{\frac{i\omega}{\alpha} + \mu^2}"
        )
        st.markdown(
            r"""
The asymptotic expansion K₀(rλ) ≈ √(π/2rλ) · e^(−rλ) for large |rλ| recovers Ångström's
linear result, with the geometric factor √(r₁/r₂) arising from the cylindrical spreading of
the wavefront. This factor appears explicitly in the combined diffusivity formula implemented
in this application:
"""
        )
        st.latex(
            r"\alpha_\text{combined} = \frac{\omega\,(\Delta r)^2}"
            r"{2\,\varphi\,\ln\!\left(\dfrac{A_1\sqrt{r_1}}{A_2\sqrt{r_2}}\right)}"
        )

    # ── Bibliography ──────────────────────────────────────────────────────────
    st.subheader("Bibliography")
    st.markdown(
        """
**[1]** Ångström, A. J. (1861). Neue Methode das Wärmeleitungsvermögen der Körper zu bestimmen.
*Annalen der Physik und Chemie*, 190(12), 513–530.
https://doi.org/10.1002/andp.18611901205

**[2]** Cowan, R. D. (1963). Pulse method of measuring thermal diffusivity at high temperatures.
*Journal of Applied Physics*, 34(4), 926–927.
https://doi.org/10.1063/1.1729564

**[3]** Cowan, R. D. (1961). Proposed method of measuring thermal diffusivity at high temperatures.
*Journal of Applied Physics*, 32(7), 1363–1370.
https://doi.org/10.1063/1.1736247

**[4]** Parker, W. J., Jenkins, R. J., Butler, C. P., & Abbott, G. L. (1961). Flash method of
determining thermal diffusivity, heat capacity, and thermal conductivity.
*Journal of Applied Physics*, 32(9), 1679–1684.
https://doi.org/10.1063/1.1728417
"""
    )

    st.divider()

    # ── Mathematical Derivation ────────────────────────────────────────────────
    st.header("Mathematical Derivation")

    # Step 1
    st.subheader("1. The Fundamental Heat Diffusion Equation")
    st.markdown(
        "The derivation begins with the general energy balance in a solid, isotropic medium. "
        "Based on Fourier's Law and conservation of energy, the temperature field *T* is governed by:"
    )
    st.latex(r"\frac{\partial T}{\partial t} = \alpha \nabla^2 T")
    st.markdown(
        "In a cylindrical system where a linear heat source is applied along the *z*-axis, we assume "
        "angular (θ) and longitudinal (*z*) symmetry. This reduces the Laplacian to its radial component:"
    )
    st.latex(
        r"\frac{1}{\alpha}\frac{\partial T}{\partial t} = "
        r"\frac{\partial^2 T}{\partial r^2} + \frac{1}{r}\frac{\partial T}{\partial r}"
    )

    # Step 2
    st.subheader("2. Accounting for Environmental Heat Loss")
    st.markdown(
        "In experimental conditions, the sample is not perfectly insulated. "
        "A linear heat loss term is introduced to represent convective and radiative exchange "
        "with the surroundings, yielding the **non-adiabatic radial heat equation**:"
    )
    st.latex(
        r"\frac{1}{\alpha}\frac{\partial T}{\partial t} = "
        r"\frac{\partial^2 T}{\partial r^2} + \frac{1}{r}\frac{\partial T}{\partial r} "
        r"- \mu^2\!\left(T - T_\infty\right)"
    )
    st.markdown(
        """
| Symbol | Meaning |
|--------|---------|
| α | Thermal diffusivity (m²/s) |
| μ² | Heat loss coefficient (m⁻²), defined as *hP/kS* for a cylindrical geometry |
| T∞ | Ambient temperature |
"""
    )

    # Step 3
    st.subheader("3. Harmonic Transformation for Periodic Sources")
    st.markdown(
        "When the heat source is modulated at angular frequency ω = 2π/T, the transient "
        "response decays, leaving a steady-state harmonic oscillation. "
        "The temperature is represented as a complex variable:"
    )
    st.latex(r"T(r,t) - T_\infty = \theta(r)\,e^{i\omega t}")
    st.markdown(
        "Substituting into the radial heat equation and dividing by the common factor *e^(iωt)*:"
    )
    st.latex(
        r"\frac{i\omega}{\alpha}\,\theta = "
        r"\frac{d^2\theta}{dr^2} + \frac{1}{r}\frac{d\theta}{dr} - \mu^2\theta"
    )

    # Step 4
    st.subheader("4. The Modified Bessel Equation of Order Zero")
    st.markdown("Rearranging yields a standard ODE:")
    st.latex(
        r"\frac{d^2\theta}{dr^2} + \frac{1}{r}\frac{d\theta}{dr} "
        r"- \left(\frac{i\omega}{\alpha} + \mu^2\right)\theta = 0"
    )
    st.markdown("We define the **Complex Propagation Constant** λ:")
    st.latex(r"\lambda = \sqrt{\dfrac{i\omega}{\alpha} + \mu^2}")
    st.markdown(
        "The spatial temperature θ(*r*) is solved using modified Bessel functions. "
        "For an infinite medium (T → 0 as r → ∞), we use the modified Bessel function "
        "of the second kind, K₀:"
    )
    st.latex(r"\theta(r) = C \cdot K_0(r\lambda)")

    # Step 5
    st.subheader("5. Dual-Sensor Analysis and Asymptotic Expansion")
    st.markdown(
        "Temperature oscillations are measured at two radii, *r₁* and *r₂*. "
        "Analysing the ratio eliminates the source constant *C*:"
    )
    st.latex(
        r"\frac{\theta(r_2)}{\theta(r_1)} = \frac{A_2}{A_1}e^{-i\phi} "
        r"= \frac{K_0(r_2\lambda)}{K_0(r_1\lambda)}"
    )
    st.markdown(
        r"Using the asymptotic expansion $K_0(z) \approx \sqrt{\pi/2z}\,e^{-z}$, "
        "valid for large |*r*λ|, the ratio simplifies to:"
    )
    st.latex(
        r"\frac{A_2}{A_1}e^{-i\phi} \approx \sqrt{\frac{r_1}{r_2}}\cdot e^{-\lambda(r_2-r_1)}"
    )
    st.markdown("Taking the natural logarithm of both sides:")
    st.latex(
        r"\ln\!\left(\frac{A_1\sqrt{r_1}}{A_2\sqrt{r_2}}\right) + i\phi = \lambda(r_2 - r_1)"
    )

    # Step 6
    st.subheader("6. Separation of Real and Imaginary Components")
    st.markdown(
        r"Let Δ*r* = *r₂* − *r₁*. We define λ in terms of its real attenuation component "
        r"(*m_A*) and imaginary phase component (*m_φ*):"
    )
    st.latex(r"\lambda = m_A + i\,m_\varphi")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"m_A = \frac{1}{\Delta r}\ln\!\left(\frac{A_1\sqrt{r_1}}{A_2\sqrt{r_2}}\right)")
    with col2:
        st.latex(r"m_\varphi = \frac{\phi}{\Delta r}")

    # Step 7
    st.subheader("7. Squaring the Propagation Constant")
    st.markdown(
        "To isolate the diffusivity α from the loss term μ², we square λ:"
    )
    st.latex(
        r"\lambda^2 = (m_A + i\,m_\varphi)^2 = (m_A^2 - m_\varphi^2) + i(2\,m_A m_\varphi)"
    )
    st.markdown(r"Recalling that $\lambda^2 = \mu^2 + i\omega/\alpha$ and equating components:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imaginary** (diffusivity):")
        st.latex(r"\frac{\omega}{\alpha} = 2\,m_A m_\varphi")
    with col2:
        st.markdown("**Real** (heat loss):")
        st.latex(r"\mu^2 = m_A^2 - m_\varphi^2")

    st.divider()

    # ── Final Solutions ────────────────────────────────────────────────────────
    st.header("Final Analytical Solutions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("A. Combined Diffusivity Solution")
        st.markdown(
            "Utilises **both amplitude and phase** information, cancelling the environmental "
            "heat-loss term:"
        )
        st.latex(
            r"\alpha_{\text{combined}} = \frac{\omega}{2\,m_A m_\varphi} "
            r"= \frac{\omega\,(\Delta r)^2}{2\,\phi\,\ln\!\left(\dfrac{A_1\sqrt{r_1}}{A_2\sqrt{r_2}}\right)}"
        )
        st.success("Recommended — robust against heat losses.")

    with col2:
        st.subheader("B. Phase-Only Solution")
        st.markdown(
            "Valid under **adiabatic conditions** where heat loss is negligible "
            "(μ² ≈ 0), giving *m_A* = *m_φ*:"
        )
        st.latex(
            r"\alpha_{\text{phase}} = \frac{\omega}{2\,m_\varphi^2} "
            r"= \frac{\omega\,(\Delta r)^2}{2\,\phi^2}"
        )
        st.info("Simpler, but assumes negligible heat loss.")


if __name__ == "__main__":
    main()
