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
    auto_detect_peaks, format_scientific
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
    
    # Header
    st.title("🌡️ Radial Heat Wave Analysis")
    st.caption("Angstrom Method for Thermal Diffusivity Measurement")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["📊 New Analysis", "📷 Upload Results Image", "📋 Results Summary", "📁 Results History"],
            index=0
        )
        
        st.divider()
        st.caption(f"Total analyses saved: {db.get_analysis_count()}")
    
    if page == "📊 New Analysis":
        render_analysis_page()
    elif page == "📷 Upload Results Image":
        render_upload_image_page()
    elif page == "📋 Results Summary":
        render_results_summary_page()
    else:
        render_history_page()


def render_analysis_page():
    """Render the main analysis page."""
    
    # Step 1: File Upload & Parameters
    st.header("1️⃣ File Upload & Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("C80 Calorimeter File")
        c80_file = st.file_uploader("Upload C80 data file", type=['csv', 'txt', 'dat', 'xls', 'xlsx'], key='c80')
        c80_time_unit = st.selectbox("Time Unit", ["Seconds", "Minutes", "Hours", "ms"], key='c80_time')
        c80_pwr_unit = st.selectbox("Power Unit", ["mW", "Watts", "uW"], key='c80_pwr')
        t_cal = st.text_input("C80 Start Time (HH:MM:SS)", "12:00:00")
    
    with col2:
        st.subheader("Keithley Source File")
        src_file = st.file_uploader("Upload Keithley data file", type=['csv', 'txt', 'dat', 'xls', 'xlsx'], key='src')
        src_time_unit = st.selectbox("Time Unit", ["Seconds", "Minutes", "Hours", "ms"], key='src_time')
        src_pwr_unit = st.selectbox("Power Unit", ["Watts", "mW", "uW"], key='src_pwr')
        t_src = st.text_input("Keithley Start Time (HH:MM:SS)", "12:05:00")
    
    st.divider()
    
    # Metadata
    st.subheader("Experiment Metadata")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model_name = st.text_input("Model/Sample Name", "Sample-01")
    with col2:
        test_date = st.text_input("Test Date (DD/MM/YYYY)", datetime.now().strftime("%d/%m/%Y"))
    with col3:
        test_time = st.text_input("Test Time (HH:MM)", datetime.now().strftime("%H:%M"))
    with col4:
        temperature_c = st.number_input("Temperature (°C)", value=25.0, min_value=-50.0, max_value=500.0, step=0.1)
    
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
                    
                    st.success(f"✅ Files loaded! C80: {len(df_cal)} points, Keithley: {len(df_src)} points")
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
                        alpha_combined_cal=results.alpha_combined_cal,
                        alpha_phase_raw=results.alpha_phase_raw,
                        alpha_phase_cal=results.alpha_phase_cal,
                        net_lag_dt=results.net_lag_dt,
                        net_phase_phi=results.net_phase_phi,
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
            'Net Δt (s)': a['net_lag_dt'] or 0.0,
            'α_comb (cal)': a['alpha_combined_cal'] or 0.0,
            'α_phase (cal)': a['alpha_phase_cal'] or 0.0,
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
                
                # Create detailed view
                details = {
                    'Created': analysis['created_at'],
                    'Model': analysis['model_name'],
                    'Test Date': analysis['test_date'],
                    'Radii': f"r₁={analysis['r1_mm']}mm, r₂={analysis['r2_mm']}mm",
                    'Period T': f"{analysis['period_t']:.2f}s",
                    'Frequency': f"{analysis['frequency_f']:.5f}Hz",
                    'Raw Δt': f"{analysis['raw_lag_dt']:.2f}s",
                    'α Combined (raw)': format_scientific(analysis['alpha_combined_raw']),
                    'α Phase (raw)': format_scientific(analysis['alpha_phase_raw']),
                }
                
                if analysis['use_calibration']:
                    details['System Lag'] = f"{analysis['system_lag']}s"
                    details['Net Δt'] = f"{analysis['net_lag_dt']:.2f}s"
                    details['α Combined (cal)'] = format_scientific(analysis['alpha_combined_cal'])
                    details['α Phase (cal)'] = format_scientific(analysis['alpha_phase_cal'])
                
                for k, v in details.items():
                    st.write(f"**{k}:** {v}")
            
            with col2:
                # Show saved graph if available (now stored as JSON)
                if analysis.get('graph_json'):
                    try:
                        saved_fig = json_to_fig(analysis['graph_json'])
                        st.plotly_chart(saved_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display graph: {str(e)}")
                else:
                    st.info("No graph saved for this analysis")
                
                # Delete button
                if st.button("🗑️ Delete this analysis", type="secondary"):
                    if db.delete_analysis(selected_id):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")


if __name__ == "__main__":
    main()
