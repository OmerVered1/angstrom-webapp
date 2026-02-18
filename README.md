# Radial Heat Wave Analysis - Web Application

A web application for analysis of thermal diffusivity measurement using radial modified Angstrom method.

## Features

- **File Upload**: Upload C80 calorimeter and Keithley source data files
- **Interactive Region Selection**: Use slider to select stable analysis region
- **Automatic Peak Detection**: Auto mode finds peaks automatically
- **Dual Calculation Methods**: Combined method and Phase-only method
- **System Calibration**: Optional lag correction
- **Results Storage**: SQLite database saves all analyses
- **Export Options**: Download results as CSV or interactive HTML report
- **History View**: Browse and review all past analyses

## Installation

1. **Install Python dependencies:**
   ```bash
   cd angstrom_webapp
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## Usage

### New Analysis

1. **Upload Files**: Select your C80 and Keithley data files (CSV, TXT, or DAT format)
2. **Set Parameters**: Configure time/power units and start times
3. **Enter Metadata**: Model name, test date, radii values
4. **Calibration** (optional): Enable and set system lag value
5. **Load Files**: Click "Load & Process Files"
6. **Select Region**: Use the slider to select a stable region
7. **Run Analysis**: Click "Run Analysis" to calculate thermal diffusivity
8. **Save Results**: Export as CSV/HTML or save to database

### Results History

- View all saved analyses in a summary table
- Click on any analysis to see full details
- Delete old analyses as needed

## File Format

The app accepts CSV/TSV files with two columns:
- Column 1: Time
- Column 2: Power/Heat flow

Example:
```
0.0,0.125
0.5,0.127
1.0,0.134
...
```

## Calculated Parameters

- **α (Combined Method)**: Uses both amplitude ratio and phase shift
- **α (Phase Method)**: Uses only phase shift (phase-only method)
- **Period (T)**: Oscillation period in seconds
- **Frequency (f)**: 1/T in Hz
- **Angular Frequency (ω)**: 2π/T in rad/s
- **Phase Shift (φ)**: ω × Δt

## Database

Results are stored in `angstrom_results.db` (SQLite) in the app directory.

## Technologies

- **Frontend**: Streamlit
- **Plotting**: Plotly (interactive graphs)
- **Database**: SQLite
- **Analysis**: NumPy, SciPy, Pandas
