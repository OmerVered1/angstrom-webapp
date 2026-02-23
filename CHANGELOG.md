# Changelog

## [2026-02-23] — Unit standardisation: α displayed in mm²/s

### Changes

**All thermal diffusivity (α) values now displayed in mm²/s throughout the app**

Previously all α values were shown in m²/s using scientific notation (e.g. `1.39e-07 m²/s`).
They are now displayed in mm²/s with 4 significant figures (e.g. `0.139 mm²/s`), which is the conventional unit for thermal diffusivity in materials science and avoids scientific notation for the range of values encountered.

Conversion: values are stored in the database in m²/s (SI) and multiplied by 10⁶ for display only.

Locations updated:
- Live analysis results — metric cards and results table
- View Details — parameter list
- Analysis History — table column headers and values
- Summary page — editable table (displays mm²/s, converts back to m²/s on save)
- Statistics page — chart builder y-axis, α vs Period chart, α vs Temperature charts, averages
- Results export CSV — column headers
- Upload Results Image form — input labels now show mm²/s; entered values are divided by 10⁶ before storage
- Summary chart for DB records — bar chart y-axis

**Database fix: Empty (air 50°C) records IDs 43–48**

Alpha values for these records were stored in mm²/s instead of m²/s (inserted incorrectly in the previous session). Corrected by dividing all alpha fields by 10⁶.

---

## [2026-02-23] — Image display fix & missing data

### Bug Fixes

**View Details — original image not displaying**
- Fixed a bug where OCR-uploaded result images were stored as base64 JPEG in the database but View Details tried to parse them as Plotly JSON, silently failed, and fell back to a generic summary chart.
- View Details now detects the content type of `graph_image`: if it is a base64-encoded JPEG or PNG it renders it with `st.image`; if it is Plotly JSON it renders it as an interactive chart; otherwise it falls back to the summary chart.

**Upload Results Image — image not saved to database**
- Fixed a regression where the "Upload Results Image" page saved `graph_json=None`, discarding the uploaded image after OCR extraction.
- The original image bytes are now base64-encoded and stored in the `graph_image` database column when saving from the OCR upload page.

### Database Changes (Supabase)

**Images uploaded to existing records**
- Added missing original front/back panel JPEG images (base64) to records IDs 27–42, which had been uploaded previously without their images.

**New records created — Empty (air 50°C)**
The following 6 records were missing from the database and have been added with data from `summary.xlsx` and images from the Results folder:

| ID | Model | Test Date | Mode | Frequency |
|----|-------|-----------|------|-----------|
| 43 | Empty | 13/01/2026 | Auto   | 0.00077 Hz |
| 44 | Empty | 13/01/2026 | Manual | 0.00077 Hz |
| 45 | Empty | 14/01/2026 | Auto   | 0.00305 Hz |
| 46 | Empty | 14/01/2026 | Manual | 0.00305 Hz |
| 47 | Empty | 05/02/2026 | Auto   | 0.00383 Hz |
| 48 | Empty | 05/02/2026 | Manual | 0.00383 Hz |

**Duplicate records deleted**
The following records were OCR duplicates of cleaner app-generated records and have been removed:

| Deleted ID | Duplicate of | Reason |
|------------|-------------|--------|
| ID 23 — Abs 11/02/2026 Auto   | ID 10 | OCR duplicate, less precise, no image |
| ID 24 — Abs 11/02/2026 Manual | ID 11 | OCR duplicate, less precise, no image |
| ID 25 — Stainless Steel 11/02/2026 Auto   | ID 7 | OCR duplicate, less precise, no image |
| ID 26 — Stainless Steel 11/02/2026 Manual | ID 8 | OCR duplicate, less precise, no image |
