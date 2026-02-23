# Changelog

## [2026-02-23]

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
