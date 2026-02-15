# Deployment Guide - Radial Heat Wave Analysis Web App

## Quick Deploy to Streamlit Cloud (FREE - Recommended)

### Step 1: Push to GitHub

```bash
cd angstrom_webapp

# Initialize git
git init
git add .
git commit -m "Initial commit - Angstrom Analysis Web App"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/angstrom-webapp.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/angstrom-webapp`
5. Main file: `app.py`
6. Click "Deploy!"

**Your app will be live at:** `https://YOUR_USERNAME-angstrom-webapp.streamlit.app`

---

## ⚠️ Important: Data Persistence on Cloud

**Streamlit Cloud has ephemeral storage** - the SQLite database resets when the app restarts.

### Options for Permanent Data Storage:

#### Option A: Supabase (FREE PostgreSQL) - Recommended
1. Create free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Get your database URL from Settings > Database
4. Add to Streamlit Cloud secrets:
   ```toml
   [database]
   url = "postgresql://user:password@host:5432/dbname"
   ```

#### Option B: Google Sheets (Simple)
Use `gspread` library to store results in a Google Sheet.

#### Option C: University Server (BGU)
Contact BGU IT to host on an internal server where SQLite will persist.

---

## Share with Colleagues

Once deployed, share the URL:
```
https://your-app-name.streamlit.app
```

Anyone with the link can:
- Upload their data files
- Run analyses
- View results
- Download reports

---

## Local Network Access (Same Wi-Fi)

To let colleagues on the same network access your local version:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then share your IP address: `http://YOUR_IP:8501`

Find your IP:
```powershell
ipconfig | findstr "IPv4"
```

---

## Environment Variables

For production, create `.streamlit/secrets.toml`:

```toml
# Optional: Admin email for notifications
admin_email = "omerve@post.bgu.ac.il"

# Optional: Database URL for cloud persistence
# database_url = "postgresql://..."
```

---

## Contact

App developed for thermal diffusivity analysis using the Angstrom method.
Contact: omerve@post.bgu.ac.il
