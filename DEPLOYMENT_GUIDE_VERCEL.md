# Complete Vercel Deployment Guide

This guide explains how to deploy the **AI-Based Student Identification & Campus Entry Monitoring System** to **Vercel** with global edge performance and zero server hanging.

---

## 1. Why Vercel?

- **Global Edge CDN**: Static assets (CSS, JS, icons, demo photos) load in < 50ms worldwide.
- **Serverless Python**: Scalable on-demand API execution (`@vercel/python`) with zero idle resource exhaustion.
- **No Cold Start Freezes**: Lightweight footprint (< 90 MB total dependencies) fits comfortably within Vercel's 250 MB serverless ceiling.
- **Always Available**: Free tier with HTTPS encryption out of the box.

---

## 2. Pre-Configured Files in the Repository

The codebase has already been fully configured for Vercel:
1. `vercel.json` — Routes static requests directly to Vercel's CDN and sends API/dynamic requests to `@vercel/python`.
2. `api/index.py` — Serverless function entrypoint exporting the Flask `app`.
3. `requirements.txt` — Optimized dependencies without bloated C++ wheels so deployment builds in under 30 seconds.
4. `config.py` — Automatically switches to writable `/tmp` filesystem on Vercel.

---

## 3. Step-by-Step Deployment Steps

### Step 1: Push the Latest Code to GitHub
In your local terminal inside `Research`, run:
```bash
git push --force -u origin main
```
Enter your GitHub Username and Personal Access Token (PAT).

---

### Step 2: Import into Vercel
1. Open [vercel.com](https://vercel.com) and sign in with your GitHub account.
2. In your Vercel Dashboard, click **Add New...** (top right) $\rightarrow$ **Project**.
3. Under **Import Git Repository**, find your repository (`good`) and click **Import**.

---

### Step 3: Configure Project Settings
1. **Project Name**: `campus-face-system` (or your preferred name).
2. **Framework Preset**: Leave as **Other** (Vercel reads `vercel.json` automatically).
3. **Root Directory**: `./` (leave default).
4. **Build & Output Settings**: Leave empty / default.

---

### Step 4: Add Environment Variables
Expand the **Environment Variables** section and add the following keys:

| Key | Value | Description |
|---|---|---|
| `MONGO_URI` | `mongodb+srv://arnnavpanda_db_user:motupanda@cluster0.ncj9dnd.mongodb.net/?appName=Cluster0` | MongoDB Atlas Connection String |
| `DB_NAME` | `campus_facial_id_db` | Database Name |
| `SECRET_KEY` | `campus_ai_super_secret_jwt_key_2026_x89q2` | JWT Token Encryption Secret |
| `DEFAULT_INSTITUTION_NAME` | `Institute` | Display Institution Name |
| `DEFAULT_CAMPUS_LOCATION` | `Main Campus, North Gate` | Active Gate Lane Name |
| `DEFAULT_ADMIN_EMAIL` | `admin@institution.edu` | Admin Portal Login Email |
| `DEFAULT_ADMIN_PASSWORD` | `AdminSecure2026!` | Admin Portal Login Password |
| `SIMILARITY_THRESHOLD` | `0.68` | Face Match Acceptance Threshold |
| `COOLDOWN_PERIOD_SECONDS` | `30` | Anti-duplicate Attendance Cooldown |

---

### Step 5: Click Deploy
1. Click the blue **Deploy** button.
2. Vercel will install dependencies and deploy in approximately 45–60 seconds.
3. Once finished, click **Continue to Dashboard** or visit your live `.vercel.app` URL!

---

## 4. Default Admin Login Credentials

Once your Vercel app is live:
1. Click **Admin Sign In** on the top navigation bar.
2. Enter:
   - **Email**: `admin@institution.edu`
   - **Password**: `AdminSecure2026!`
   *(Or click the "Auto-Fill Default Credentials" button inside the modal)*
3. Click **Sign In to Admin Portal**. You will enter the dashboard immediately.
