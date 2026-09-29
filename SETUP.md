# Step-by-Step Setup & Deployment Guide

## 1. Prerequisites
- Python 3.10+
- A [Hindsight Cloud API key](https://hindsight.vectorize.io/) (`HINDSIGHT_API_KEY`)
- A [Groq API key](https://console.groq.com/) (`GROQ_API_KEY`)

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

### On macOS / Linux:
```bash
export HINDSIGHT_API_KEY="your-hindsight-api-key"
export HINDSIGHT_BASE_URL="https://api.hindsight.vectorize.io"
export GROQ_API_KEY="your-groq-api-key"
export BANK_ID="factory-floor-v2"
```

### On Windows (PowerShell):
```powershell
$env:HINDSIGHT_API_KEY="your-hindsight-api-key"
$env:HINDSIGHT_BASE_URL="https://api.hindsight.vectorize.io"
$env:GROQ_API_KEY="your-groq-api-key"
$env:BANK_ID="factory-floor-v2"
```

## 4. Seed Hindsight Memory & Create Digital Twins + Directives (Run Once)

```bash
python seed_memory.py --reset
python setup_advanced_memory.py
```

- `seed_memory.py` retains all 11 historical maintenance records (`MAINT-2025-001` through `MAINT-2025-011`) into Hindsight bank `factory-floor-v2`.
- `setup_advanced_memory.py` creates the 7 self-updating per-machine **Mental Models** (`digital-twin-mch-017`, etc.) with `trigger={"refresh_after_consolidation": True}` and registers the 3 standing **Directives** (`grounding-policy`, `loto-before-rotating-equipment`, `mch-042-filter-before-reset`).

## 5. Run the Web Application Locally

```bash
python app.py
```

Open **`http://127.0.0.1:5000`** in your browser.

## 6. Run the Test Suite (No API Keys Required)

```bash
pytest
```
Runs 35 unit and integration tests covering transient retry logic, confidence scoring, citation filtering, thread-safe live feedback logging, CMMS outbox queuing, and Flask API endpoints.