# 🛰️ NASA EarthData Explorer — QNASAgent

> **End-to-End MODIS Aerosol Pipeline** — Search · Download · Process · Visualize with AI Agents
>
> Built with **LangGraph + MCP + Streamlit + vLLM (Qwen2.5)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.17.1-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Render-blue)

---

## 📽️ Demo

End-to-end pipeline running live — from NASA EarthData search to aerosol visualizations stored in PostgreSQL:

| Step | Description |
|------|-------------|
| 🔍 **Search** | EarthAgent queries NASA CMR API and returns granule metadata |
| ⬇️ **Download** | HDF4 files saved to `/tmp/earthdata` via `earthaccess` |
| ⚙️ **Pipeline** | HDF4 → CSV → PostgreSQL → 2D Plots (920,400 rows processed) |
| 📊 **Visualize** | Heatmap · Scatter Map · Time Series of Optical Depth |
| 🗄️ **Database** | Statistical summary and custom SQL queries on aerosol data |

---

## 🧠 Architecture

```
User (Streamlit UI)
        │
        ▼
  ┌─────────────┐      ┌──────────────────────┐
  │  EarthAgent │─────▶│  earthaccess_server   │  (MCP Server)
  │  LangGraph  │      │  NASA CMR API         │
  └─────────────┘      └──────────────────────┘
        │
        ▼
  ┌─────────────┐
  │  LLM Router │  Qwen2.5-1.5B-Instruct via vLLM + ngrok
  └─────────────┘
        │
        ▼
  ┌─────────────────────────────────────┐
  │         HDF Pipeline                │
  │  HDF4 (pyhdf) → CSV → PostgreSQL   │
  │  → Scatter Map · Heatmap · TimeSeries│
  └─────────────────────────────────────┘
```

---

## ✨ Features

- **Multi-agent routing** — LangGraph routes queries to `search`, `download`, or `discover` agents automatically
- **MCP integration** — Tools exposed via Model Context Protocol (`earthaccess_server.py`, `hdf_pipeline_server.py`)
- **Local LLM** — Runs `Qwen/Qwen2.5-1.5B-Instruct` locally via vLLM on ROCm/CUDA, exposed through ngrok
- **HDF4 processing** — Extracts aerosol variables from MODIS MYD04_3K using `pyhdf`
- **PostgreSQL persistence** — All processed data stored in Render-hosted PostgreSQL
- **3 plot types** — Scatter map, gridded heatmap, and time series per variable
- **6-tab Streamlit UI** — Search · Download · Discover · Pipeline · Plots · Database

---

## 🗂️ Repository Structure

```
QNASAgent/
├── Main.py                  # Streamlit app (6-tab UI + LangGraph agents)
├── earthaccess_server.py    # MCP server — NASA EarthData search & download
├── hdf_pipeline_server.py   # MCP server — HDF4 processing tools
├── hdf_pipeline.py          # Direct pipeline: HDF4 → CSV → PostgreSQL → Plots
├── pipeline_agent.py        # LangGraph pipeline agent
├── environment.yml          # Conda environment
└── README.md
```

---

## 🚀 Deployment

### Prerequisites

- Python 3.11
- PostgreSQL (local or [Render](https://render.com))
- vLLM with ROCm or CUDA for local LLM inference
- [ngrok](https://ngrok.com) to expose the local LLM endpoint

### 1. Clone & install

```bash
git clone https://github.com/spideriron299/QNASAgent.git
cd QNASAgent
conda env create -f environment.yml
conda activate qnas
```

### 2. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
[earthdata]
username = "your_earthdata_username"
password = "your_earthdata_password"

[postgres]
url = "postgresql://user:pass@host:5432/dbname"

[llm]
base_url = "https://your-ngrok-url.ngrok-free.app/v1"
model = "Qwen/Qwen2.5-1.5B-Instruct"

[dirs]
hdf    = "/tmp/earthdata"
output = "/tmp/aerosol_csv"
plots  = "/tmp/aerosol_plots"
```

> ⚠️ Never commit `secrets.toml` to the repository. Add it to `.gitignore`.

### 3. Start the local LLM (vLLM)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 4. Expose with ngrok

```bash
ngrok http 8000
```

Copy the `https://xxxx.ngrok-free.app` URL into your `[llm] base_url` secret.

### 5. Run the app

```bash
streamlit run Main.py
```

### Streamlit Cloud deployment

Add all secrets in **Settings → Secrets** on [share.streamlit.io](https://share.streamlit.io). Use `/tmp/...` paths for the `[dirs]` section since Streamlit Cloud has a read-only filesystem.

---

## 📊 Aerosol Variables Available

| Variable | Description |
|----------|-------------|
| `Optical_Depth_Land_And_Ocean` | Primary AOD product over land and ocean |
| `Image_Optical_Depth_Land_And_Ocean` | Image-based AOD |
| `Corrected_Optical_Depth_Land_wav2p1` | Corrected AOD at 2.1 µm |
| `Optical_Depth_Ratio_Small_Land` | Fine-mode fraction over land |
| `Angstrom_Exponent_1_Ocean` | Ångström exponent (ocean, band 1) |
| `Angstrom_Exponent_2_Ocean` | Ångström exponent (ocean, band 2) |
| `Mass_Concentration_Land` | Aerosol mass concentration over land |
| `Aerosol_Cloud_Fraction_Land` | Cloud fraction over land |
| `Aerosol_Cloud_Fraction_Ocean` | Cloud fraction over ocean |
| `Fitting_Error_Land` | Retrieval fitting error over land |

---

## 🗄️ Database Schema

```sql
CREATE TABLE aerosol_data (
    id         SERIAL PRIMARY KEY,
    lat        DOUBLE PRECISION,
    lon        DOUBLE PRECISION,
    value      DOUBLE PRECISION,
    variable   TEXT,
    fecha      TIMESTAMP,
    filename   TEXT
);
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | [Streamlit](https://streamlit.io) |
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Tool protocol | [MCP](https://modelcontextprotocol.io) (Model Context Protocol) |
| LLM inference | [vLLM](https://github.com/vllm-project/vllm) — Qwen2.5-1.5B-Instruct |
| LLM tunnel | [ngrok](https://ngrok.com) |
| NASA data | [earthaccess](https://github.com/nsidc/earthaccess) |
| HDF4 reading | [pyhdf](https://pyhdf.sourceforge.net) |
| Database | [PostgreSQL](https://www.postgresql.org) via [psycopg2](https://www.psycopg.org) (hosted on [Render](https://render.com)) |
| Plotting | [Matplotlib](https://matplotlib.org) |
| Data processing | [Pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) |

---

## 📝 Notes

- The free tier of ngrok generates a new URL on every restart — update `[llm] base_url` in your secrets accordingly, or use a paid ngrok static domain.
- `/tmp` on Streamlit Cloud is ephemeral. Downloaded HDF files are lost on app restart, but data already loaded into PostgreSQL persists.
- The LLM (Qwen2.5-1.5B) runs locally on your machine and is accessed remotely by the Streamlit Cloud app via the ngrok tunnel.

---

## 📄 License

MIT

---

*NASA EarthData Explorer · MODIS MYD04_3K Aerosols · LangGraph + MCP + Streamlit*
