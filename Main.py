"""
NASA EarthData Explorer — Streamlit App
earthaccess_server.py + hdf_pipeline_server.py + hdf_pipeline.py + pipeline_agent.py

Run:
    streamlit run Main.py

Secrets — add to .streamlit/secrets.toml (local) or Streamlit Cloud → Settings → Secrets:
    [earthdata]
    username = "your_earthdata_user"
    password = "your_earthdata_password"

    [postgres]
    url = "postgresql://user:pass@host:5432/dbname"

    [llm]
    base_url = "http://localhost:8000/v1"
    model    = "Qwen/Qwen2.5-1.5B-Instruct"

    [dirs]
    hdf    = "~/Downloads/earthdata"
    output = "~/aerosol_csv"
    plots  = "~/aerosol_plots"
"""

import asyncio
import logging
import operator
import os
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st

st.set_page_config(
    page_title="NASA EarthData Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Absolute path to this file's directory ──────────────────────────────────
# Fixes "Connection error" on Streamlit Cloud: MCP subprocess is launched with
# an absolute path so it is found regardless of the working directory.
HERE = Path(__file__).resolve().parent

# ─── Secrets loader: secrets.toml → env var → default ────────────────────────
def _s(section: str, key: str, env_var: str = "", default: str = "") -> str:
    try:
        return st.secrets[section][key]
    except Exception:
        pass
    if env_var:
        v = os.environ.get(env_var, "")
        if v:
            return v
    return default

# Load all secrets once at startup and push them into os.environ so MCP
# subprocesses inherit them automatically.
_ED_USER = _s("earthdata", "username", "EARTHDATA_USERNAME")
_ED_PASS = _s("earthdata", "password", "EARTHDATA_PASSWORD")
_DB_URL  = _s("postgres",  "url",      "DB_URL")
_LLM_URL = _s("llm", "base_url", "OPENAI_BASE_URL", "http://localhost:8000/v1")
_LLM_MOD = _s("llm", "model",    "LLM_MODEL",       "Qwen/Qwen2.5-1.5B-Instruct")
_HDF_DIR = _s("dirs", "hdf",    "HDF_DIR",    str(Path.home() / "Downloads" / "earthdata"))
_OUT_DIR = _s("dirs", "output", "OUTPUT_DIR", str(Path.home() / "aerosol_csv"))
_PLT_DIR = _s("dirs", "plots",  "PLOTS_DIR",  str(Path.home() / "aerosol_plots"))

if _ED_USER: os.environ["EARTHDATA_USERNAME"] = _ED_USER
if _ED_PASS: os.environ["EARTHDATA_PASSWORD"] = _ED_PASS
if _DB_URL:  os.environ["DB_URL"]             = _DB_URL

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
h1,h2,h3{font-family:'IBM Plex Mono',monospace}
.hero{background:linear-gradient(135deg,#050d1a 0%,#0a1e35 60%,#061525 100%);border:1px solid #1a3555;border-radius:14px;padding:28px 36px;margin-bottom:22px;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-60px;right:-60px;width:260px;height:260px;background:radial-gradient(circle,rgba(0,160,255,.10) 0%,transparent 70%);border-radius:50%}
.hero h1{color:#dff0ff;font-size:1.75rem;margin:0;letter-spacing:-.5px}
.hero p{color:#6aa6cc;margin:6px 0 0;font-size:.92rem}
.step-title{font-family:'IBM Plex Mono',monospace;font-size:.85rem;color:#60b0ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.result-box{background:#060e1c;border:1px solid #1a3555;border-radius:9px;padding:16px 20px;font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#b8d8f0;line-height:1.75;white-space:pre-wrap;overflow-x:auto}
.metric-row{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}
.metric-pill{background:#0d1f38;border:1px solid #1e4070;border-radius:8px;padding:10px 18px;text-align:center;flex:1;min-width:120px}
.metric-pill .val{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;color:#4cc9f0}
.metric-pill .lbl{font-size:.75rem;color:#5a8aaa;margin-top:3px}
section[data-testid="stSidebar"]{background:#07101d;border-right:1px solid #142236}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛰️ NASA EarthData Explorer</h1>
    <p>Search · Download · Process · Visualize — MODIS aerosol data with AI agents</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    with st.expander("🔑 NASA EarthData credentials",
                     expanded=not bool(_ED_USER)):
        ed_user = st.text_input("Username", value="", placeholder="Loaded from secrets",
                                type="password", key="ed_user")
        ed_pass = st.text_input("Password", value="", placeholder="Loaded from secrets",
                                type="password", key="ed_pass")
        if _ED_USER:
            st.caption("✅ Credentials loaded from secrets")

    with st.expander("📂 Directories"):
        hdf_dir    = st.text_input("HDF / Downloads", value=_HDF_DIR)
        output_dir = st.text_input("CSV output",      value=_OUT_DIR)
        plots_dir  = st.text_input("Plots output",    value=_PLT_DIR)

    with st.expander("🗄️ PostgreSQL", expanded=not bool(_DB_URL)):
        db_url = st.text_input("DB URL", value="", placeholder="Loaded from secrets",
                               type="password", key="db_url_input")
        if _DB_URL:
            st.caption("✅ DB URL loaded from secrets")

    with st.expander("🤖 LLM Model (LM Studio)"):
        llm_base_url = st.text_input("Base URL", value=_LLM_URL)
        llm_model    = st.text_input("Model",    value=_LLM_MOD)

    variable = st.selectbox("📡 Aerosol variable", [
        "Optical_Depth_Land_And_Ocean",
        "Image_Optical_Depth_Land_And_Ocean",
        "Corrected_Optical_Depth_Land_wav2p1",
        "Optical_Depth_Ratio_Small_Land",
        "Angstrom_Exponent_1_Ocean",
        "Angstrom_Exponent_2_Ocean",
        "Mass_Concentration_Land",
        "Aerosol_Cloud_Fraction_Land",
        "Aerosol_Cloud_Fraction_Ocean",
        "Fitting_Error_Land",
    ])

    if st.button("💾 Apply settings", use_container_width=True):
        # Only override secrets if the user typed something
        if ed_user: os.environ["EARTHDATA_USERNAME"] = ed_user
        if ed_pass: os.environ["EARTHDATA_PASSWORD"] = ed_pass
        _active_db = db_url if db_url else _DB_URL
        if _active_db: os.environ["DB_URL"] = _active_db
        os.environ["OPENAI_BASE_URL"] = llm_base_url
        os.environ["LLM_MODEL"]       = llm_model
        for d in [hdf_dir, output_dir, plots_dir]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)
        st.success("✅ Done")

# Resolved DB URL to use in widgets (never displayed in plaintext)
_active_db_url = db_url if db_url else _DB_URL


# ─── Async runner ─────────────────────────────────────────────────────────────
def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─── EarthAgent ───────────────────────────────────────────────────────────────
def _build_earth_agent():
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    # ✅ KEY FIX: absolute path so the subprocess is always found
    EARTH_SERVER = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "earthaccess_server.py")],
        env={**os.environ},
    )

    model = ChatOpenAI(model=llm_model, base_url=llm_base_url, api_key="not-required")

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]

    async def router(state):
        msg = await model.ainvoke(
            [SystemMessage(content=(
                "You are a NASA EarthData assistant router. "
                "Return exactly one word: 'search', 'download', or 'discover'."
            ))] + state["messages"]
        )
        return {"messages": [msg]}

    def _route(state):
        raw = re.sub(r"<think>.*?</think>", "", state["messages"][-1].content,
                     flags=re.DOTALL).strip().lower()
        for opt in ("download", "discover", "search"):
            if opt in raw:
                return opt
        return "search"

    async def _agent(state, sys_prompt):
        user_q = state["messages"][-2]
        async with stdio_client(EARTH_SERVER) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                mwt   = model.bind_tools(tools)
                msgs  = [SystemMessage(content=sys_prompt), user_q]
                resp  = await mwt.ainvoke(msgs)
                msgs.append(resp)
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        res = await session.call_tool(tc["name"], arguments=tc["args"])
                        msgs.append(ToolMessage(content=res.content[-1].text,
                                                tool_call_id=tc["id"]))
                    resp = await mwt.ainvoke(msgs)
                    msgs.append(resp)
                return {"messages": msgs[1:]}

    async def search_agent(state):
        return await _agent(state,
            "You are a NASA EarthData specialist. Use MCP tools to search granules. "
            "Extract bbox and dates from the request. Report how many granules were found "
            "and a brief metadata summary.")

    async def download_agent(state):
        return await _agent(state,
            "You are a NASA EarthData download specialist. Use download_granules. "
            "Extract concept_id/short_name, bbox, dates, and output dir. "
            "Confirm how many files were saved and where.")

    async def discover_agent(state):
        return await _agent(state,
            "You are a NASA EarthData catalog specialist. Use discover_datasets. "
            "Present results as a table: concept_id, short_name, version, provider, title.")

    g = StateGraph(AgentState)
    g.add_node("router",         router)
    g.add_node("search_agent",   search_agent)
    g.add_node("download_agent", download_agent)
    g.add_node("discover_agent", discover_agent)
    g.add_edge(START, "router")
    g.add_edge("search_agent",   END)
    g.add_edge("download_agent", END)
    g.add_edge("discover_agent", END)
    g.add_conditional_edges("router", _route, {
        "search": "search_agent", "download": "download_agent", "discover": "discover_agent"
    })
    return g.compile()


# ─── Dependency check ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _check_deps():
    missing = []
    for m in ["langgraph", "langchain_openai", "mcp", "langchain_mcp_adapters",
              "earthaccess", "pyhdf", "psycopg2", "matplotlib", "pandas"]:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    return missing

_missing = _check_deps()
if _missing:
    st.warning(f"⚠️ Missing dependencies: `{'`, `'.join(_missing)}`\n\n"
               f"```\npip install {' '.join(_missing)}\n```")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔍 Search granules",
    "⬇️ Download",
    "🗂️ Discover datasets",
    "⚙️ HDF Pipeline",
    "📊 Plots",
    "🗄️ Database",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 0 — SEARCH GRANULES
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="step-title">🔍 Search satellite granules</div>',
                unsafe_allow_html=True)
    st.caption("Calls `search_by_concept_id` or `search_by_short_name` via EarthAgent.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        s_concept = st.text_input("Concept ID", "C1443528505-LAADS", key="s_concept",
                                  help="e.g. C1443528505-LAADS for MYD04_3K v6.1")
    with c2:
        s_short = st.text_input("Short name (alternative)", "MYD04_3K", key="s_short")
    with c3:
        s_max = st.number_input("Max granules", min_value=1, max_value=50,
                                value=5, key="s_max")

    st.markdown("**Bounding box** — (west, south, east, north)")
    b1, b2, b3, b4 = st.columns(4)
    s_w = b1.number_input("West",  value=-10.0, min_value=-180.0, max_value=180.0, key="s_w")
    s_s = b2.number_input("South", value= 20.0, min_value= -90.0, max_value= 90.0, key="s_s")
    s_e = b3.number_input("East",  value= 10.0, min_value=-180.0, max_value=180.0, key="s_e")
    s_n = b4.number_input("North", value= 50.0, min_value= -90.0, max_value= 90.0, key="s_n")

    dc1, dc2 = st.columns(2)
    s_d1 = dc1.date_input("Start date", value=None, key="s_d1")
    s_d2 = dc2.date_input("End date",   value=None, key="s_d2")

    if st.button("🔍 Search", key="btn_search", use_container_width=True):
        id_part   = f"(concept_id {s_concept})" if s_concept else f"(short_name {s_short})"
        date_part = f" from {s_d1} to {s_d2}," if s_d1 and s_d2 else ""
        query = (f"Search MODIS aerosol data {id_part} "
                 f"in region ({s_w}, {s_s}, {s_e}, {s_n})"
                 f"{date_part} max {s_max} granules.")
        with st.spinner("🛰️ Querying NASA EarthData..."):
            try:
                from langchain_core.messages import HumanMessage
                result = _run(_build_earth_agent().ainvoke(
                    {"messages": [HumanMessage(content=query)]}))
                reply = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"**Error:** {e}")
                st.info("Check that credentials are set in secrets and that "
                        "`earthaccess_server.py` is in the same directory as `Main.py`.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="step-title">⬇️ Download granules</div>',
                unsafe_allow_html=True)
    st.caption("Calls `download_granules` via EarthAgent. HDF files are saved to the configured directory.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        d_concept = st.text_input("Concept ID", "C1443528505-LAADS", key="d_concept")
    with c2:
        d_short   = st.text_input("Short name", "MYD04_3K", key="d_short")
    with c3:
        d_max = st.number_input("Max granules", min_value=1, max_value=20,
                                value=2, key="d_max")

    st.markdown("**Bounding box**")
    b1, b2, b3, b4 = st.columns(4)
    d_w = b1.number_input("West",  value=-10.0, min_value=-180.0, max_value=180.0, key="d_w")
    d_s = b2.number_input("South", value= 20.0, min_value= -90.0, max_value= 90.0, key="d_s")
    d_e = b3.number_input("East",  value= 10.0, min_value=-180.0, max_value=180.0, key="d_e")
    d_n = b4.number_input("North", value= 50.0, min_value= -90.0, max_value= 90.0, key="d_n")

    dc1, dc2 = st.columns(2)
    d_d1 = dc1.date_input("Start date", value=None, key="d_d1")
    d_d2 = dc2.date_input("End date",   value=None, key="d_d2")
    dl_dir = st.text_input("📁 Destination directory", value=hdf_dir, key="dl_dir")

    if st.button("⬇️ Download", key="btn_dl", use_container_width=True):
        date_part = f" from {d_d1} to {d_d2}" if d_d1 and d_d2 else ""
        query = (f"Download {d_max} granules of {d_short} ({d_concept}) "
                 f"bbox ({d_w}, {d_s}, {d_e}, {d_n}){date_part} into {dl_dir}")
        with st.spinner("⬇️ Downloading satellite data..."):
            try:
                from langchain_core.messages import HumanMessage
                result = _run(_build_earth_agent().ainvoke(
                    {"messages": [HumanMessage(content=query)]}))
                reply = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>',
                            unsafe_allow_html=True)
                dl_path   = Path(dl_dir).expanduser()
                hdf_files = list(dl_path.glob("*.hdf")) + list(dl_path.glob("*.HDF"))
                if hdf_files:
                    st.success(f"✅ {len(hdf_files)} HDF file(s) in `{dl_path}`")
                    with st.expander("View downloaded files"):
                        for f in sorted(hdf_files):
                            st.text(f"📄 {f.name}  —  {f.stat().st_size/1024**2:.1f} MB")
            except Exception as e:
                st.error(f"**Error:** {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — DISCOVER DATASETS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="step-title">🗂️ Discover NASA CMR collections</div>',
                unsafe_allow_html=True)
    st.caption("Calls `discover_datasets`. Returns concept_id, short_name, version, provider and title.")

    disc_kw = st.text_input("🔑 Keyword", "aerosol",
                             placeholder="aerosol · sea surface temperature · NDVI · precipitation …")

    if st.button("🗂️ Discover datasets", key="btn_discover", use_container_width=True):
        with st.spinner("🔎 Querying NASA CMR catalog..."):
            try:
                from langchain_core.messages import HumanMessage
                result = _run(_build_earth_agent().ainvoke({"messages": [
                    HumanMessage(content=f"Find NASA EarthData datasets related to {disc_kw}.")
                ]}))
                reply = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"**Error:** {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — HDF PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="step-title">⚙️ Pipeline HDF → CSV → PostgreSQL → Plots</div>',
                unsafe_allow_html=True)
    st.info("**Step 1** HDF4 → CSV via `pyhdf`  ·  "
            "**Step 2** CSV → PostgreSQL  ·  "
            "**Step 3** 2-D plots (scatter map · heatmap · time series)")

    pc1, pc2 = st.columns(2)
    with pc1:
        p_hdf   = st.text_input("📁 HDF directory",   value=hdf_dir,    key="p_hdf")
        p_out   = st.text_input("📂 CSV directory",   value=output_dir, key="p_out")
    with pc2:
        p_plots = st.text_input("🖼️ Plots directory", value=plots_dir,  key="p_plots")
        # DB URL comes from secrets; shown masked
        p_db_override = st.text_input("🗄️ DB URL (leave blank to use secret)",
                                      value="", type="password", key="p_db")

    p_var = st.selectbox("Aerosol variable", [
        "Optical_Depth_Land_And_Ocean",
        "Image_Optical_Depth_Land_And_Ocean",
        "Corrected_Optical_Depth_Land_wav2p1",
        "Optical_Depth_Ratio_Small_Land",
        "Angstrom_Exponent_1_Ocean",
        "Angstrom_Exponent_2_Ocean",
        "Mass_Concentration_Land",
        "Aerosol_Cloud_Fraction_Land",
        "Aerosol_Cloud_Fraction_Ocean",
        "Fitting_Error_Land",
    ], key="p_var")

    sk1, sk2, sk3 = st.columns(3)
    skip_convert = sk1.checkbox("⏭️ Skip HDF→CSV")
    skip_load    = sk2.checkbox("⏭️ Skip DB load")
    skip_plot    = sk3.checkbox("⏭️ Skip plots")

    p_mode = st.radio("Execution mode",
        ["⚡ Direct — run_pipeline()",
         "🤖 Agents — PipelineAgent (LangGraph + MCP)"],
        horizontal=True)

    if st.button("▶️ Run pipeline", key="btn_pipe", use_container_width=True):
        p_db = p_db_override if p_db_override else _active_db_url
        for d in [p_hdf, p_out, p_plots]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)

        log_ph: list[str] = []
        log_box = st.empty()

        class _SH(logging.Handler):
            def emit(self, r):
                log_ph.append(self.format(r))
                log_box.markdown(
                    '<div class="result-box">' + "<br>".join(log_ph[-30:]) + "</div>",
                    unsafe_allow_html=True)

        plog = logging.getLogger("pipe_run")
        plog.setLevel(logging.INFO)
        sh = _SH()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          "%H:%M:%S"))
        plog.addHandler(sh)

        with st.spinner("⚙️ Running pipeline..."):
            try:
                if "Direct" in p_mode:
                    from hdf_pipeline import run_pipeline
                    results = run_pipeline(
                        hdf_dir=p_hdf, output_dir=p_out, plots_dir=p_plots,
                        db_url=p_db, variable=p_var,
                        skip_convert=skip_convert, skip_load=skip_load,
                        skip_plot=skip_plot, logger=plog,
                    )
                else:
                    from pipeline_agent import PipelineAgent, run_pipeline_agents
                    results = _run(run_pipeline_agents(
                        hdf_dir=p_hdf, output_dir=p_out, plots_dir=p_plots,
                        db_url=p_db, variable=p_var,
                        agent=PipelineAgent(model_name=llm_model),
                    ))

                csvs_n = len(results.get("csvs",  []))
                rows_n = results.get("rows_loaded", 0)
                plot_n = len(results.get("plots", []))
                st.markdown(f"""
<div class="metric-row">
  <div class="metric-pill"><div class="val">{csvs_n}</div><div class="lbl">CSVs generated</div></div>
  <div class="metric-pill"><div class="val">{rows_n:,}</div><div class="lbl">Rows in DB</div></div>
  <div class="metric-pill"><div class="val">{plot_n}</div><div class="lbl">Plots</div></div>
</div>""", unsafe_allow_html=True)

                if results.get("plots"):
                    st.markdown("#### Generated plots")
                    cols = st.columns(min(3, len(results["plots"])))
                    for i, p in enumerate(results["plots"]):
                        cols[i % 3].image(str(p), caption=Path(p).name,
                                          use_container_width=True)

                if results.get("messages") and "Agents" in p_mode:
                    with st.expander("📋 Agent messages"):
                        for m in results["messages"]:
                            st.text(m.content if hasattr(m, "content") else str(m))

                st.session_state["last_plots"] = results.get("plots", [])

            except ImportError as e:
                st.error(f"Module not found: **{e}**")
                st.info("Make sure `hdf_pipeline.py` and `pipeline_agent.py` are in the "
                        "same directory as `Main.py`.")
            except Exception as e:
                st.error(f"**Pipeline error:** {e}")
            finally:
                plog.removeHandler(sh)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — PLOTS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="step-title">📊 Aerosol visualizations</div>',
                unsafe_allow_html=True)

    st.button("🔄 Refresh from disk", key="btn_refresh")

    plots_path = Path(plots_dir).expanduser()
    disk_pngs  = sorted(plots_path.glob("*.png")) if plots_path.exists() else []
    last_plots = [Path(p) for p in st.session_state.get("last_plots", [])]
    all_pngs   = list({str(p): p for p in disk_pngs + last_plots}.values())

    if not all_pngs:
        st.info(f"No plots in `{plots_path}`. Run the pipeline first.")
    else:
        st.caption(f"{len(all_pngs)} image(s) in `{plots_path}`")
        cols = st.columns(2)
        for i, png in enumerate(sorted(all_pngs, key=str)):
            cols[i % 2].image(str(png), caption=Path(png).name,
                              use_container_width=True)

    st.divider()
    st.markdown("#### Generate plots directly from PostgreSQL")

    g1, g2 = st.columns(2)
    with g1:
        g_var  = st.selectbox("Variable", [
            "Optical_Depth_Land_And_Ocean",
            "Image_Optical_Depth_Land_And_Ocean",
            "Corrected_Optical_Depth_Land_wav2p1",
        ], key="g_var")
        g_vmin = st.number_input("Vmin", value=0.0, key="g_vmin")
        g_vmax = st.number_input("Vmax", value=1.5, key="g_vmax")
    with g2:
        g_res     = st.number_input("Heatmap resolution (°)", value=0.5, key="g_res")
        g_db_over = st.text_input("DB URL (leave blank to use secret)",
                                  value="", type="password", key="g_db")

    g_types = st.multiselect("Plot types",
        ["scatter_map", "heatmap", "time_series"],
        default=["scatter_map", "heatmap", "time_series"], key="g_types")

    if st.button("📊 Generate from DB", key="btn_gen", use_container_width=True):
        g_db = g_db_over if g_db_over else _active_db_url
        with st.spinner("Generating plots..."):
            try:
                from hdf_pipeline import AerosolPlotter, PostgresLoader
                loader  = PostgresLoader(db_url=g_db)
                plotter = AerosolPlotter(loader=loader, plots_dir=plots_dir)
                generated = []
                if "scatter_map" in g_types:
                    p = plotter.scatter_map(variable=g_var, vmin=g_vmin, vmax=g_vmax)
                    if p: generated.append(p)
                if "heatmap" in g_types:
                    p = plotter.heatmap(variable=g_var, resolution=g_res,
                                        vmin=g_vmin, vmax=g_vmax)
                    if p: generated.append(p)
                if "time_series" in g_types:
                    p = plotter.time_series(variable=g_var)
                    if p: generated.append(p)

                if generated:
                    cols = st.columns(min(3, len(generated)))
                    for i, png in enumerate(generated):
                        cols[i % 3].image(str(png), caption=png.name,
                                          use_container_width=True)
                    st.session_state["last_plots"] = [str(p) for p in generated]
                else:
                    st.warning("No plots generated — DB may be empty for this variable.")
            except ImportError:
                st.error("`hdf_pipeline.py` not found.")
            except Exception as e:
                st.error(f"**Error:** {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — DATABASE
# ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="step-title">🗄️ PostgreSQL — aerosol_data</div>',
                unsafe_allow_html=True)

    # DB URL always comes from secrets unless the user explicitly overrides
    db5_override = st.text_input("DB URL (leave blank to use secret)",
                                 value="", type="password", key="db5")
    db5 = db5_override if db5_override else _active_db_url

    col_sum, col_sql = st.columns(2)

    with col_sum:
        st.markdown("**Statistical summary**")
        if st.button("📋 View summary", key="btn_summary", use_container_width=True):
            with st.spinner("Querying..."):
                try:
                    from hdf_pipeline import PostgresLoader
                    df = PostgresLoader(db_url=db5).summary()
                    if df.empty:
                        st.info("Table `aerosol_data` is empty.")
                    else:
                        st.dataframe(df, use_container_width=True)
                        if "puntos" in df.columns:
                            tot = int(df["puntos"].sum())
                            st.markdown(
                                f'<div class="metric-row">'
                                f'<div class="metric-pill"><div class="val">{len(df)}</div>'
                                f'<div class="lbl">Variables</div></div>'
                                f'<div class="metric-pill"><div class="val">{tot:,}</div>'
                                f'<div class="lbl">Total points</div></div>'
                                f'</div>', unsafe_allow_html=True)
                except ImportError:
                    st.error("`hdf_pipeline.py` not found.")
                except Exception as e:
                    st.error(f"**Error:** {e}")

    with col_sql:
        st.markdown("**Custom SQL query**")
        sql = st.text_area("SQL",
            "SELECT variable, COUNT(*) AS n "
            "FROM aerosol_data GROUP BY variable ORDER BY n DESC;",
            height=130, key="sql")
        if st.button("▶️ Run SQL", key="btn_sql", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    import pandas as pd, psycopg2
                    conn = psycopg2.connect(db5)
                    df   = pd.read_sql(sql, conn)
                    conn.close()
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} row(s) returned")
                except Exception as e:
                    st.error(f"**SQL error:** {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#2a5a8a;font-size:.77rem;"
    "font-family:IBM Plex Mono,monospace;'>"
    "NASA EarthData Explorer · MODIS MYD04_3K Aerosols · LangGraph + MCP + Streamlit"
    "</div>", unsafe_allow_html=True)