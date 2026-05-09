"""
NASA EarthData Explorer — Streamlit App
Basado en prueba1.ipynb + earthaccess_server.py + hdf_pipeline.py
                         + hdf_pipeline_server.py + pipeline_agent.py

Uso:
    streamlit run Main.py

Secrets (Streamlit Cloud → Settings → Secrets  O  .streamlit/secrets.toml local):
─────────────────────────────────────────────────────────────────────────────────
[earthdata]
username = "TU_USUARIO_EARTHDATA"
password = "TU_PASSWORD_EARTHDATA"

[postgres]
url = "postgresql://usuario:password@host:5432/base_de_datos"

[llm]
base_url = "http://localhost:8000/v1"
model    = "Qwen/Qwen2.5-1.5B-Instruct"

[dirs]
hdf    = "~/Downloads/earthdata"
output = "~/aerosol_csv"
plots  = "~/aerosol_plots"
─────────────────────────────────────────────────────────────────────────────────
"""

# ─── std-lib ─────────────────────────────────────────────────────────────────
import asyncio
import logging
import operator
import os
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# ─── Streamlit ───────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="NASA EarthData Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helper para leer secrets: secrets.toml → env vars → default ─────────────
def _s(section: str, key: str, env_var: str = "", default: str = "") -> str:
    """Prioridad: st.secrets → os.environ → default."""
    try:
        return st.secrets[section][key]
    except Exception:
        pass
    if env_var:
        val = os.environ.get(env_var, "")
        if val:
            return val
    return default

# Cargar todos los secrets al inicio para inyectarlos en os.environ
# (los servidores MCP los leen desde el entorno al arrancar)
_ED_USER  = _s("earthdata", "username", "EARTHDATA_USERNAME")
_ED_PASS  = _s("earthdata", "password", "EARTHDATA_PASSWORD")
_DB_URL   = _s("postgres",  "url",      "DB_URL")
_LLM_URL  = _s("llm",       "base_url", "OPENAI_BASE_URL", "http://localhost:8000/v1")
_LLM_MOD  = _s("llm",       "model",    "LLM_MODEL",       "Qwen/Qwen2.5-1.5B-Instruct")
_HDF_DIR  = _s("dirs",      "hdf",      "HDF_DIR",  str(Path.home() / "Downloads" / "earthdata"))
_OUT_DIR  = _s("dirs",      "output",   "OUTPUT_DIR", str(Path.home() / "aerosol_csv"))
_PLT_DIR  = _s("dirs",      "plots",    "PLOTS_DIR",  str(Path.home() / "aerosol_plots"))

# Propagar al entorno para que los subprocesos MCP los hereden
if _ED_USER: os.environ["EARTHDATA_USERNAME"] = _ED_USER
if _ED_PASS: os.environ["EARTHDATA_PASSWORD"] = _ED_PASS
if _DB_URL:  os.environ["DB_URL"]             = _DB_URL

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }

.hero {
    background: linear-gradient(135deg,#050d1a 0%,#0a1e35 60%,#061525 100%);
    border:1px solid #1a3555; border-radius:14px;
    padding:28px 36px; margin-bottom:22px; position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(0,160,255,.10) 0%,transparent 70%);
    border-radius:50%;
}
.hero h1 { color:#dff0ff; font-size:1.75rem; margin:0; letter-spacing:-0.5px; }
.hero p  { color:#6aa6cc; margin:6px 0 0; font-size:.92rem; }

.step-title {
    font-family:'IBM Plex Mono',monospace; font-size:.85rem;
    color:#60b0ff; text-transform:uppercase; letter-spacing:1px;
    margin-bottom:8px;
}
.result-box {
    background:#060e1c; border:1px solid #1a3555; border-radius:9px;
    padding:16px 20px; font-family:'IBM Plex Mono',monospace; font-size:.82rem;
    color:#b8d8f0; line-height:1.75; white-space:pre-wrap; overflow-x:auto;
}
.metric-row { display:flex; gap:16px; flex-wrap:wrap; margin:12px 0; }
.metric-pill {
    background:#0d1f38; border:1px solid #1e4070;
    border-radius:8px; padding:10px 18px; text-align:center; flex:1; min-width:120px;
}
.metric-pill .val { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; color:#4cc9f0; }
.metric-pill .lbl { font-size:.75rem; color:#5a8aaa; margin-top:3px; }

section[data-testid="stSidebar"] {
    background:#07101d; border-right:1px solid #142236;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛰️ NASA EarthData Explorer</h1>
    <p>Busca · Descarga · Procesa · Visualiza — datos MODIS aerosoles con agentes IA</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — muestra los valores cargados desde secrets (editables en runtime)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    with st.expander("🔑 Credenciales NASA EarthData", expanded=bool(not _ED_USER)):
        ed_user = st.text_input(
            "EARTHDATA_USERNAME",
            value=_ED_USER,
            placeholder="usuario@email.com",
        )
        ed_pass = st.text_input(
            "EARTHDATA_PASSWORD",
            value=_ED_PASS,
            type="password",
        )
        if _ED_USER:
            st.caption("✅ Cargado desde secrets")

    with st.expander("📂 Directorios"):
        hdf_dir    = st.text_input("HDF / Descargas",  value=_HDF_DIR)
        output_dir = st.text_input("CSV output",       value=_OUT_DIR)
        plots_dir  = st.text_input("Gráficas output",  value=_PLT_DIR)

    with st.expander("🗄️ PostgreSQL", expanded=bool(not _DB_URL)):
        db_url = st.text_input(
            "DB URL",
            value=_DB_URL,
            placeholder="postgresql://user:pass@host:5432/db",
            type="password",
        )
        if _DB_URL:
            st.caption("✅ Cargado desde secrets")

    with st.expander("🤖 Modelo LLM (LM Studio)"):
        llm_base_url = st.text_input("Base URL", value=_LLM_URL)
        llm_model    = st.text_input("Modelo",   value=_LLM_MOD)

    variable = st.selectbox(
        "📡 Variable aerosol",
        [
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
        ],
    )

    if st.button("💾 Aplicar configuración", use_container_width=True):
        os.environ["EARTHDATA_USERNAME"] = ed_user
        os.environ["EARTHDATA_PASSWORD"] = ed_pass
        if db_url:
            os.environ["DB_URL"] = db_url
        os.environ["OPENAI_BASE_URL"] = llm_base_url
        os.environ["LLM_MODEL"]       = llm_model
        for d in [hdf_dir, output_dir, plots_dir]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)
        st.success("✅ Listo")


# ─── Helper: correr async desde Streamlit sync ───────────────────────────────
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


# ─── EarthAgent (igual al notebook) ──────────────────────────────────────────
def _build_earth_agent():
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    EARTHACCESS_SERVER = StdioServerParameters(
        command=sys.executable,
        args=["earthaccess_server.py"],
        env={**os.environ},
    )

    model = ChatOpenAI(
        model=llm_model,
        base_url=llm_base_url,
        api_key="not-required",
    )

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]

    async def router(state):
        prompt = (
            "You are a NASA EarthData assistant router. "
            "Analyze the user query and return exactly one word: "
            "'search', 'download', or 'discover'. No explanation."
        )
        msg = await model.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        return {"messages": [msg]}

    def router_decision(state):
        raw = state["messages"][-1].content.lower()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        for opt in ("download", "discover", "search"):
            if opt in raw:
                return opt
        return "search"

    async def _agent(state, prompt_text):
        user_query = state["messages"][-2]
        async with stdio_client(EARTHACCESS_SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                mwt   = model.bind_tools(tools)
                msgs  = [SystemMessage(content=prompt_text), user_query]
                resp  = await mwt.ainvoke(msgs)
                msgs.append(resp)
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        result = await session.call_tool(tc["name"], arguments=tc["args"])
                        msgs.append(ToolMessage(content=result.content[-1].text, tool_call_id=tc["id"]))
                    resp = await mwt.ainvoke(msgs)
                    msgs.append(resp)
                return {"messages": msgs[1:]}

    async def search_agent(state):
        return await _agent(state, (
            "You are a NASA EarthData specialist. Use MCP tools to search granules. "
            "Extract bbox and dates from user request. Present results clearly: "
            "how many granules were found and a brief summary of metadata."
        ))

    async def download_agent(state):
        return await _agent(state, (
            "You are a NASA EarthData download specialist. Use download_granules tool. "
            "Extract concept_id or short_name, bounding box, dates, and output dir. "
            "After downloading, confirm how many files were saved and where."
        ))

    async def discover_agent(state):
        return await _agent(state, (
            "You are a NASA EarthData catalog specialist. Use discover_datasets tool. "
            "Extract keywords and present results as a clear table: "
            "concept_id, short_name, version, provider, title."
        ))

    g = StateGraph(AgentState)
    g.add_node("router",         router)
    g.add_node("search_agent",   search_agent)
    g.add_node("download_agent", download_agent)
    g.add_node("discover_agent", discover_agent)
    g.add_edge(START, "router")
    g.add_edge("search_agent",   END)
    g.add_edge("download_agent", END)
    g.add_edge("discover_agent", END)
    g.add_conditional_edges("router", router_decision, {
        "search":   "search_agent",
        "download": "download_agent",
        "discover": "discover_agent",
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

missing_deps = _check_deps()
if missing_deps:
    st.warning(
        f"⚠️ Dependencias faltantes: `{'`, `'.join(missing_deps)}`\n\n"
        f"```\npip install {' '.join(missing_deps)}\n```"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔍 Buscar granules",
    "⬇️ Descargar",
    "🗂️ Descubrir datasets",
    "⚙️ Pipeline HDF",
    "📊 Gráficas",
    "🗄️ Base de datos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — BUSCAR GRANULES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="step-title">🔍 Buscar granules satelitales</div>', unsafe_allow_html=True)
    st.caption("Llama a `search_by_concept_id` o `search_by_short_name` vía EarthAgent.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        s_concept = st.text_input("Concept ID", "C1443528505-LAADS", key="s_concept",
                                  help="Ej: C1443528505-LAADS para MYD04_3K v6.1")
    with col2:
        s_short = st.text_input("Short name (alternativa)", "MYD04_3K", key="s_short")
    with col3:
        # ✅ FIX: usar keywords explícitos en number_input
        s_max = st.number_input("Máx. granules", min_value=1, max_value=50, value=5, key="s_max")

    st.markdown("**Bounding box** — (oeste, sur, este, norte)")
    b1, b2, b3, b4 = st.columns(4)
    # ✅ FIX: value= siempre como keyword, nunca como segundo argumento posicional
    s_w = b1.number_input("Oeste", value=-10.0, min_value=-180.0, max_value=180.0, key="s_w")
    s_s = b2.number_input("Sur",   value= 20.0, min_value= -90.0, max_value= 90.0, key="s_s")
    s_e = b3.number_input("Este",  value= 10.0, min_value=-180.0, max_value=180.0, key="s_e")
    s_n = b4.number_input("Norte", value= 50.0, min_value= -90.0, max_value= 90.0, key="s_n")

    d1, d2 = st.columns(2)
    s_date1 = d1.date_input("Fecha inicio", value=None, key="s_d1")
    s_date2 = d2.date_input("Fecha fin",    value=None, key="s_d2")

    if st.button("🔍 Buscar", key="btn_search", use_container_width=True):
        id_part   = f"(concept_id {s_concept})" if s_concept else f"(short_name {s_short})"
        date_part = f" desde {s_date1} hasta {s_date2}," if s_date1 and s_date2 else ""
        query = (
            f"Busca datos de aerosoles MODIS {id_part} "
            f"en la región ({s_w}, {s_s}, {s_e}, {s_n})"
            f"{date_part} máximo {s_max} granules."
        )
        with st.spinner("🛰️ Consultando NASA EarthData..."):
            try:
                from langchain_core.messages import HumanMessage
                agent  = _build_earth_agent()
                result = _run(agent.ainvoke({"messages": [HumanMessage(content=query)]}))
                reply  = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"**Error:** {e}")
                st.info("Verifica que `earthaccess_server.py` esté en el mismo directorio y las credenciales sean correctas.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCARGAR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="step-title">⬇️ Descargar granules</div>', unsafe_allow_html=True)
    st.caption("Llama a `download_granules` vía EarthAgent. Los archivos HDF se guardan en el directorio configurado.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        d_concept = st.text_input("Concept ID", "C1443528505-LAADS", key="d_concept")
    with col2:
        d_short   = st.text_input("Short name", "MYD04_3K", key="d_short")
    with col3:
        # ✅ FIX
        d_max = st.number_input("Máx. granules", min_value=1, max_value=20, value=2, key="d_max")

    st.markdown("**Bounding box**")
    b1, b2, b3, b4 = st.columns(4)
    # ✅ FIX
    d_w = b1.number_input("Oeste", value=-10.0, min_value=-180.0, max_value=180.0, key="d_w")
    d_s = b2.number_input("Sur",   value= 20.0, min_value= -90.0, max_value= 90.0, key="d_s")
    d_e = b3.number_input("Este",  value= 10.0, min_value=-180.0, max_value=180.0, key="d_e")
    d_n = b4.number_input("Norte", value= 50.0, min_value= -90.0, max_value= 90.0, key="d_n")

    d1c, d2c = st.columns(2)
    d_date1 = d1c.date_input("Fecha inicio", value=None, key="d_d1")
    d_date2 = d2c.date_input("Fecha fin",    value=None, key="d_d2")

    dl_dir = st.text_input("📁 Directorio destino", value=hdf_dir, key="dl_dir")

    if st.button("⬇️ Descargar", key="btn_dl", use_container_width=True):
        date_part = f" del {d_date1} al {d_date2}" if d_date1 and d_date2 else ""
        query = (
            f"Descarga {d_max} granules de {d_short} ({d_concept}) "
            f"bbox ({d_w}, {d_s}, {d_e}, {d_n}){date_part} "
            f"en {dl_dir}"
        )
        with st.spinner("⬇️ Descargando datos satelitales..."):
            try:
                from langchain_core.messages import HumanMessage
                agent  = _build_earth_agent()
                result = _run(agent.ainvoke({"messages": [HumanMessage(content=query)]}))
                reply  = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>', unsafe_allow_html=True)

                dl_path   = Path(dl_dir).expanduser()
                hdf_files = list(dl_path.glob("*.hdf")) + list(dl_path.glob("*.HDF"))
                if hdf_files:
                    st.success(f"✅ {len(hdf_files)} archivos HDF en `{dl_path}`")
                    with st.expander("Ver archivos descargados"):
                        for f in sorted(hdf_files):
                            st.text(f"📄 {f.name}  —  {f.stat().st_size/1024**2:.1f} MB")
            except Exception as e:
                st.error(f"**Error:** {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCUBRIR DATASETS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="step-title">🗂️ Descubrir colecciones NASA CMR</div>', unsafe_allow_html=True)
    st.caption("Llama a `discover_datasets`. Devuelve concept_id, short_name, versión, proveedor y título.")

    disc_kw = st.text_input(
        "🔑 Palabra clave", "aerosol",
        placeholder="aerosol · sea surface temperature · NDVI · precipitation …",
    )

    if st.button("🗂️ Descubrir datasets", key="btn_discover", use_container_width=True):
        with st.spinner("🔎 Consultando catálogo NASA CMR..."):
            try:
                from langchain_core.messages import HumanMessage
                agent  = _build_earth_agent()
                result = _run(agent.ainvoke({"messages": [
                    HumanMessage(content=f"Encuentra datasets de NASA EarthData relacionados con {disc_kw}.")
                ]}))
                reply = result["messages"][-1].content
                st.markdown(f'<div class="result-box">{reply}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"**Error:** {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE HDF
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="step-title">⚙️ Pipeline HDF → CSV → PostgreSQL → Gráficas</div>', unsafe_allow_html=True)
    st.info(
        "**Paso 1** HDF4 → CSV via `pyhdf`  ·  "
        "**Paso 2** CSV → PostgreSQL  ·  "
        "**Paso 3** Gráficas 2D (scatter map · heatmap · time series)"
    )

    pc1, pc2 = st.columns(2)
    with pc1:
        p_hdf   = st.text_input("📁 Dir. HDF",   value=hdf_dir,    key="p_hdf")
        p_out   = st.text_input("📂 Dir. CSV",   value=output_dir, key="p_out")
    with pc2:
        p_plots = st.text_input("🖼️ Dir. plots", value=plots_dir,  key="p_plots")
        p_db    = st.text_input("🗄️ DB URL",     value=db_url,     type="password", key="p_db")

    p_var = st.selectbox("Variable aerosol", [
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
    skip_convert = sk1.checkbox("⏭️ Saltar HDF→CSV")
    skip_load    = sk2.checkbox("⏭️ Saltar carga BD")
    skip_plot    = sk3.checkbox("⏭️ Saltar gráficas")

    p_mode = st.radio(
        "Modo de ejecución",
        ["⚡ Directo — run_pipeline()", "🤖 Agentes — PipelineAgent (LangGraph + MCP)"],
        horizontal=True,
    )

    if st.button("▶️ Ejecutar pipeline", key="btn_pipe", use_container_width=True):
        for d in [p_hdf, p_out, p_plots]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)

        log_placeholder = st.empty()
        log_lines: list[str] = []

        class StreamlitHandler(logging.Handler):
            def emit(self, record):
                log_lines.append(self.format(record))
                log_placeholder.markdown(
                    '<div class="result-box">' + "<br>".join(log_lines[-30:]) + "</div>",
                    unsafe_allow_html=True,
                )

        pipe_logger = logging.getLogger("pipe_run")
        pipe_logger.setLevel(logging.INFO)
        sh = StreamlitHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        pipe_logger.addHandler(sh)

        with st.spinner("⚙️ Ejecutando pipeline..."):
            try:
                if "Directo" in p_mode:
                    from hdf_pipeline import run_pipeline
                    results = run_pipeline(
                        hdf_dir=p_hdf,
                        output_dir=p_out,
                        plots_dir=p_plots,
                        db_url=p_db,
                        variable=p_var,
                        skip_convert=skip_convert,
                        skip_load=skip_load,
                        skip_plot=skip_plot,
                        logger=pipe_logger,
                    )
                else:
                    from pipeline_agent import PipelineAgent, run_pipeline_agents
                    pa = PipelineAgent(model_name=llm_model)
                    results = _run(run_pipeline_agents(
                        hdf_dir=p_hdf,
                        output_dir=p_out,
                        plots_dir=p_plots,
                        db_url=p_db,
                        variable=p_var,
                        agent=pa,
                    ))

                csvs_n = len(results.get("csvs",  []))
                rows_n = results.get("rows_loaded", 0)
                plot_n = len(results.get("plots", []))

                st.markdown(f"""
<div class="metric-row">
  <div class="metric-pill"><div class="val">{csvs_n}</div><div class="lbl">CSVs generados</div></div>
  <div class="metric-pill"><div class="val">{rows_n:,}</div><div class="lbl">Filas en BD</div></div>
  <div class="metric-pill"><div class="val">{plot_n}</div><div class="lbl">Gráficas</div></div>
</div>""", unsafe_allow_html=True)

                if results.get("plots"):
                    st.markdown("#### Gráficas generadas")
                    cols = st.columns(min(3, len(results["plots"])))
                    for i, p in enumerate(results["plots"]):
                        cols[i % 3].image(str(p), caption=Path(p).name, use_container_width=True)

                if results.get("messages") and "Agentes" in p_mode:
                    with st.expander("📋 Mensajes de los agentes"):
                        for m in results["messages"]:
                            st.text(m.content if hasattr(m, "content") else str(m))

                st.session_state["last_plots"] = results.get("plots", [])

            except ImportError as e:
                st.error(f"Módulo no encontrado: **{e}**")
                st.info("Asegúrate de que `hdf_pipeline.py` y `pipeline_agent.py` estén en el directorio.")
            except Exception as e:
                st.error(f"**Error en pipeline:** {e}")
            finally:
                pipe_logger.removeHandler(sh)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="step-title">📊 Visualización de aerosoles</div>', unsafe_allow_html=True)

    st.button("🔄 Actualizar desde disco", key="btn_refresh")

    plots_path = Path(plots_dir).expanduser()
    disk_pngs  = sorted(plots_path.glob("*.png")) if plots_path.exists() else []
    last_plots = [Path(p) for p in st.session_state.get("last_plots", [])]
    all_pngs   = list({str(p): p for p in disk_pngs + last_plots}.values())

    if not all_pngs:
        st.info(f"Sin gráficas en `{plots_path}`. Ejecuta el pipeline primero.")
    else:
        st.caption(f"{len(all_pngs)} imagen(es) en `{plots_path}`")
        cols = st.columns(2)
        for i, png in enumerate(sorted(all_pngs, key=lambda p: str(p))):
            cols[i % 2].image(str(png), caption=Path(png).name, use_container_width=True)

    st.divider()
    st.markdown("#### Generar gráficas directamente desde PostgreSQL")
    st.caption("Usa `AerosolPlotter` de `hdf_pipeline.py`")

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
        g_res = st.number_input("Resolución heatmap (°)", value=0.5, key="g_res")
        g_db  = st.text_input("DB URL", value=db_url, type="password", key="g_db")

    g_types = st.multiselect(
        "Tipos de gráfica",
        ["scatter_map", "heatmap", "time_series"],
        default=["scatter_map", "heatmap", "time_series"],
        key="g_types",
    )

    if st.button("📊 Generar desde BD", key="btn_gen", use_container_width=True):
        with st.spinner("Generando gráficas..."):
            try:
                from hdf_pipeline import AerosolPlotter, PostgresLoader
                loader  = PostgresLoader(db_url=g_db)
                plotter = AerosolPlotter(loader=loader, plots_dir=plots_dir)
                generated = []
                if "scatter_map" in g_types:
                    p = plotter.scatter_map(variable=g_var, vmin=g_vmin, vmax=g_vmax)
                    if p: generated.append(p)
                if "heatmap" in g_types:
                    p = plotter.heatmap(variable=g_var, resolution=g_res, vmin=g_vmin, vmax=g_vmax)
                    if p: generated.append(p)
                if "time_series" in g_types:
                    p = plotter.time_series(variable=g_var)
                    if p: generated.append(p)

                if generated:
                    cols = st.columns(min(3, len(generated)))
                    for i, png in enumerate(generated):
                        cols[i % 3].image(str(png), caption=png.name, use_container_width=True)
                    st.session_state["last_plots"] = [str(p) for p in generated]
                else:
                    st.warning("No se generaron gráficas — BD vacía o sin datos para esa variable.")
            except ImportError:
                st.error("`hdf_pipeline.py` no encontrado.")
            except Exception as e:
                st.error(f"**Error:** {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BASE DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="step-title">🗄️ PostgreSQL — aerosol_data</div>', unsafe_allow_html=True)
    st.caption("Usa `PostgresLoader.summary()` de `hdf_pipeline.py` y consultas directas.")

    db_conn = st.text_input("DB URL", value=db_url, type="password", key="db_conn")

    col_sum, col_sql = st.columns(2)

    with col_sum:
        st.markdown("**Resumen estadístico**")
        if st.button("📋 Ver resumen", key="btn_summary", use_container_width=True):
            with st.spinner("Consultando..."):
                try:
                    from hdf_pipeline import PostgresLoader
                    loader = PostgresLoader(db_url=db_conn)
                    df = loader.summary()
                    if df.empty:
                        st.info("La tabla `aerosol_data` está vacía.")
                    else:
                        st.dataframe(df, use_container_width=True)
                        if "puntos" in df.columns:
                            total_pts = int(df["puntos"].sum())
                            st.markdown(
                                f'<div class="metric-row">'
                                f'<div class="metric-pill"><div class="val">{len(df)}</div>'
                                f'<div class="lbl">Variables</div></div>'
                                f'<div class="metric-pill"><div class="val">{total_pts:,}</div>'
                                f'<div class="lbl">Total puntos</div></div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                except ImportError:
                    st.error("`hdf_pipeline.py` no encontrado.")
                except Exception as e:
                    st.error(f"**Error:** {e}")

    with col_sql:
        st.markdown("**Consulta SQL personalizada**")
        sql = st.text_area(
            "SQL",
            "SELECT variable, COUNT(*) AS n FROM aerosol_data GROUP BY variable ORDER BY n DESC;",
            height=130,
            key="sql",
        )
        if st.button("▶️ Ejecutar SQL", key="btn_sql", use_container_width=True):
            with st.spinner("Ejecutando..."):
                try:
                    import pandas as pd
                    import psycopg2
                    conn = psycopg2.connect(db_conn)
                    df   = pd.read_sql(sql, conn)
                    conn.close()
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} fila(s) devuelta(s)")
                except Exception as e:
                    st.error(f"**Error SQL:** {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#2a5a8a;font-size:.77rem;font-family:IBM Plex Mono,monospace;'>"
    "NASA EarthData Explorer · MODIS MYD04_3K Aerosoles · LangGraph + MCP + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)