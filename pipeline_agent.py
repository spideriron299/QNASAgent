# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Pipeline Agent — orquesta HDF4 → CSV → PostgreSQL → Gráficas via agentes LangGraph.

Reemplaza la llamada directa a run_pipeline() con un grafo de 3 agentes especializados:
  - convert_agent   → llama a convert_hdf_to_csv
  - load_agent      → llama a load_csvs_to_postgres
  - plot_agent      → llama a generate_plots (+ query_db_summary)

El PipelineOrchestrator ejecuta los 3 en secuencia pasando el contexto entre ellos,
tal como hacía run_pipeline(), pero ahora cada paso es un nodo autónomo del grafo.
"""

import os
import sys
import re
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del servidor MCP del pipeline
# ─────────────────────────────────────────────────────────────────────────────

HDF_PIPELINE_SERVER = StdioServerParameters(
    command=sys.executable,
    args=["hdf_pipeline_server.py"],
    env={**os.environ},
)


# ─────────────────────────────────────────────────────────────────────────────
# Estado compartido entre todos los nodos del grafo
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    messages:   Annotated[list[AnyMessage], operator.add]
    # Contexto acumulado que se pasa de nodo a nodo
    hdf_dir:    str
    output_dir: str
    plots_dir:  str
    db_url:     str
    variable:   str
    # Resultados de cada etapa (se van llenando)
    csvs:       list[str]
    rows_loaded: int
    plots:      list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agente base: lógica reutilizable para invocar un tool MCP y obtener respuesta
# ─────────────────────────────────────────────────────────────────────────────

async def _run_mcp_agent(
    model: ChatOpenAI,
    system_prompt: str,
    user_message: AnyMessage,
) -> list[AnyMessage]:
    """
    Abre una sesión con hdf_pipeline_server, vincula los tools al modelo,
    invoca el modelo con el mensaje del usuario y ejecuta los tool_calls resultantes.
    Devuelve la lista de mensajes intercambiados (sin el SystemMessage inicial).
    """
    async with stdio_client(HDF_PIPELINE_SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model_with_tools = model.bind_tools(tools)

            messages = [SystemMessage(content=system_prompt), user_message]
            response = await model_with_tools.ainvoke(messages)
            messages.append(response)

            if response.tool_calls:
                for tc in response.tool_calls:
                    result = await session.call_tool(tc["name"], arguments=tc["args"])
                    messages.append(ToolMessage(
                        content=result.content[-1].text,
                        tool_call_id=tc["id"],
                    ))
                # Segunda invocación para que el modelo procese los resultados
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)

            # Devolvemos todo menos el SystemMessage
            return messages[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: PipelineAgent
# ─────────────────────────────────────────────────────────────────────────────

class PipelineAgent:
    """
    Agente LangGraph que ejecuta el pipeline HDF → CSV → PostgreSQL → Gráficas
    usando 3 nodos especializados que se comunican vía MCP con hdf_pipeline_server.py.

    Flujo:
        START → convert_agent → load_agent → plot_agent → END

    Uso:
        agent = PipelineAgent()
        result = await agent.graph.ainvoke({
            "messages": [HumanMessage(content="Procesa los archivos en ~/Downloads/earthdata")],
            "hdf_dir":    "/home/user/Downloads/earthdata",
            "output_dir": "~/aerosol_csv",
            "plots_dir":  "~/aerosol_plots",
            "db_url":     "postgresql://...",
            "variable":   "Optical_Depth_Land_And_Ocean",
            "csvs":       [],
            "rows_loaded": 0,
            "plots":      [],
        })
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model = ChatOpenAI(
            model=model_name,
            base_url="http://localhost:8000/v1",
            api_key="not-required",
        )

        graph = StateGraph(PipelineState)

        graph.add_node("convert_agent", self.convert_agent)
        graph.add_node("load_agent",    self.load_agent)
        graph.add_node("plot_agent",    self.plot_agent)

        # Secuencia fija: conversión → carga → gráficas
        graph.add_edge(START,           "convert_agent")
        graph.add_edge("convert_agent", "load_agent")
        graph.add_edge("load_agent",    "plot_agent")
        graph.add_edge("plot_agent",    END)

        self.graph = graph.compile()

    # ------------------------------------------------------------------
    # Nodo 1 — Convertir HDF4 → CSV
    # ------------------------------------------------------------------

    async def convert_agent(self, state: PipelineState) -> dict:
        """
        Invoca convert_hdf_to_csv con el directorio HDF del estado.
        Extrae la lista de CSVs generados para pasarla al siguiente nodo.
        """
        prompt = f"""Eres un especialista en conversión de datos satelitales HDF4.
Usa la herramienta convert_hdf_to_csv para convertir los archivos HDF del directorio
'{state['hdf_dir']}' a CSV en '{state['output_dir']}'.

Después de llamar la herramienta, reporta:
- Cuántos archivos HDF encontraste
- Cuántos CSV se generaron correctamente
- Si hubo errores, cuáles fueron
"""
        user_msg = HumanMessage(content=(
            f"Convierte los archivos HDF de '{state['hdf_dir']}' a CSV "
            f"y guárdalos en '{state['output_dir']}'."
        ))

        msgs = await _run_mcp_agent(self.model, prompt, user_msg)

        # Extraer rutas de CSVs del ToolMessage (resultado JSON del tool)
        csvs = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                try:
                    import json
                    data = json.loads(m.content)
                    csvs = [item["file"] for item in data.get("csvs", [])]
                except Exception:
                    pass

        return {
            "messages":   msgs,
            "csvs":       csvs,
        }

    # ------------------------------------------------------------------
    # Nodo 2 — Cargar CSVs a PostgreSQL
    # ------------------------------------------------------------------

    async def load_agent(self, state: PipelineState) -> dict:
        """
        Invoca load_csvs_to_postgres. Recibe los CSVs del nodo anterior
        vía state['output_dir'] y reporta cuántas filas se insertaron.
        """
        prompt = f"""Eres un especialista en carga de datos a PostgreSQL.
Usa la herramienta load_csvs_to_postgres para cargar los CSV del directorio
'{state['output_dir']}' a la base de datos.

Los CSV fueron generados en el paso anterior ({len(state.get('csvs', []))} archivos).

Reporta:
- Cuántos CSV cargaste
- Total de filas insertadas
- Resumen estadístico por variable si está disponible
"""
        user_msg = HumanMessage(content=(
            f"Carga todos los CSV de '{state['output_dir']}' "
            f"a la base de datos en '{state['db_url']}'."
        ))

        msgs = await _run_mcp_agent(self.model, prompt, user_msg)

        # Extraer rows_loaded
        rows_loaded = 0
        for m in msgs:
            if isinstance(m, ToolMessage):
                try:
                    import json
                    data = json.loads(m.content)
                    rows_loaded = data.get("rows_loaded", 0)
                except Exception:
                    pass

        return {
            "messages":    msgs,
            "rows_loaded": rows_loaded,
        }

    # ------------------------------------------------------------------
    # Nodo 3 — Generar gráficas 2D
    # ------------------------------------------------------------------

    async def plot_agent(self, state: PipelineState) -> dict:
        """
        Invoca generate_plots (scatter, heatmap, timeseries) y opcionalmente
        query_db_summary para verificar qué variables están disponibles antes.
        """
        prompt = f"""Eres un especialista en visualización de datos de aerosoles satelitales.

Primero usa query_db_summary para ver qué variables están disponibles en la BD.
Luego usa generate_plots con:
  - variable: '{state['variable']}'
  - plot_types: 'all'  (scatter_map, heatmap y time_series)
  - plots_dir: '{state['plots_dir']}'

Reporta:
- Cuántas gráficas se generaron
- Las rutas de cada imagen
- Cualquier advertencia sobre datos faltantes
"""
        user_msg = HumanMessage(content=(
            f"Genera las gráficas de aerosoles para la variable '{state['variable']}' "
            f"desde la base de datos y guárdalas en '{state['plots_dir']}'."
        ))

        msgs = await _run_mcp_agent(self.model, prompt, user_msg)

        # Extraer rutas de gráficas
        plots = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                try:
                    import json
                    data = json.loads(m.content)
                    plots = [item["path"] for item in data.get("plots", [])]
                except Exception:
                    pass

        return {
            "messages": msgs,
            "plots":    plots,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Función de conveniencia — equivalente a run_pipeline() pero con agentes
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_agents(
    hdf_dir:    str = "/home/user/Downloads/earthdata",
    output_dir: str = "~/aerosol_csv",
    plots_dir:  str = "~/aerosol_plots",
    db_url:     str = "",
    variable:   str = "Optical_Depth_Land_And_Ocean",
    agent:      "PipelineAgent | None" = None,
) -> dict:
    """
    Punto de entrada equivalente a run_pipeline() del pipeline original,
    pero orquestado por el PipelineAgent (LangGraph + MCP).

    Args:
        hdf_dir:    Directorio con los archivos HDF descargados.
        output_dir: Directorio para los CSV generados.
        plots_dir:  Directorio para las gráficas PNG.
        db_url:     URL de conexión PostgreSQL.
        variable:   Variable de aerosol a procesar.
        agent:      Instancia de PipelineAgent (se crea una si no se pasa).

    Returns:
        dict con claves 'csvs', 'rows_loaded', 'plots' y 'messages'.
    """
    if agent is None:
        agent = PipelineAgent()

    initial_state: PipelineState = {
        "messages":    [HumanMessage(content=(
            f"Procesa los archivos HDF de '{hdf_dir}', "
            f"cárgalos en la BD y genera gráficas de '{variable}'."
        ))],
        "hdf_dir":     hdf_dir,
        "output_dir":  output_dir,
        "plots_dir":   plots_dir,
        "db_url":      db_url,
        "variable":    variable,
        "csvs":        [],
        "rows_loaded": 0,
        "plots":       [],
    }

    result = await agent.graph.ainvoke(initial_state)

    return {
        "csvs":        result.get("csvs",        []),
        "rows_loaded": result.get("rows_loaded",  0),
        "plots":       result.get("plots",        []),
        "messages":    result.get("messages",     []),
    }