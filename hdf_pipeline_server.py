# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
MCP Server para el pipeline HDF4 -> CSV -> PostgreSQL -> Graficas.

Expone las 3 etapas del pipeline original como herramientas MCP independientes,
permitiendo que un agente LangGraph las orqueste de forma inteligente.

Dependencias:
    pyhdf, numpy, pandas, matplotlib, psycopg2-binary, mcp
"""

# ============================================================================
# CRITICO: PROTEGER STDOUT DURANTE TODA LA EJECUCION
#
# hdf_pipeline tiene print() que contaminan stdout tanto al importar
# como durante la ejecucion de los tools (ej: "Tabla aerosol_data lista.")
#
# Solucion definitiva:
#   1. Duplicar fd1 -> _mcp_fd   (fd privado para JSON-RPC)
#   2. Redirigir fd1 -> stderr   (todos los print() van a stderr SIEMPRE)
#   3. Reemplazar sys.stdout con un wrapper al _mcp_fd
#      FastMCP usa sys.stdout internamente, asi recibe el fd correcto
# ============================================================================

import io
import os
import sys

# 1. Guardar copia del stdout real del SO
_mcp_fd = os.dup(1)

# 2. Redirigir fd1 -> stderr (TODOS los print() van aqui, para siempre)
os.dup2(2, 1)

# 3. sys.stdout apunta al fd real guardado -> FastMCP escribe JSON-RPC ahi
sys.stdout = io.TextIOWrapper(
    io.FileIO(_mcp_fd, mode="w", closefd=False),
    encoding="utf-8",
    line_buffering=True,
)

# ============================================================================
# IMPORTS (ya seguros: cualquier print() de hdf_pipeline va a stderr)
# ============================================================================

import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP + PIPELINE
# ============================================================================

from mcp.server.fastmcp import FastMCP

from hdf_pipeline import (
    HdfToCsvConverter,
    PostgresLoader,
    AerosolPlotter,
    AEROSOL_VARS_2D,
)

mcp = FastMCP("HDF Pipeline Server")


# ============================================================================
# TOOL 1: HDF4 -> CSV
# ============================================================================

@mcp.tool()
def convert_hdf_to_csv(
    hdf_dir: str,
    output_dir: str = "~/aerosol_csv",
) -> str:
    """
    Convierte archivos HDF4 (.hdf / .he4) a CSV.

    Args:
        hdf_dir:    Directorio con los archivos HDF4.
        output_dir: Directorio donde se guardan los CSV generados.

    Returns:
        JSON con status, lista de CSVs generados y conteo de filas.
    """
    try:
        converter = HdfToCsvConverter(output_dir=output_dir)
        hdf_path = Path(hdf_dir).expanduser()

        if not hdf_path.exists():
            return json.dumps({
                "status": "error",
                "message": "Directorio no encontrado: " + hdf_dir,
                "csvs": [],
                "count": 0,
            }, indent=2)

        hdf_files = (
            list(hdf_path.glob("*.hdf")) +
            list(hdf_path.glob("*.he4"))
        )

        if not hdf_files:
            return json.dumps({
                "status": "no_files",
                "message": "No se encontraron archivos HDF en " + hdf_dir,
                "csvs": [],
                "count": 0,
            }, indent=2)

        logger.info("Convirtiendo %d archivos HDF...", len(hdf_files))
        csv_paths = converter.convert_directory(hdf_dir)

        results = []
        for p in csv_paths:
            try:
                import pandas as pd
                df = pd.read_csv(p)
                results.append({
                    "file": str(p),
                    "rows": len(df),
                    "variables": (
                        df["variable"].unique().tolist()
                        if "variable" in df.columns
                        else []
                    ),
                })
            except Exception as e:
                logger.exception("Error leyendo CSV %s", p)
                results.append({
                    "file": str(p),
                    "rows": -1,
                    "variables": [],
                    "error": str(e),
                })

        return json.dumps({
            "status": "success",
            "hdf_dir": str(hdf_path),
            "output_dir": str(converter.output_dir),
            "hdf_files_found": len(hdf_files),
            "csvs_generated": len(csv_paths),
            "csvs": results,
        }, indent=2)

    except Exception as e:
        logger.exception("Error en convert_hdf_to_csv")
        return json.dumps({
            "status": "error",
            "message": str(e),
        }, indent=2)


# ============================================================================
# TOOL 2: CSV -> PostgreSQL
# ============================================================================

@mcp.tool()
def load_csvs_to_postgres(
    db_url: str,
    csv_dir: str = "~/aerosol_csv",
    batch_size: int = 1000,
) -> str:
    """
    Carga todos los CSV de un directorio a PostgreSQL.

    Args:
        db_url:     URL de conexion PostgreSQL.
        csv_dir:    Directorio con los archivos CSV.
        batch_size: Filas por batch de insercion.

    Returns:
        JSON con status, filas cargadas y resumen estadistico.
    """
    try:
        csv_path = Path(csv_dir).expanduser()
        csv_files = list(csv_path.glob("*.csv"))

        if not csv_files:
            return json.dumps({
                "status": "no_csvs",
                "message": "No se encontraron CSV en " + csv_dir,
                "rows_loaded": 0,
            }, indent=2)

        logger.info("Cargando %d CSVs...", len(csv_files))
        loader = PostgresLoader(db_url=db_url)
        loader.init_schema()
        total_rows = loader.load_all_csvs(csv_files, batch_size=batch_size)

        try:
            summary = loader.summary().to_dict(orient="records")
        except Exception as e:
            logger.exception("Error generando resumen")
            summary = [{"warning": "No se pudo obtener resumen: " + str(e)}]

        return json.dumps({
            "status": "success",
            "csv_dir": str(csv_path),
            "csvs_loaded": len(csv_files),
            "rows_loaded": total_rows,
            "db_summary": summary,
        }, indent=2)

    except Exception as e:
        logger.exception("Error en load_csvs_to_postgres")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "rows_loaded": 0,
        }, indent=2)


# ============================================================================
# TOOL 3: GRAFICAS
# ============================================================================

@mcp.tool()
def generate_plots(
    db_url: str,
    variable: str = "Optical_Depth_Land_And_Ocean",
    plots_dir: str = "~/aerosol_plots",
    plot_types: str = "all",
    vmin: float = 0.0,
    vmax: float = 1.5,
    heatmap_resolution: float = 0.5,
) -> str:
    """
    Genera graficas de aerosoles desde PostgreSQL.

    Args:
        db_url:              URL de conexion PostgreSQL.
        variable:            Variable a graficar.
        plots_dir:           Directorio donde se guardan las imagenes PNG.
        plot_types:          Tipos a generar: 'all', 'scatter', 'heatmap', 'timeseries'.
        vmin:                Valor minimo de escala de color.
        vmax:                Valor maximo de escala de color.
        heatmap_resolution:  Resolucion del grid del heatmap en grados.

    Returns:
        JSON con lista de graficas generadas y errores si los hay.
    """
    try:
        loader = PostgresLoader(db_url=db_url)
        plotter = AerosolPlotter(loader=loader, plots_dir=plots_dir)
        generated = []
        errors = []
        pt = plot_types.lower()

        logger.info("Generando plots variable=%s tipo=%s", variable, pt)

        if pt in ("all", "scatter"):
            p = plotter.scatter_map(variable=variable, vmin=vmin, vmax=vmax)
            if p:
                generated.append({"type": "scatter_map", "path": str(p)})
            else:
                errors.append("scatter_map: sin datos")

        if pt in ("all", "heatmap"):
            p = plotter.heatmap(
                variable=variable,
                resolution=heatmap_resolution,
                vmin=vmin,
                vmax=vmax,
            )
            if p:
                generated.append({"type": "heatmap", "path": str(p)})
            else:
                errors.append("heatmap: sin datos")

        if pt in ("all", "timeseries"):
            p = plotter.time_series(variable=variable)
            if p:
                generated.append({"type": "time_series", "path": str(p)})
            else:
                errors.append("time_series: sin datos")

        return json.dumps({
            "status": "success" if generated else "no_plots",
            "variable": variable,
            "plots_dir": str(plotter.plots_dir),
            "plots_generated": len(generated),
            "plots": generated,
            "errors": errors,
        }, indent=2)

    except Exception as e:
        logger.exception("Error en generate_plots")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "plots_generated": 0,
        }, indent=2)


# ============================================================================
# TOOL 4: RESUMEN BD
# ============================================================================

@mcp.tool()
def query_db_summary(db_url: str) -> str:
    """
    Consulta resumen estadistico de la tabla aerosol_data.

    Args:
        db_url: URL de conexion PostgreSQL.

    Returns:
        JSON con resumen por variable (count, min, max, mean).
    """
    try:
        loader = PostgresLoader(db_url=db_url)
        df = loader.summary()

        if df.empty:
            return json.dumps({
                "status": "empty",
                "message": "La tabla aerosol_data existe pero esta vacia.",
                "summary": [],
            }, indent=2)

        return json.dumps({
            "status": "success",
            "variables_count": len(df),
            "summary": df.to_dict(orient="records"),
        }, indent=2)

    except Exception as e:
        logger.exception("Error en query_db_summary")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "summary": [],
        }, indent=2)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Iniciando HDF Pipeline MCP Server...")
    # sys.stdout ya apunta al _mcp_fd (stdout real del SO)
    # fd1 permanece -> stderr, por lo que ningun print() contamina JSON-RPC
    mcp.run(transport="stdio")