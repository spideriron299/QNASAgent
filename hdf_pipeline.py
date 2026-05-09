# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
hdf_pipeline.py  —  HDF4 (MYD04_3K / MOD04_3K)  →  CSV  →  PostgreSQL  →  Gráficas 2D
Requiere: pyhdf, numpy, pandas, matplotlib, psycopg2-binary
"""

from __future__ import annotations

import logging
import warnings
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psycopg2
from psycopg2.extras import execute_values

try:
    from tqdm.notebook import tqdm as tqdm_notebook
    from tqdm import tqdm as tqdm_terminal
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from pyhdf.SD import SD, SDC
    PYHDF_AVAILABLE = True
except ImportError:
    PYHDF_AVAILABLE = False
    warnings.warn("pyhdf no disponible — instala con:  pip install pyhdf")

# ── Logger por defecto (silencioso a menos que el llamador configure handlers) ──
_default_logger = logging.getLogger(__name__)
_default_logger.addHandler(logging.NullHandler())


def _make_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Devuelve el logger provisto o el módulo-level por defecto."""
    return logger if logger is not None else _default_logger


# Variables 2D de aerosol que existen en MYD04_3K
AEROSOL_VARS_2D = [
    "Optical_Depth_Land_And_Ocean",
    "Image_Optical_Depth_Land_And_Ocean",
    "Corrected_Optical_Depth_Land_wav2p1",
    "Optical_Depth_Ratio_Small_Land",
    "Angstrom_Exponent_1_Ocean",
    "Angstrom_Exponent_2_Ocean",
    "Optical_Depth_Ratio_Small_Ocean_0.55micron",
    "Mass_Concentration_Land",
    "Aerosol_Cloud_Fraction_Land",
    "Aerosol_Cloud_Fraction_Ocean",
    "Fitting_Error_Land",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSOR HDF4 → CSV
# ─────────────────────────────────────────────────────────────────────────────

class HdfToCsvConverter:

    def __init__(
        self,
        output_dir: str = "~/aerosol_csv",
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = _make_logger(logger)

    def convert_directory(self, hdf_dir: str) -> list[Path]:
        hdf_dir = Path(hdf_dir).expanduser()
        files   = list(hdf_dir.glob("*.hdf")) + list(hdf_dir.glob("*.he4"))
        if not files:
            self.log.warning("No se encontraron archivos HDF en %s", hdf_dir)
            return []

        csv_paths = []
        for f in files:
            self.log.info("Procesando: %s", f.name)
            try:
                p = self._convert_file(f)
                if p:
                    csv_paths.append(p)
                    self.log.info("  → %s", p.name)
                else:
                    self.log.warning("  Sin datos extraíbles")
            except Exception as e:
                self.log.error("  %s: %s", f.name, e)

        self.log.info("%d CSV generados en %s", len(csv_paths), self.output_dir)
        return csv_paths

    def convert_file(self, path: str) -> Optional[Path]:
        return self._convert_file(Path(path).expanduser())

    def _convert_file(self, path: Path) -> Optional[Path]:
        if not PYHDF_AVAILABLE:
            raise RuntimeError("pyhdf no disponible. Instala con: pip install pyhdf")

        hdf  = SD(str(path), SDC.READ)
        lats = self._read_scaled(hdf, "Latitude").flatten()
        lons = self._read_scaled(hdf, "Longitude").flatten()

        frames    = []
        available = hdf.datasets()

        for var in AEROSOL_VARS_2D:
            if var not in available:
                continue
            try:
                data = self._read_scaled(hdf, var)
                if data.ndim == 3:
                    data = data[0]
                flat = data.flatten()
                n    = min(len(flat), len(lats), len(lons))

                mask = (
                    np.isfinite(flat[:n])
                    & np.isfinite(lats[:n])
                    & (lats[:n] >= -90)  & (lats[:n] <= 90)
                    & (lons[:n] >= -180) & (lons[:n] <= 180)
                )

                df = pd.DataFrame({
                    "lat":      lats[:n][mask],
                    "lon":      lons[:n][mask],
                    "variable": var,
                    "value":    flat[:n][mask],
                })
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                self.log.warning("    %s: %s", var, e)

        hdf.end()

        if not frames:
            return None

        result = pd.concat(frames, ignore_index=True)
        result["filename"]  = path.name
        result["timestamp"] = self._extract_timestamp(path.name)

        out = self.output_dir / (path.stem + ".csv")
        result.to_csv(out, index=False)
        return out

    @staticmethod
    def _read_scaled(hdf, var_name: str) -> np.ndarray:
        sds   = hdf.select(var_name)
        data  = sds.get().astype(float)
        attrs = sds.attributes()

        fill = attrs.get("_FillValue")
        if fill is not None:
            data[data == float(fill)] = np.nan

        scale  = float(attrs.get("scale_factor", 1.0))
        offset = float(attrs.get("add_offset",   0.0))
        data   = data * scale + offset
        sds.endaccess()
        return data

    @staticmethod
    def _extract_timestamp(filename: str) -> Optional[str]:
        m = re.search(r"\.A(\d{4})(\d{3})\.", filename)
        if m:
            try:
                dt = datetime(int(m.group(1)), 1, 1) + pd.Timedelta(days=int(m.group(2)) - 1)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGADOR POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────

class PostgresLoader:

    DDL = """
    CREATE TABLE IF NOT EXISTS aerosol_data (
        id        SERIAL PRIMARY KEY,
        filename  TEXT,
        variable  TEXT,
        lat       DOUBLE PRECISION,
        lon       DOUBLE PRECISION,
        value     DOUBLE PRECISION,
        timestamp DATE,
        loaded_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_aerosol_latlon    ON aerosol_data (lat, lon);
    CREATE INDEX IF NOT EXISTS idx_aerosol_variable  ON aerosol_data (variable);
    CREATE INDEX IF NOT EXISTS idx_aerosol_timestamp ON aerosol_data (timestamp);
    """

    def __init__(
        self,
        db_url: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.db_url = db_url
        self._conn  = None
        self.log    = _make_logger(logger)

    def connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
        return self._conn

    def init_schema(self):
        with self.connect().cursor() as cur:
            cur.execute(self.DDL)
        self.log.info("Tabla aerosol_data lista.")

    def load_csv(self, csv_path: Path, batch_size: int = 1000) -> int:
        df = pd.read_csv(csv_path).dropna(subset=["value"])
        if df.empty:
            return 0

        rows = [
            (
                row["filename"], row["variable"],
                float(row["lat"])       if pd.notna(row["lat"])       else None,
                float(row["lon"])       if pd.notna(row["lon"])       else None,
                float(row["value"]),
                row.get("timestamp")    if pd.notna(row.get("timestamp")) else None,
            )
            for _, row in df.iterrows()
        ]

        n_batches = (len(rows) + batch_size - 1) // batch_size

        if TQDM_AVAILABLE:
            try:
                get_ipython()           # noqa: F821  — sólo existe en Jupyter
                progress = tqdm_notebook
            except NameError:
                progress = tqdm_terminal
        else:
            progress = None

        with self.connect().cursor() as cur:
            if progress:
                bar = progress(
                    range(0, len(rows), batch_size),
                    total=n_batches,
                    desc=f"  {csv_path.stem[:35]}",
                    unit="lote",
                    colour="green",
                )
                for i in bar:
                    execute_values(
                        cur,
                        "INSERT INTO aerosol_data (filename,variable,lat,lon,value,timestamp) VALUES %s",
                        rows[i : i + batch_size],
                    )
                    bar.set_postfix({"filas": f"{min(i+batch_size, len(rows)):,}/{len(rows):,}"})
            else:
                for i in range(0, len(rows), batch_size):
                    execute_values(
                        cur,
                        "INSERT INTO aerosol_data (filename,variable,lat,lon,value,timestamp) VALUES %s",
                        rows[i : i + batch_size],
                    )
                    self.log.debug(
                        "  %s/%s filas insertadas",
                        min(i + batch_size, len(rows)),
                        len(rows),
                    )

        return len(rows)

    def load_all_csvs(self, csv_paths: list[Path], batch_size: int = 1000) -> int:
        self.init_schema()
        total = 0
        for i, p in enumerate(csv_paths, 1):
            n_rows = pd.read_csv(p).shape[0]
            self.log.info("[%d/%d] %s  (%s filas)", i, len(csv_paths), p.name, f"{n_rows:,}")
            n = self.load_csv(p, batch_size=batch_size)
            self.log.info("  %s filas insertadas", f"{n:,}")
            total += n
        self.log.info("Total: %s filas en aerosol_data", f"{total:,}")
        return total

    def query_to_df(self, sql: str, params=None) -> pd.DataFrame:
        return pd.read_sql(sql, self.connect(), params=params)

    def summary(self) -> pd.DataFrame:
        return self.query_to_df("""
            SELECT variable,
                   COUNT(*)                      AS puntos,
                   ROUND(AVG(value)::numeric,4)  AS media,
                   ROUND(MIN(value)::numeric,4)  AS minimo,
                   ROUND(MAX(value)::numeric,4)  AS maximo,
                   MIN(timestamp)                AS fecha_ini,
                   MAX(timestamp)                AS fecha_fin
            FROM aerosol_data
            GROUP BY variable ORDER BY puntos DESC
        """)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAFICADOR 2D
# ─────────────────────────────────────────────────────────────────────────────

class AerosolPlotter:

    CMAP = "YlOrRd"

    def __init__(
        self,
        loader: PostgresLoader,
        plots_dir: str = "~/aerosol_plots",
        logger: Optional[logging.Logger] = None,
    ):
        self.loader    = loader
        self.plots_dir = Path(plots_dir).expanduser()
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.log       = _make_logger(logger)

    def scatter_map(
        self,
        variable: str = "Optical_Depth_Land_And_Ocean",
        vmin: float = 0.0,
        vmax: float = 1.5,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        df = self.loader.query_to_df(
            "SELECT lat,lon,value,timestamp FROM aerosol_data WHERE variable=%s AND lat IS NOT NULL",
            (variable,),
        )
        if df.empty:
            self.log.warning("Sin datos para '%s'", variable)
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(
            df["lon"], df["lat"],
            c=df["value"].clip(vmin, vmax),
            cmap=self.CMAP, s=6, alpha=0.75, vmin=vmin, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label=variable)
        ax.set(
            xlabel="Longitud", ylabel="Latitud",
            title=(
                f"Aerosoles MODIS — {variable}\n"
                f"{df['timestamp'].min()} → {df['timestamp'].max()}"
            ),
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = self.plots_dir / (filename or f"scatter_{variable[:40]}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.log.info("Scatter guardado: %s", out)
        return out

    def heatmap(
        self,
        variable: str = "Optical_Depth_Land_And_Ocean",
        resolution: float = 0.5,
        vmin: float = 0.0,
        vmax: float = 1.5,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        df = self.loader.query_to_df(
            "SELECT lat,lon,value FROM aerosol_data WHERE variable=%s AND lat IS NOT NULL",
            (variable,),
        )
        if df.empty:
            self.log.warning("Sin datos para '%s'", variable)
            return None

        df["lb"] = (df["lat"] / resolution).round() * resolution
        df["gb"] = (df["lon"] / resolution).round() * resolution
        grid = df.groupby(["lb", "gb"])["value"].mean().reset_index()

        lats = sorted(grid["lb"].unique())
        lons = sorted(grid["gb"].unique())
        mat  = pd.DataFrame(index=lats, columns=lons, dtype=float)
        for _, r in grid.iterrows():
            mat.loc[r["lb"], r["gb"]] = r["value"]

        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(
            mat.values, aspect="auto", origin="lower",
            extent=[min(lons), max(lons), min(lats), max(lats)],
            cmap=self.CMAP, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label=f"{variable} (media {resolution}°)")
        ax.set(xlabel="Longitud", ylabel="Latitud", title=f"Heatmap — {variable}")
        plt.tight_layout()

        out = self.plots_dir / (filename or f"heatmap_{variable[:40]}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.log.info("Heatmap guardado: %s", out)
        return out

    def time_series(
        self,
        variable: str = "Optical_Depth_Land_And_Ocean",
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        df = self.loader.query_to_df(
            """SELECT timestamp, AVG(value) AS media, STDDEV(value) AS std
               FROM aerosol_data
               WHERE variable=%s AND timestamp IS NOT NULL
               GROUP BY timestamp ORDER BY timestamp""",
            (variable,),
        )
        if df.empty:
            self.log.warning("Sin datos temporales para '%s'", variable)
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["media"], color="darkorange", lw=2, label="Media")
        ax.fill_between(
            df["timestamp"],
            (df["media"] - df["std"]).clip(lower=0),
            df["media"] + df["std"],
            alpha=0.25, color="orange", label="±1 std",
        )
        ax.set(xlabel="Fecha", ylabel=variable, title=f"Serie Temporal — {variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = self.plots_dir / (filename or f"timeseries_{variable[:40]}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.log.info("Serie temporal guardada: %s", out)
        return out

    def plot_all(self, variable: str = "Optical_Depth_Land_And_Ocean") -> list[Path]:
        paths = []
        for fn in (self.scatter_map, self.heatmap, self.time_series):
            try:
                p = fn(variable=variable)
                if p:
                    paths.append(p)
            except Exception as e:
                self.log.error("%s: %s", fn.__name__, e)
        return paths


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    hdf_dir:      str  = "/home/user/Downloads/earthdata",
    output_dir:   str  = "~/aerosol_csv",
    plots_dir:    str  = "~/aerosol_plots",
    db_url:       str  = "postgresql://gnasaconsultores:o8hGJt0PFHeueUMqL5ahvRxCiunly9Yy@d",
    variable:     str  = "Optical_Depth_Land_And_Ocean",
    skip_convert: bool = False,
    skip_load:    bool = False,
    skip_plot:    bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict:

    log     = _make_logger(logger)
    results = {"csvs": [], "rows_loaded": 0, "plots": []}
    conv    = HdfToCsvConverter(output_dir=output_dir, logger=log)
    loader  = PostgresLoader(db_url=db_url, logger=log)
    plotter = AerosolPlotter(loader=loader, plots_dir=plots_dir, logger=log)

    # Paso 1 ──────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("PASO 1 — Conversión HDF4 → CSV  (via pyhdf)")
    log.info("=" * 55)
    if not skip_convert:
        csvs = conv.convert_directory(hdf_dir)
    else:
        csvs = list(Path(output_dir).expanduser().glob("*.csv"))
        log.info("Omitido. %d CSV existentes.", len(csvs))
    results["csvs"] = csvs

    # Paso 2 ──────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("PASO 2 — Carga a PostgreSQL")
    log.info("=" * 55)
    loader.init_schema()
    if not skip_load and csvs:
        n = loader.load_all_csvs(csvs)
        results["rows_loaded"] = n
        try:
            log.info("Resumen en BD:\n%s", loader.summary().to_string(index=False))
        except Exception:
            pass
    elif skip_load:
        log.info("Carga omitida.")
    else:
        log.warning("Sin CSVs — tabla creada pero vacía.")

    # Paso 3 ──────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("PASO 3 — Gráficas 2D de Aerosoles")
    log.info("=" * 55)
    if not skip_plot:
        results["plots"] = plotter.plot_all(variable=variable)
    else:
        log.info("Gráficas omitidas.")

    log.info("Pipeline completado.")
    return results


if __name__ == "__main__":
    import argparse

    # Configura logging a stdout sólo cuando se ejecuta como script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--hdf-dir",      default="/home/user/Downloads/earthdata")
    p.add_argument("--output-dir",   default="~/aerosol_csv")
    p.add_argument("--plots-dir",    default="~/aerosol_plots")
    p.add_argument("--db-url",       default="postgresql://gnasaconsultores:o8hGJt0PFHeueUMqL5ahvRxCiunly9Yy@d")
    p.add_argument("--variable",     default="Optical_Depth_Land_And_Ocean")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--skip-load",    action="store_true")
    p.add_argument("--skip-plot",    action="store_true")
    a = p.parse_args()

    run_pipeline(
        a.hdf_dir, a.output_dir, a.plots_dir,
        a.db_url, a.variable,
        a.skip_convert, a.skip_load, a.skip_plot,
    )
