import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np
import tempfile
import os
import psycopg2
from psycopg2.extras import execute_values

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

st.set_page_config(page_title="HDF Geo-Explorer", layout="wide")
st.title("🛰️ Visor de Archivos HDF con Mapa")


# ──────────────────────────────────────────────
# Base de datos
# ──────────────────────────────────────────────
@st.cache_resource
def get_connection():
    url = st.secrets["database"]["url"]
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def init_db():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS archivos_hdf (
                id SERIAL PRIMARY KEY,
                nombre TEXT NOT NULL,
                dataset TEXT,
                formato TEXT,
                fecha_carga TIMESTAMP DEFAULT NOW(),
                num_puntos INTEGER
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS puntos_geo (
                id SERIAL PRIMARY KEY,
                archivo_id INTEGER REFERENCES archivos_hdf(id) ON DELETE CASCADE,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION,
                value DOUBLE PRECISION
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_puntos_archivo ON puntos_geo(archivo_id);
        """)


def save_to_db(nombre, dataset, formato, geo_df):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO archivos_hdf (nombre, dataset, formato, num_puntos)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (nombre, dataset, formato, len(geo_df)))
        archivo_id = cur.fetchone()[0]
        rows = [
            (archivo_id, float(row.lat), float(row.lon), float(row.value) if hasattr(row, 'value') else None)
            for row in geo_df.itertuples()
        ]
        execute_values(cur,
            "INSERT INTO puntos_geo (archivo_id, lat, lon, value) VALUES %s", rows)
    return archivo_id


def load_archivos():
    conn = get_connection()
    return pd.read_sql(
        "SELECT id, nombre, dataset, formato, fecha_carga, num_puntos "
        "FROM archivos_hdf ORDER BY fecha_carga DESC", conn)


def load_puntos(archivo_id):
    conn = get_connection()
    return pd.read_sql(
        "SELECT lat, lon, value FROM puntos_geo WHERE archivo_id = %s",
        conn, params=(archivo_id,))


def delete_archivo(archivo_id):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM archivos_hdf WHERE id = %s", (archivo_id,))


# ──────────────────────────────────────────────
# Lectores HDF
# ──────────────────────────────────────────────
def collect_datasets_hdf5(hdf_file):
    found = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            found.append(name)
    hdf_file.visititems(_visitor)
    return found


def read_hdf5(tmp_path, key):
    with h5py.File(tmp_path, 'r') as f:
        datasets = collect_datasets_hdf5(f)
        if not datasets:
            return None, None, None
        target = st.sidebar.selectbox("Dataset", datasets, key=key)
        node = f[target]
        if node.ndim < 2:
            st.warning("El dataset debe ser al menos bidimensional.")
            return None, None, None
        raw = node[:10_000]
        df = pd.DataFrame(raw)
    return df, None, target


def read_netcdf(tmp_path, key):
    if not NETCDF4_AVAILABLE:
        return None, None, None
    try:
        ds = nc.Dataset(tmp_path, 'r')
    except Exception as e:
        st.error(f"No se pudo abrir: {e}")
        return None, None, None
    var_names = [v for v in ds.variables if ds.variables[v].ndim >= 2]
    if not var_names:
        ds.close()
        return None, None, None
    target = st.sidebar.selectbox("Variable", var_names, key=key)
    raw = np.array(ds.variables[target][:])
    ds.close()
    if raw.ndim > 2:
        raw = raw[0]
    return pd.DataFrame(raw[:10_000]), None, target


def raster_to_geo_df(band_ds, max_points=50_000):
    gt = band_ds.GetGeoTransform()
    src_srs = band_ds.GetSpatialRef()
    cols = band_ds.RasterXSize
    rows = band_ds.RasterYSize

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    need_reproject = src_srs is not None and not src_srs.IsSame(wgs84)
    if need_reproject:
        transform = osr.CoordinateTransformation(src_srs, wgs84)

    total = rows * cols
    step = int(np.ceil(np.sqrt(total / max_points))) if total > max_points else 1

    band = band_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data = band.ReadAsArray()

    records = []
    for r in range(0, rows, step):
        for c in range(0, cols, step):
            val = float(data[r, c])
            if nodata is not None and val == nodata:
                continue
            x = gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2]
            y = gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5]
            if need_reproject:
                try:
                    lon, lat, _ = transform.TransformPoint(x, y)
                except Exception:
                    continue
            else:
                lon, lat = x, y
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                records.append({'lat': lat, 'lon': lon, 'value': val})
    return pd.DataFrame(records)


def read_hdf4(tmp_path, key):
    if not GDAL_AVAILABLE:
        st.error("GDAL no está disponible.")
        return None, None, None
    ds = gdal.Open(tmp_path)
    if ds is None:
        st.error("GDAL no pudo abrir el archivo.")
        return None, None, None
    subdatasets = ds.GetSubDatasets()
    ds = None
    if not subdatasets:
        return None, None, None

    options = [s[1] for s in subdatasets]
    paths   = [s[0] for s in subdatasets]
    target_label = st.sidebar.selectbox("Dataset", options, key=key)
    target_path  = paths[options.index(target_label)]

    band_ds = gdal.Open(target_path)
    if band_ds is None:
        return None, None, None

    raw = band_ds.GetRasterBand(1).ReadAsArray()
    df_preview = pd.DataFrame(raw[:100])
    with st.spinner("Calculando coordenadas lat/lon..."):
        geo_df = raster_to_geo_df(band_ds)
    band_ds = None
    return df_preview, geo_df, target_label


def process_upload(uploaded_file, key_suffix):
    """Lee el archivo subido y devuelve (df, geo_df, dataset, formato)."""
    ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if ext in ('h5', 'hdf5'):
            df, geo_df, dataset = read_hdf5(tmp_path, key_suffix)
            formato = "HDF5"
        elif ext in ('nc', 'nc4', 'he5'):
            df, geo_df, dataset = read_netcdf(tmp_path, key_suffix)
            formato = "NetCDF"
        else:
            df, geo_df, dataset = read_hdf4(tmp_path, key_suffix)
            formato = "HDF4"
    finally:
        os.unlink(tmp_path)
    return df, geo_df, dataset, formato


# ──────────────────────────────────────────────
# Mapa
# ──────────────────────────────────────────────
def render_map(map_df):
    if map_df is None or map_df.empty:
        st.warning("No hay puntos para mostrar.")
        return

    map_df = map_df.copy()
    if 'value' in map_df.columns and map_df['value'].notna().any():
        vmin = map_df['value'].quantile(0.02)
        vmax = map_df['value'].quantile(0.98)
        norm = ((map_df['value'] - vmin) / (vmax - vmin + 1e-9)).clip(0, 1)
        map_df['r'] = (norm * 200).astype(int)
        map_df['g'] = 30
        map_df['b'] = ((1 - norm) * 200).astype(int)
        color = "[r, g, b, 180]"
    else:
        color = "[200, 30, 0, 160]"

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=map_df['lat'].mean(),
            longitude=map_df['lon'].mean(),
            zoom=4, pitch=0,
        ),
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_color=color,
            get_radius=5_000,
            pickable=True,
        )],
        tooltip={"text": "Lat: {lat}\nLon: {lon}\nValor: {value}"},
    ))
    st.caption(f"{len(map_df):,} puntos graficados.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
init_db()

# Uploader en sidebar — persiste entre tabs
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo HDF4, HDF5 o NetCDF",
    type=['h5', 'hdf5', 'hdf', 'he4', 'he5', 'nc', 'nc4']
)

tab_datos, tab_historial, tab_subir = st.tabs([
    "📊 Datos cargados",
    "🗄️ Historial",
    "💾 Subir a BD",
])

# Procesar archivo una sola vez y guardar en session_state
if uploaded_file is not None:
    file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Leyendo archivo..."):
            df, geo_df, dataset, formato = process_upload(uploaded_file, "main")
        st.session_state["file_id"]  = file_id
        st.session_state["df"]       = df
        st.session_state["geo_df"]   = geo_df
        st.session_state["dataset"]  = dataset
        st.session_state["formato"]  = formato
        st.session_state["nombre"]   = uploaded_file.name
        st.sidebar.success(f"{formato} cargado ✅")
else:
    # Limpiar si se quitó el archivo
    for k in ["file_id","df","geo_df","dataset","formato","nombre"]:
        st.session_state.pop(k, None)

# ── TAB 1: Datos cargados ──
with tab_datos:
    if "df" not in st.session_state:
        st.info("Sube un archivo desde la barra lateral para comenzar.")
    else:
        df     = st.session_state["df"]
        geo_df = st.session_state["geo_df"]

        st.subheader(f"📄 Dataset: `{st.session_state['dataset']}`")
        st.dataframe(df.head(100), use_container_width=True)

        st.divider()
        st.subheader("🗺️ Mapa")

        map_df = geo_df if (geo_df is not None and not geo_df.empty) else None

        # Para HDF5/NetCDF sin geo_df, el usuario elige columnas
        if map_df is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    lat_col = st.selectbox("Columna Latitud", numeric_cols, index=0)
                with c2:
                    lon_options = [c for c in numeric_cols if c != lat_col]
                    lon_col = st.selectbox("Columna Longitud", lon_options)
                map_df = (
                    df[[lat_col, lon_col]].dropna()
                    .rename(columns={lat_col: 'lat', lon_col: 'lon'})
                )
                map_df = map_df[
                    map_df['lat'].between(-90, 90) &
                    map_df['lon'].between(-180, 180)
                ]
            else:
                st.info("No hay columnas numéricas suficientes para el mapa.")

        if map_df is not None:
            render_map(map_df)
            # Guardar map_df en session para usarlo en tab 3
            st.session_state["map_df"] = map_df

# ── TAB 2: Historial ──
with tab_historial:
    st.subheader("Archivos guardados en la base de datos")
    try:
        archivos = load_archivos()
        if archivos.empty:
            st.info("No hay archivos guardados aún.")
        else:
            archivos['fecha_carga'] = pd.to_datetime(archivos['fecha_carga']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(archivos, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                archivo_sel = st.selectbox(
                    "Selecciona un archivo",
                    archivos['id'].tolist(),
                    format_func=lambda x: archivos.loc[archivos['id']==x, 'nombre'].values[0]
                )
            with col2:
                if st.button("🗑️ Eliminar"):
                    delete_archivo(archivo_sel)
                    st.success("Archivo eliminado.")
                    st.rerun()

            if st.button("🗺️ Ver en mapa"):
                with st.spinner("Cargando puntos..."):
                    puntos = load_puntos(archivo_sel)
                st.subheader("🗺️ Mapa desde base de datos")
                render_map(puntos)

    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")

# ── TAB 3: Subir a BD ──
with tab_subir:
    if "map_df" not in st.session_state:
        st.info("Primero carga un archivo en la tab **📊 Datos cargados** para poder guardarlo.")
    else:
        map_df  = st.session_state["map_df"]
        nombre  = st.session_state.get("nombre", "desconocido")
        dataset = st.session_state.get("dataset", "")
        formato = st.session_state.get("formato", "")

        st.subheader("💾 Guardar en base de datos")
        st.write(f"**Archivo:** {nombre}")
        st.write(f"**Dataset:** {dataset}")
        st.write(f"**Formato:** {formato}")
        st.write(f"**Puntos a guardar:** {len(map_df):,}")

        if st.button("💾 Confirmar y guardar"):
            with st.spinner("Guardando en la base de datos..."):
                archivo_id = save_to_db(nombre, str(dataset), formato, map_df)
            st.success(f"✅ Guardado correctamente (ID: {archivo_id}, {len(map_df):,} puntos)")