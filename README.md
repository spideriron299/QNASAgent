# 🛰️ HDF Geo-Explorer

An interactive web application built with Streamlit for loading, visualizing, and storing geospatial data from HDF4, HDF5, and NetCDF files — directly from the browser, with no GIS software required.

---

## Features

- **Multi-format support** — HDF4 (`.hdf`, `.he4`), HDF5 (`.h5`, `.hdf5`), and NetCDF (`.nc`, `.nc4`)
- **Automatic georeferencing** — extracts lat/lon coordinates directly from the raster geotransform using GDAL
- **Automatic reprojection** — converts any coordinate reference system (e.g. MODIS sinusoidal) to WGS84
- **Interactive map** — renders data points with a blue-to-red colormap scaled to the raster values
- **PostgreSQL persistence** — saves extracted points to a database for historical querying
- **3-tab interface** — view loaded data, browse history, and push to the database independently

---

## App Structure

| Tab | Description |
|-----|-------------|
|  **Loaded Data** | Preview the dataset table and explore the auto-generated map |
|  **History** | Browse previously saved files, reload their maps, or delete records |
|  **Save to DB** | Confirm and push the current file's georeferenced points to PostgreSQL |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | [Streamlit](https://streamlit.io) |
| Geospatial | [GDAL](https://gdal.org), [osr](https://gdal.org/python/) |
| HDF5 reading | [h5py](https://www.h5py.org) |
| NetCDF reading | [netCDF4](https://unidata.github.io/netcdf4-python/) |
| Map rendering | [PyDeck](https://deckgl.readthedocs.io) |
| Data processing | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Database | [PostgreSQL](https://www.postgresql.org) via [psycopg2](https://www.psycopg.org) |

---

##  Deployment (Streamlit Cloud)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Configure system dependencies

Streamlit Cloud reads `packages.txt` for `apt-get` installs. This file is already included and installs:

```
libgdal-dev
gdal-bin
libnetcdf-dev
```

### 3. Configure Python dependencies

All Python packages are listed in `requirements.txt`:

```
h5py
netCDF4
gdal==3.10.3
pandas
pydeck
numpy
psycopg2-binary
```

>  `gdal` must be pinned to match the system `libgdal` version installed by `apt`. On Debian trixie this is `3.10.3`.

### 4. Set up Streamlit Secrets

In your Streamlit Cloud dashboard go to **Settings → Secrets** and add:

```toml
[database]
url = "postgresql://USER:PASSWORD@HOST/DBNAME"
```

> Never commit credentials to the repository.

### 5. Deploy

Push to GitHub and connect the repo in [share.streamlit.io](https://share.streamlit.io). Streamlit Cloud will automatically install system and Python dependencies on each deploy.

---

## Database Schema

The app auto-creates two tables on first run:

```sql
-- Stores metadata for each uploaded file
CREATE TABLE archivos_hdf (
    id          SERIAL PRIMARY KEY,
    nombre      TEXT NOT NULL,
    dataset     TEXT,
    formato     TEXT,
    fecha_carga TIMESTAMP DEFAULT NOW(),
    num_puntos  INTEGER
);

-- Stores the georeferenced points extracted from each file
CREATE TABLE puntos_geo (
    id         SERIAL PRIMARY KEY,
    archivo_id INTEGER REFERENCES archivos_hdf(id) ON DELETE CASCADE,
    lat        DOUBLE PRECISION,
    lon        DOUBLE PRECISION,
    value      DOUBLE PRECISION
);
```

---

## Repository Structure

```
your-repo/
├── Main.py            # Main Streamlit application
├── requirements.txt   # Python dependencies
├── packages.txt       # System (apt) dependencies
└── README.md
```
