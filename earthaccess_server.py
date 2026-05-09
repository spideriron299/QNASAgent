# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
MCP Server for NASA EarthData Access via earthaccess.
Credentials via environment variables: EARTHDATA_USER, EARTHDATA_PASSWORD
"""

import os
import json
from typing import Optional
from mcp.server.fastmcp import FastMCP

import earthaccess

mcp = FastMCP("EarthAccess Server")

# ---------------------------------------------------------------------------
# Auth helper — called once per tool invocation (earthaccess caches session)
# ---------------------------------------------------------------------------

def _login() -> None:
    """Login using env vars. Raises if credentials are missing."""
    #user = os.environ.get("EARTHDATA_USER")
    #pwd  = os.environ.get("EARTHDATA_PASSWORD")
    user = os.environ.get("EARTHDATA_USERNAME")
    pwd  = os.environ.get("EARTHDATA_PASSWORD")
    if not user or not pwd:
        raise EnvironmentError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables must be set."
        )
    earthaccess.login(strategy="environment", persist=False)


# ---------------------------------------------------------------------------
# Tool 1 — Search by concept_id (specific, like the original notebook)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_by_concept_id(
    concept_id: str,
    bbox_west: float,
    bbox_south: float,
    bbox_east: float,
    bbox_north: float,
    date_start: str,
    date_end: str,
    count: int = 10,
) -> str:
    """
    Search NASA EarthData granules using a known concept_id.

    Args:
        concept_id:  CMR concept ID, e.g. 'C1443528505-LAADS' for MYD04_3K v6.1
        bbox_west:   West longitude  (e.g. -10)
        bbox_south:  South latitude  (e.g.  20)
        bbox_east:   East longitude  (e.g.  10)
        bbox_north:  North latitude  (e.g.  50)
        date_start:  ISO date string 'YYYY-MM-DD'
        date_end:    ISO date string 'YYYY-MM-DD'
        count:       Maximum number of granules to return (default 10)

    Returns:
        JSON string with granule count and metadata of each result.
    """
    _login()
    results = earthaccess.search_data(
        concept_id=concept_id,
        bounding_box=(bbox_west, bbox_south, bbox_east, bbox_north),
        temporal=(date_start, date_end),
        count=count,
    )

    output = {
        "concept_id": concept_id,
        "granules_found": len(results),
        "bounding_box": {
            "west": bbox_west, "south": bbox_south,
            "east": bbox_east, "north": bbox_north,
        },
        "temporal": {"start": date_start, "end": date_end},
        "granules": [str(r) for r in results],
    }
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Tool 2 — Search by short name (generic, user does not need to know IDs)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_by_short_name(
    short_name: str,
    bbox_west: float,
    bbox_south: float,
    bbox_east: float,
    bbox_north: float,
    date_start: str,
    date_end: str,
    version: Optional[str] = None,
    count: int = 10,
) -> str:
    """
    Search NASA EarthData granules using a dataset short name.

    Args:
        short_name:  Dataset short name, e.g. 'MYD04_3K', 'MOD11A1', 'GPM_3IMERGHH'
        bbox_west:   West longitude
        bbox_south:  South latitude
        bbox_east:   East longitude
        bbox_north:  North latitude
        date_start:  ISO date string 'YYYY-MM-DD'
        date_end:    ISO date string 'YYYY-MM-DD'
        version:     Optional dataset version string, e.g. '061'
        count:       Maximum granules to return (default 10)

    Returns:
        JSON string with granule count and metadata.
    """
    _login()

    kwargs = dict(
        short_name=short_name,
        bounding_box=(bbox_west, bbox_south, bbox_east, bbox_north),
        temporal=(date_start, date_end),
        count=count,
    )
    if version:
        kwargs["version"] = version

    results = earthaccess.search_data(**kwargs)

    output = {
        "short_name": short_name,
        "version": version,
        "granules_found": len(results),
        "bounding_box": {
            "west": bbox_west, "south": bbox_south,
            "east": bbox_east, "north": bbox_north,
        },
        "temporal": {"start": date_start, "end": date_end},
        "granules": [str(r) for r in results],
    }
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Tool 3 — Download granules returned by a previous search
# ---------------------------------------------------------------------------

@mcp.tool()
def download_granules(
    concept_id: str,
    bbox_west: float,
    bbox_south: float,
    bbox_east: float,
    bbox_north: float,
    date_start: str,
    date_end: str,
    output_dir: str,
    count: int = 5,
) -> str:
    """
    Search and download granules to a local directory.

    Args:
        concept_id:  CMR concept ID of the dataset
        bbox_west:   West longitude
        bbox_south:  South latitude
        bbox_east:   East longitude
        bbox_north:  North latitude
        date_start:  ISO date string 'YYYY-MM-DD'
        date_end:    ISO date string 'YYYY-MM-DD'
        output_dir:  Absolute path where files will be saved
        count:       Maximum granules to download (default 5)

    Returns:
        JSON string listing downloaded file paths.
    """
    _login()

    results = earthaccess.search_data(
        concept_id=concept_id,
        bounding_box=(bbox_west, bbox_south, bbox_east, bbox_north),
        temporal=(date_start, date_end),
        count=count,
    )

    if not results:
        return json.dumps({"status": "no_granules_found", "files": []})

    expanded = os.path.expanduser(output_dir)
    os.makedirs(expanded, exist_ok=True)

    files = earthaccess.download(results, expanded)

    return json.dumps({
        "status": "success",
        "output_dir": expanded,
        "granules_found": len(results),
        "files_downloaded": len(files),
        "files": [str(f) for f in files],
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 4 — Dataset discovery (search collections, not granules)
# ---------------------------------------------------------------------------

@mcp.tool()
def discover_datasets(
    keyword: str,
    count: int = 5,
) -> str:
    """
    Search NASA CMR for dataset collections matching a keyword.
    Useful to find the correct short_name or concept_id before downloading.

    Args:
        keyword:  Free-text keyword, e.g. 'aerosol', 'land surface temperature'
        count:    Maximum collections to return (default 5)

    Returns:
        JSON string with collection metadata including concept_id and short_name.
    """
    _login()

    results = earthaccess.search_datasets(keyword=keyword, count=count)

    collections = []
    for r in results:
        meta = r.get("umm", {})
        collections.append({
            "concept_id": r.get("meta", {}).get("concept-id", ""),
            "short_name": meta.get("ShortName", ""),
            "version":    meta.get("Version", ""),
            "title":      meta.get("EntryTitle", ""),
            "provider":   r.get("meta", {}).get("provider-id", ""),
        })

    return json.dumps({
        "keyword": keyword,
        "collections_found": len(collections),
        "collections": collections,
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")