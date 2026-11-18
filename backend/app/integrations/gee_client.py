import json
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from app.core.exceptions import ProviderUnavailableError

S1_COLLECTION = "COPERNICUS/S1_GRD"
JRC_COLLECTION = "JRC/GSW1_4/GlobalSurfaceWater"
WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"

MAX_TILE_PIXELS = 1200
METERS_PER_DEGREE = 111320.0


class GEEClient:
    def __init__(self, project: str):
        if not project:
            raise ProviderUnavailableError(
                "GEE_PROJECT_MISSING", "Google Earth Engine project is not configured.", 503
            )
        try:
            import ee
        except ImportError as error:
            raise ProviderUnavailableError(
                "GEE_UNAVAILABLE", "Earth Engine Python package is not installed.", 503
            ) from error
        self.ee = ee
        self.ee.Initialize(project=project)

    def s1_mosaic(self, aoi_geojson: dict, start: str, end: str, orbit: str | None = None):
        ee = self.ee
        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterBounds(ee.Geometry(aoi_geojson))
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        if orbit:
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit))
        image_count = int(collection.size().getInfo())
        if image_count == 0:
            raise ProviderUnavailableError(
                "S1_UNAVAILABLE", "No Sentinel-1 imagery found for the requested period.", 503
            )
        product_ids = collection.limit(50).aggregate_array("system:id").getInfo()
        mosaic = collection.mean()
        return mosaic, {
            "collection": S1_COLLECTION,
            "image_count": image_count,
            "composite": "mean_dB",
            "bands": ["VV", "VH"],
            "product_ids": product_ids,
            "start": start,
            "end": end,
        }

    def jrc_occurrence(self, aoi_geojson: dict):
        image = self.ee.Image(JRC_COLLECTION).select("occurrence")
        return image, {"collection": JRC_COLLECTION, "version": "1.4", "band": "occurrence"}

    def worldcover(self, aoi_geojson: dict):
        image = self.ee.ImageCollection(WORLDCOVER_COLLECTION).first().select("Map")
        return image, {
            "collection": WORLDCOVER_COLLECTION,
            "version": "v200",
            "band": "Map",
            "agricultural_classes": [40],
        }

    def export_image(
        self,
        image,
        aoi_geojson: dict,
        crs: str,
        scale_m: float,
        out_path: str | Path,
        band_names: list[str] | None = None,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tiles = self._tiles(aoi_geojson, scale_m)
        datasets = []
        tile_files = []
        try:
            for index, tile in enumerate(tiles):
                params = {
                    "scale": scale_m,
                    "crs": crs,
                    "region": tile,
                    "format": "GeoTIFF",
                    "filePerBand": False,
                    "name": f"tile_{index}",
                }
                if band_names:
                    params["bands"] = band_names
                url = image.getDownloadURL(params)
                tile_path = out_path.parent / f"{out_path.stem}_tile{index}.tif"
                self._download(url, tile_path)
                datasets.append(rasterio.open(tile_path))
                tile_files.append(tile_path)
            if not datasets:
                raise ProviderUnavailableError("GEE_EXPORT_FAILED", "Earth Engine export returned no data.", 503)
            if len(datasets) == 1:
                with datasets[0] as src:
                    merged = src.read()
                    merged_transform = src.transform
                    source_crs = src.crs
                    nodata = src.nodata
            else:
                merged, merged_transform = rio_merge(datasets)
                source_crs = datasets[0].crs
                nodata = datasets[0].nodata
            if merged.shape[0] == 1:
                merged = merged[0]
            from app.geospatial.raster_io import write_raster

            result = write_raster(
                out_path,
                merged.astype(np.float32),
                merged_transform,
                source_crs,
                nodata=nodata,
            )
        finally:
            for dataset in datasets:
                dataset.close()
            for path in tile_files:
                Path(path).unlink(missing_ok=True)
        return result

    def _download(self, url: str, out_path: Path) -> None:
        try:
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
                response.raise_for_status()
                with open(out_path, "wb") as handle:
                    for chunk in response.iter_bytes(1024 * 512):
                        handle.write(chunk)
        except httpx.HTTPError as error:
            raise ProviderUnavailableError(
                "GEE_EXPORT_FAILED", "Earth Engine export failed; reduce the area or resolution.", 503
            ) from error

    def _tiles(self, aoi_geojson: dict, scale_m: float) -> list[list[float]]:
        geometry = aoi_geojson.get("geometry", aoi_geojson)
        coordinates = geometry.get("coordinates")
        xs, ys = [], []

        def walk(coords):
            if isinstance(coords[0], (int, float)):
                xs.append(coords[0])
                ys.append(coords[1])
            else:
                for item in coords:
                    walk(item)

        walk(coordinates)
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)
        tile_deg = MAX_TILE_PIXELS * scale_m / METERS_PER_DEGREE
        tiles = []
        lat = south
        while lat < north:
            lon = west
            lat_end = min(lat + tile_deg, north)
            while lon < east:
                lon_end = min(lon + tile_deg, east)
                tiles.append([lon, lat, lon_end, lat_end])
                lon = lon_end
            lat = lat_end
        return tiles
