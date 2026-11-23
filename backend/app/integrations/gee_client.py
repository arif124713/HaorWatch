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
        """Build a Sentinel-1 VV/VH backscatter composite for the observation window.

        Returns ``(image, meta)``. ``meta`` explicitly documents that the returned
        image is a **temporal mean composite** of multiple acquisitions, not a
        single scene, together with the count, product IDs, acquisition dates,
        orbit direction, polarisation, and compositing method. Downstream code
        must surface this so results are never described as a single-date
        observation.
        """
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
        acquisition_millis = collection.limit(50).aggregate_array("system:time_start").getInfo()
        import datetime as _dt

        acquisition_dates = sorted(
            {
                _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).strftime("%Y-%m-%d")
                for ms in (acquisition_millis or [])
                if ms is not None
            }
        )
        mosaic = collection.mean()
        meta = {
            "collection": S1_COLLECTION,
            "observation_type": "temporal_composite",
            "compositing_method": "mean",
            "value_domain": "dB",
            "image_count": image_count,
            "product_ids": product_ids,
            "acquisition_dates": acquisition_dates,
            "orbit_direction": orbit,
            "polarisation": ["VV", "VH"],
            "period_start": start,
            "period_end": end,
            "bands": ["VV", "VH"],
            "start": start,
            "end": end,
            "note": (
                "Multi-date temporal mean composite. NOT a single-date observation and NOT a "
                "peak-flood-extent product; the mean can smooth or damp transient flood peaks."
            ),
        }
        return mosaic, meta

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
        # Tile in the target projected CRS so tile ground size is constant across
        # latitude. Region coordinates are expressed in the same CRS passed to GEE.
        tiles = self._tiles_projected(aoi_geojson, crs, scale_m)
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
                # Guard against stitching tiles with inconsistent grids.
                crs0 = datasets[0].crs
                nodata = datasets[0].nodata
                for dataset in datasets[1:]:
                    if dataset.crs != crs0:
                        raise ProviderUnavailableError(
                            "GEE_MOSAIC_CRS_MISMATCH",
                            "GEE export tiles have inconsistent CRS; refusing to stitch mismatched rasters.",
                            503,
                        )
                merged, merged_transform = rio_merge(datasets)
                source_crs = crs0
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

    def _aoi_wgs84_bounds(self, aoi_geojson: dict) -> tuple[float, float, float, float]:
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
        return min(xs), min(ys), max(xs), max(ys)

    def _tiles_projected(self, aoi_geojson: dict, crs: str, scale_m: float) -> list[list[float]]:
        """Tile the AOI in a projected CRS so every tile covers a constant ground area.

        Degree-based tiling (``MAX_TILE_PIXELS * scale_m / METERS_PER_DEGREE``)
        produces tiles whose ground width shrinks with latitude, so a fixed
        pixel budget is not preserved. Tiling in the projected CRS keeps tiles at
        ``MAX_TILE_PIXELS * scale_m`` metres on each side at any latitude.
        """
        west, south, east, north = self._aoi_wgs84_bounds(aoi_geojson)
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        corners_x, corners_y = [], []
        for lon, lat in ((west, south), (west, north), (east, south), (east, north)):
            x, y = transformer.transform(lon, lat)
            corners_x.append(x)
            corners_y.append(y)
        minx, maxx = min(corners_x), max(corners_x)
        miny, maxy = min(corners_y), max(corners_y)

        tile_m = MAX_TILE_PIXELS * scale_m
        tiles: list[list[float]] = []
        y = miny
        # Guard against pathological inputs producing huge tile counts.
        max_tiles = 256
        while y < maxy and len(tiles) < max_tiles:
            x = minx
            y_end = min(y + tile_m, maxy)
            while x < maxx and len(tiles) < max_tiles:
                x_end = min(x + tile_m, maxx)
                tiles.append([x, y, x_end, y_end])
                x = x_end
            y = y_end
        return tiles
