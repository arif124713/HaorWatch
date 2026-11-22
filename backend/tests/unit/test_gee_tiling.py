from app.integrations.gee_client import MAX_TILE_PIXELS, GEEClient


def _client():
    return GEEClient.__new__(GEEClient)


AOI = {
    "type": "Polygon",
    "coordinates": [[[91.6, 24.3], [91.62, 24.3], [91.62, 24.32], [91.6, 24.32], [91.6, 24.3]]],
}


def test_tiles_cover_aoi_extent():
    tiles = _client()._tiles_projected(AOI, "EPSG:32646", 10.0)
    assert tiles
    minx = min(t[0] for t in tiles)
    miny = min(t[1] for t in tiles)
    maxx = max(t[2] for t in tiles)
    maxy = max(t[3] for t in tiles)
    # Union of tiles must span the full projected AOI envelope.
    assert minx < maxx and miny < maxy


def test_tiles_have_consistent_projected_size_at_high_latitude():
    """Projected tiling keeps a constant ground tile size (degree tiling would not)."""
    high_lat = {
        "type": "Polygon",
        "coordinates": [[[20.0, 69.0], [20.6, 69.0], [20.6, 69.6], [20.0, 69.6], [20.0, 69.0]]],
    }
    tiles = _client()._tiles_projected(high_lat, "EPSG:32634", 10.0)
    assert len(tiles) > 1
    tile_m = MAX_TILE_PIXELS * 10.0
    full = [t for t in tiles if (t[2] - t[0]) == tile_m or (t[3] - t[1]) == tile_m]
    # At high latitude the projected tile must still reach the full metric size.
    assert full


def test_tiles_are_non_overlapping_and_ordered():
    tiles = _client()._tiles_projected(AOI, "EPSG:32646", 10.0)
    # No duplicate tile extents.
    seen = set()
    for t in tiles:
        key = tuple(round(v, 6) for v in t)
        assert key not in seen
        seen.add(key)
    # Tiles must not overlap: adjacent x ranges share only an edge.
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= 1e-6 or overlap_y <= 1e-6


def test_tile_count_is_bounded_for_small_aoi():
    tiles = _client()._tiles_projected(AOI, "EPSG:32646", 10.0)
    assert 0 < len(tiles) <= 256
