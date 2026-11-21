from types import SimpleNamespace

from app.integrations.gee_client import GEEClient


class _FakeCollection:
    def __init__(self, count, product_ids, times):
        self._count = count
        self._product_ids = product_ids
        self._times = times

    def filterBounds(self, *_a, **_k):
        return self

    def filterDate(self, *_a, **_k):
        return self

    def filter(self, *_a, **_k):
        return self

    def select(self, *_a, **_k):
        return self

    def size(self):
        return SimpleNamespace(getInfo=lambda: self._count)

    def limit(self, _n):
        return self

    def aggregate_array(self, key):
        values = self._product_ids if key == "system:id" else self._times
        return SimpleNamespace(getInfo=lambda: values)

    def mean(self):
        return "MEAN_COMPOSITE"


class _FakeEE:
    def __init__(self, collection):
        self._collection = collection

    def ImageCollection(self, _name):  # noqa: N802 - mirror ee API
        return self._collection

    def Geometry(self, _geojson):  # noqa: N802
        return object()

    class Filter:
        @staticmethod
        def listContains(*_a):
            return object()

        @staticmethod
        def eq(*_a):
            return object()


def _client_with(collection):
    client = GEEClient.__new__(GEEClient)
    client.ee = _FakeEE(collection)
    return client


def test_s1_mosaic_documents_temporal_composite():
    ms = 1_700_000_000_000
    collection = _FakeCollection(count=3, product_ids=["S1A_x", "S1A_y", "S1A_z"], times=[ms, ms, ms])
    client = _client_with(collection)
    image, meta = client.s1_mosaic({"type": "Polygon", "coordinates": []}, "2026-05-01", "2026-05-15")

    assert image == "MEAN_COMPOSITE"
    assert meta["observation_type"] == "temporal_composite"
    assert meta["compositing_method"] == "mean"
    assert meta["value_domain"] == "dB"
    assert meta["image_count"] == 3
    assert meta["product_ids"] == ["S1A_x", "S1A_y", "S1A_z"]
    assert meta["acquisition_dates"] == ["2023-11-14"]
    assert meta["polarisation"] == ["VV", "VH"]
    assert "NOT a single-date observation" in meta["note"]
    assert "peak-flood-extent" in meta["note"]
