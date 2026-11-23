import collections

from app.main import create_app


EXPECTED_DOCUMENTED_PATHS = {
    "/api/v1/health",
    "/api/v1/study-areas",
    "/api/v1/study-areas/{slug}",
    "/api/v1/study-areas/{slug}/analyses",
    "/api/v1/study-areas/{slug}/analyses/latest",
    "/api/v1/analyses/{analysis_id}",
    "/api/v1/analyses/{analysis_id}/layers",
    "/api/v1/analyses/{analysis_id}/layers/{layer_key}",
    "/api/v1/analyses/{analysis_id}/metrics",
    "/api/v1/analyses/{analysis_id}/sources",
    "/api/v1/analyses/{analysis_id}/reports/{language}",
    "/api/v1/admin/analyses",
}


def _route_keys(app):
    keys = []
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            if method.upper() in ("HEAD", "OPTIONS"):
                continue
            keys.append((method.upper(), path))
    return keys


def test_no_duplicate_routes():
    app = create_app()
    keys = _route_keys(app)
    duplicates = {key: count for key, count in collections.Counter(keys).items() if count > 1}
    assert duplicates == {}, f"Duplicate (method, path) registrations detected: {duplicates}"


def test_routes_registered_only_under_versioned_prefix():
    app = create_app()
    paths = {path for _, path in _route_keys(app)}
    unprefixed = sorted(path for path in paths if not path.startswith("/api/v1/"))
    assert unprefixed == [], f"Unexpected unprefixed routes: {unprefixed}"


def test_documented_contract_paths_present():
    app = create_app()
    paths = {path for _, path in _route_keys(app)}
    assert EXPECTED_DOCUMENTED_PATHS <= paths, (
        f"Missing documented routes: {sorted(EXPECTED_DOCUMENTED_PATHS - paths)}"
    )
