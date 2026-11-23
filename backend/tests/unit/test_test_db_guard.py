"""Unit tests for the integration-test database identity guard.

These run without any database: they exercise the pure policy function
``classify`` so a production-like target can never be silently accepted.
"""

import importlib

import pytest

from tests.integration import _test_db


@pytest.fixture()
def fresh_guard(monkeypatch):
    """Import a fresh guard module so env changes are observed deterministically."""

    def _load(env):
        for key in ("HAORWATCH_TEST_DB_ALLOWLIST", "HAORWATCH_ALLOW_NON_TEST_DB"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(_test_db)

    return _load


def test_haorwatch_database_is_protected(fresh_guard):
    guard = fresh_guard({})
    allowed, reason = guard.classify("postgresql://u:p@localhost:5432/haorwatch")
    assert not allowed
    assert "haorwatch" in reason


@pytest.mark.parametrize("name", ["prod", "production", "haorwatch_prod", "live", "staging"])
def test_production_like_names_refused(fresh_guard, name):
    guard = fresh_guard({})
    allowed, _ = guard.classify(f"postgresql://u:p@localhost:5432/{name}")
    assert not allowed


def test_unknown_name_refused_without_override(fresh_guard):
    guard = fresh_guard({})
    allowed, reason = guard.classify("postgresql://u:p@localhost:5432/mysterydb")
    assert not allowed
    assert "HAORWATCH_TEST_DB_ALLOWLIST" in reason


def test_safe_name_accepted(fresh_guard):
    guard = fresh_guard({})
    for name in ("haorwatch_test", "ci_db", "tmp_scratch", "sandbox_x"):
        allowed, _ = guard.classify(f"postgresql://u:p@localhost:5432/{name}")
        assert allowed, name


def test_allowlist_permits_otherwise_refused_name(fresh_guard):
    guard = fresh_guard({"HAORWATCH_TEST_DB_ALLOWLIST": "haorwatch, otherdb"})
    allowed, reason = guard.classify("postgresql://u:p@localhost:5432/haorwatch")
    assert allowed
    assert "allow-list" in reason


def test_deliberate_override_permits_unknown_name(fresh_guard):
    guard = fresh_guard({"HAORWATCH_ALLOW_NON_TEST_DB": "1"})
    allowed, _ = guard.classify("postgresql://u:p@localhost:5432/mysterydb")
    assert allowed


def test_reason_never_contains_credentials(fresh_guard):
    guard = fresh_guard({})
    _allowed, reason = guard.classify("postgresql://secret_user:secret_pw@localhost:5432/haorwatch")
    assert "secret_user" not in reason
    assert "secret_pw" not in reason


def test_empty_url_refused(fresh_guard):
    guard = fresh_guard({})
    allowed, reason = guard.classify("")
    assert not allowed
    assert reason


def test_resolve_target_without_url_is_unavailable(fresh_guard, monkeypatch):
    monkeypatch.delenv("HAORWATCH_TEST_DATABASE_URL", raising=False)
    guard = importlib.reload(_test_db)
    target = guard.resolve_target()
    assert not target.available
    assert target.skip_reason is not None


# --------------------------------------------------------------------------- #
# PostGIS / search_path construction
# --------------------------------------------------------------------------- #

def test_quote_ident_quotes_and_rejects_unsafe():
    assert _test_db.quote_ident("haorwatch_test_abc") == '"haorwatch_test_abc"'
    for unsafe in ("bad name", "bad;drop", "", "a-b"):
        with pytest.raises(ValueError):
            _test_db.quote_ident(unsafe)


def test_build_search_path_isolated_first_public_last():
    path = _test_db.build_search_path("haorwatch_test_abc", ("postgis_schema",))
    assert path == '"haorwatch_test_abc", "postgis_schema", "public"'


def test_build_search_path_does_not_assume_public_for_extension():
    # Extension installed in a non-public schema: isolated schema first, then the
    # actual extension schema, then public.
    path = _test_db.build_search_path("isolated_x", ("extensions",))
    assert path.startswith('"isolated_x"')
    assert '"extensions"' in path
    assert path.endswith('"public"')


def test_build_search_path_deduplicates():
    path = _test_db.build_search_path("iso", ("iso", "public"))
    assert path == '"iso", "public"'


def test_build_search_path_with_no_extension_schemas():
    assert _test_db.build_search_path("iso") == '"iso", "public"'


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    def __init__(self, mapping):
        self._mapping = mapping

    def execute(self, _stmt, params):
        return _FakeResult(self._mapping.get(params["name"]))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self, mapping):
        self._mapping = mapping

    def connect(self):
        return _FakeConn(self._mapping)


def test_discover_extension_schemas_finds_non_public_schema():
    engine = _FakeEngine({"postgis": "extensions"})
    assert _test_db.discover_extension_schemas(engine) == ("extensions",)


def test_discover_extension_schemas_falls_back_to_public():
    engine = _FakeEngine({})
    assert _test_db.discover_extension_schemas(engine) == ("public",)


def test_discover_extension_schemas_handles_query_error(monkeypatch):
    class _RaisingConn:
        def execute(self, *_a, **_k):
            raise RuntimeError("catalog unavailable")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _RaisingEngine:
        def connect(self):
            return _RaisingConn()

    assert _test_db.discover_extension_schemas(_RaisingEngine()) == ("public",)
