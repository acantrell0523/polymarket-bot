"""Global test isolation.

Every test gets a throwaway SQLite DB — no test may read or write the real
data/trades.db. This matters doubly since position-state persistence landed:
an unmocked open_position() would leave phantom state rows that a real
paper bot (restore_state=True) would then resurrect as positions.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_trade_db(tmp_path, monkeypatch):
    import bot.trade_db as tdb
    import bot.edge_log as elog

    test_db = str(tmp_path / "trades_test.db")
    monkeypatch.setattr(tdb, "DB_PATH", test_db)
    if hasattr(elog, "DB_PATH"):
        monkeypatch.setattr(elog, "DB_PATH", test_db)
    tdb.init_db()
    yield
