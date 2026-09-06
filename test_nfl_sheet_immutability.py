import google.auth.transport.requests as google_requests

import modules.nfl_google_sheets as sheets


class FakeResponse:
    def __init__(self, payload=None):
        self._payload = payload or {}
        self.ok = True
        self.text = ""
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def request(self, method, url, timeout=30, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET":
            return FakeResponse({"values": self.values})
        return FakeResponse({})


def bet(probability=61.0, odds=-140, stake=150.0):
    return {
        "game": "BUF @ NYJ",
        "pick": "NYJ ML",
        "probability": probability,
        "odds": odds,
        "edge": 4.5,
        "ev": 4.1,
        "kelly": 3.0,
        "stake": stake,
    }


def test_rescan_does_not_rewrite_original_snapshot(monkeypatch):
    rec_id = sheets._record_id(2026, 1, bet())
    original = [
        "2026-09-01 10:00:00", 2026, 1, "BUF @ NYJ", "NYJ ML", 58.0, -125,
        3.2, 3.1, 2.0, 100.0, "BET", "PENDIENTE", "", "", rec_id,
    ]
    session = FakeSession([sheets.HEADERS, original])
    monkeypatch.setattr(sheets, "_credentials", lambda: (object(), "test-project"))
    monkeypatch.setattr(google_requests, "AuthorizedSession", lambda credentials: session)

    result = sheets.sync_bets([bet(probability=66.0, odds=-160, stake=250.0)], 2026, 1, 5000)

    assert result["ok"] is True
    assert result["inserted"] == 0
    assert result["updated"] == 0
    assert result["skipped_existing"] == 1
    assert [method for method, _, _ in session.calls] == ["GET"]


def test_duplicate_inside_same_scan_is_inserted_once(monkeypatch):
    session = FakeSession([])
    monkeypatch.setattr(sheets, "_credentials", lambda: (object(), "test-project"))
    monkeypatch.setattr(google_requests, "AuthorizedSession", lambda credentials: session)

    row = bet()
    result = sheets.sync_bets([row, dict(row)], 2026, 1, 5000)

    assert result["ok"] is True
    assert result["inserted"] == 1
    assert result["updated"] == 0
    assert result["skipped_existing"] == 1
    methods = [method for method, _, _ in session.calls]
    assert methods.count("PUT") == 1
    assert methods.count("POST") == 1
