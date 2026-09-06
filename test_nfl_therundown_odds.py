from modules import nfl_therundown_odds as odds


class FakeResponse:
    status_code = 200
    headers = {"X-Datapoints": "7"}

    def json(self):
        return {
            "events": [{
                "event_id": "e1",
                "teams_normalized": [
                    {"abbreviation": "NE", "is_home": False},
                    {"abbreviation": "SEA", "is_home": True},
                ],
                "markets": [{
                    "market_id": 1,
                    "period_id": 0,
                    "participants": [
                        {"name": "New England Patriots", "lines": [{"prices": {
                            "19": {"price": -105, "is_main_line": True, "updated_at": "2026-09-06T10:00:00Z"},
                            "22": {"price": -110, "is_main_line": True},
                            "3": {"price": -112, "is_main_line": True},
                        }}]},
                        {"name": "Seattle Seahawks", "lines": [{"prices": {
                            "19": {"price": -115, "is_main_line": True},
                            "22": {"price": -108, "is_main_line": True},
                            "3": {"price": -109, "is_main_line": True},
                        }}]},
                    ],
                }],
            }]
        }


class IncompletePrimaryResponse(FakeResponse):
    def json(self):
        data = super().json()
        del data["events"][0]["markets"][0]["participants"][1]["lines"][0]["prices"]["19"]
        del data["events"][0]["markets"][0]["participants"][1]["lines"][0]["prices"]["22"]
        return data


class EmptyResponse:
    status_code = 200
    headers = {}

    def json(self):
        return {"events": []}


def fake_get(url, **kwargs):
    assert "/sports/2/events/2026-09-09" in url
    assert kwargs["params"]["market_ids"] == "1"
    assert kwargs["params"]["main_line"] == "true"
    assert kwargs["headers"]["X-TheRundown-Key"]
    return FakeResponse()


def test_get_moneyline_uses_priority_book(monkeypatch):
    odds._CACHE.clear()
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,23,3")
    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=fake_get)
    assert q["book"] == "DraftKings"
    assert q["home_moneyline"] == -115
    assert q["away_moneyline"] == -105
    assert q["source"] == "TheRundown"
    assert q["fetched_at"].endswith("Z")


def test_missing_key_returns_no_quote(monkeypatch):
    odds._CACHE.clear()
    monkeypatch.delenv("THERUNDOWN_KEY", raising=False)
    assert odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=fake_get) is None


def test_incomplete_primary_falls_back_to_other_legit_book(monkeypatch):
    odds._CACHE.clear()
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,3")

    def get_incomplete(url, **kwargs):
        return IncompletePrimaryResponse()

    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=get_incomplete)
    assert q["book"] == "Pinnacle"
    assert q["home_moneyline"] == -109
    assert q["away_moneyline"] == -112


def test_default_priority_keeps_main_books_first(monkeypatch):
    monkeypatch.delenv("THERUNDOWN_AFFILIATE_IDS", raising=False)
    priority = odds._priority()
    assert priority[:3] == ["19", "22", "23"]
    assert "3" in priority


def test_diagnostic_is_safe_and_reports_books(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY", "super-secret-key")
    monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,3")
    result = odds.diagnose_date("2026-09-09", get_fn=fake_get)
    assert result["ok"] is True
    assert result["reason"] == "OK"
    assert result["events"] == 1
    assert result["complete_moneyline_quotes"] == 1
    assert "DraftKings" in result["offered_books"]
    assert "Pinnacle" in result["offered_books"]
    assert result["datapoints"] == "7"
    assert "super-secret-key" not in repr(result)


def test_diagnostic_explains_no_events(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")

    def get_empty(url, **kwargs):
        return EmptyResponse()

    result = odds.diagnose_date("2026-09-09", get_fn=get_empty)
    assert result["ok"] is True
    assert result["reason"] == "NO_EVENTS_FOR_DATE"
    assert result["complete_moneyline_quotes"] == 0
