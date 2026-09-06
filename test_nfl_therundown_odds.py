import os

from modules import nfl_therundown_odds as odds


class FakeResponse:
    status_code = 200
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
                            "19": [{"price": -105, "is_main_line": True, "updated_at": "2026-09-06T10:00:00Z"}],
                            "22": [{"price": -110, "is_main_line": True}],
                        }}]},
                        {"name": "Seattle Seahawks", "lines": [{"prices": {
                            "19": [{"price": -115, "is_main_line": True}],
                            "22": [{"price": -108, "is_main_line": True}],
                        }}]},
                    ],
                }],
            }]
        }


def fake_get(url, **kwargs):
    assert "/sports/2/events/2026-09-09" in url
    assert kwargs["params"]["market_ids"] == "1"
    assert kwargs["headers"]["X-TheRundown-Key"] == "test-key"
    return FakeResponse()


def test_get_moneyline_uses_priority_book(monkeypatch):
    odds._CACHE.clear()
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
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


def test_incomplete_primary_falls_back(monkeypatch):
    odds._CACHE.clear()
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "22,19")
    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=fake_get)
    assert q["book"] == "BetMGM"
    assert q["home_moneyline"] == -108
    assert q["away_moneyline"] == -110
