from modules import nfl_therundown_odds as odds


class FakeResponse:
    status_code = 200
    headers = {"X-Datapoints": "21"}

    def json(self):
        return {"events": [{
            "event_id": "e1",
            "teams_normalized": [
                {"abbreviation": "NE", "is_home": False},
                {"abbreviation": "SEA", "is_home": True},
            ],
            "markets": [
                {"market_id": 1, "period_id": 0, "participants": [
                    {"name": "New England Patriots", "lines": [{"prices": {
                        "19": {"price": 140, "is_main_line": True, "updated_at": "2026-09-06T10:00:00Z"},
                        "22": {"price": 138, "is_main_line": True}, "3": {"price": 135, "is_main_line": True}}}]},
                    {"name": "Seattle Seahawks", "lines": [{"prices": {
                        "19": {"price": -166, "is_main_line": True},
                        "22": {"price": -164, "is_main_line": True}, "3": {"price": -160, "is_main_line": True}}}]},
                ]},
                {"market_id": 2, "period_id": 0, "participants": [
                    {"name": "New England Patriots", "lines": [{"value": "+3.5", "prices": {
                        "19": {"price": -110, "is_main_line": True}, "3": {"price": -108, "is_main_line": True}}}]},
                    {"name": "Seattle Seahawks", "lines": [{"value": "-3.5", "prices": {
                        "19": {"price": -110, "is_main_line": True}, "3": {"price": -112, "is_main_line": True}}}]},
                ]},
                {"market_id": 3, "period_id": 0, "participants": [
                    {"name": "Over", "lines": [{"value": "44.5", "prices": {
                        "19": {"price": -108, "is_main_line": True}, "3": {"price": -105, "is_main_line": True}}}]},
                    {"name": "Under", "lines": [{"value": "44.5", "prices": {
                        "19": {"price": -112, "is_main_line": True}, "3": {"price": -115, "is_main_line": True}}}]},
                ]},
            ],
        }]}


class AliasResponse:
    status_code = 200
    headers = {}
    def __init__(self, away, home): self.away, self.home = away, home
    def json(self):
        names = {"SFO": "San Francisco 49ers", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "LVR": "Las Vegas Raiders"}
        return {"events": [{"event_id": "alias", "teams_normalized": [
            {"abbreviation": self.away, "is_home": False}, {"abbreviation": self.home, "is_home": True}],
            "markets": [{"market_id": 1, "period_id": 0, "participants": [
                {"name": names[self.away], "lines": [{"prices": {"19": {"price": 120, "is_main_line": True}}}]},
                {"name": names[self.home], "lines": [{"prices": {"19": {"price": -140, "is_main_line": True}}}]},
            ]}]}]}


class IncompletePrimaryResponse(FakeResponse):
    def json(self):
        data = super().json()
        del data["events"][0]["markets"][0]["participants"][1]["lines"][0]["prices"]["19"]
        del data["events"][0]["markets"][0]["participants"][1]["lines"][0]["prices"]["22"]
        return data


class EmptyResponse:
    status_code = 200
    headers = {}
    def json(self): return {"events": []}


def fake_get(url, **kwargs):
    assert "/sports/2/events/2026-09-09" in url
    assert kwargs["params"]["market_ids"] == "1,2,3"
    assert kwargs["params"]["main_line"] == "true"
    assert kwargs["headers"]["X-TheRundown-Key"]
    return FakeResponse()


def test_get_markets_uses_priority_book_and_extracts_all_three(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key"); monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,23,3")
    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=fake_get)
    assert q["book"] == "DraftKings" and q["home_moneyline"] == -166 and q["away_moneyline"] == 140
    assert q["home_spread"] == -3.5 and q["away_spread"] == 3.5
    assert q["home_spread_odds"] == -110 and q["away_spread_odds"] == -110
    assert q["total_line"] == 44.5 and q["over_odds"] == -108 and q["under_odds"] == -112
    assert q["spread_book"] == "DraftKings" and q["total_book"] == "DraftKings"
    assert q["source"] == "TheRundown" and q["fetched_at"].endswith("Z") and q["slate_date"] == "2026-09-09"


def test_missing_key_returns_no_quote(monkeypatch):
    odds._CACHE.clear(); monkeypatch.delenv("THERUNDOWN_KEY", raising=False)
    assert odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=fake_get) is None


def test_incomplete_primary_moneyline_falls_back_to_other_legit_book(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key"); monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,3")
    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=lambda url, **kwargs: IncompletePrimaryResponse())
    assert q["book"] == "Pinnacle" and q["home_moneyline"] == -160 and q["away_moneyline"] == 135
    assert q["spread_book"] == "DraftKings" and q["total_book"] == "DraftKings"


def test_default_priority_keeps_main_books_first(monkeypatch):
    monkeypatch.delenv("THERUNDOWN_AFFILIATE_IDS", raising=False)
    priority = odds._priority(); assert priority[:3] == ["19", "22", "23"] and "3" in priority


def test_diagnostic_reports_three_core_markets_without_secret(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY", "super-secret-key"); monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19,22,3")
    result = odds.diagnose_date("2026-09-09", get_fn=fake_get)
    assert result["ok"] is True and result["reason"] == "OK" and result["events"] == 1
    assert result["moneyline_markets"] == 1 and result["spread_markets"] == 1 and result["total_markets"] == 1
    assert result["games_with_moneyline"] == 1 and result["games_with_spread"] == 1 and result["games_with_total"] == 1
    assert result["datapoints"] == "21" and "super-secret-key" not in repr(result)


def test_diagnostic_explains_no_events(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    result = odds.diagnose_date("2026-09-09", get_fn=lambda url, **kwargs: EmptyResponse())
    assert result["ok"] is True and result["reason"] == "NO_EVENTS_FOR_DATE" and result["games_with_any_quote"] == 0


def test_sunday_game_can_match_thursday_weekly_slate(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key"); requested_dates = []
    def get_weekly_slate(url, **kwargs):
        requested_dates.append(url.rsplit("/", 1)[-1]); return FakeResponse() if url.endswith("2026-09-10") else EmptyResponse()
    q = odds.get_moneyline("SEA", "NE", "2026-09-13", get_fn=get_weekly_slate)
    assert q is not None and q["book"] == "DraftKings" and q["slate_date"] == "2026-09-10"
    assert requested_dates[:4] == ["2026-09-13", "2026-09-12", "2026-09-11", "2026-09-10"]


def test_extended_window_can_find_earlier_week_snapshot(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    q = odds.get_moneyline("SEA", "NE", "2026-09-16", get_fn=lambda url, **kwargs: FakeResponse() if url.endswith("2026-09-10") else EmptyResponse())
    assert q is not None and q["slate_date"] == "2026-09-10"


def test_forward_snapshot_is_last_resort(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key")
    q = odds.get_moneyline("SEA", "NE", "2026-09-09", get_fn=lambda url, **kwargs: FakeResponse() if url.endswith("2026-09-10") else EmptyResponse())
    assert q is not None and q["slate_date"] == "2026-09-10"


def test_team_aliases_cover_remaining_provider_variants():
    assert odds._norm_team("WSH") == "WAS" and odds._norm_team("JAC") == "JAX"
    assert odds._norm_team("OAK") == "LV" and odds._norm_team("STL") == "LA"
    assert odds._norm_team("LVR") == "LV" and odds._norm_team("SFO") == "SF"
    assert odds._norm_team("GNB") == "GB" and odds._norm_team("NWE") == "NE"


def test_alias_matchups_sfo_lar_and_mia_lvr(monkeypatch):
    odds._CACHE.clear(); monkeypatch.setenv("THERUNDOWN_KEY", "test-key"); monkeypatch.setenv("THERUNDOWN_AFFILIATE_IDS", "19")
    q1 = odds.get_moneyline("LAR", "SF", "2026-09-13", get_fn=lambda url, **kwargs: AliasResponse("SFO", "LAR") if url.endswith("2026-09-13") else EmptyResponse())
    odds._CACHE.clear()
    q2 = odds.get_moneyline("LV", "MIA", "2026-09-13", get_fn=lambda url, **kwargs: AliasResponse("MIA", "LVR") if url.endswith("2026-09-13") else EmptyResponse())
    assert q1 is not None and q1["home_moneyline"] == -140
    assert q2 is not None and q2["home_moneyline"] == -140
