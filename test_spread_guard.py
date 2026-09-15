from modules.nfl_google_sheets import _spread_key


def test_same_team_spread_variants_share_key():
    assert _spread_key('DEN @ KC', 'DEN +3') == _spread_key('DEN @ KC', 'DEN +2.5')
    assert _spread_key('DEN @ KC', 'DEN +3') == _spread_key('DEN @ KC', 'DEN +3.5')


def test_other_team_or_game_is_independent():
    assert _spread_key('DEN @ KC', 'DEN +3') != _spread_key('DEN @ KC', 'KC -3')
    assert _spread_key('DEN @ KC', 'DEN +3') != _spread_key('DEN @ LV', 'DEN +3')


def test_ml_and_totals_are_not_restricted():
    assert _spread_key('DEN @ KC', 'DEN ML') is None
    assert _spread_key('DEN @ KC', 'Over 43.5') is None
    assert _spread_key('DEN @ KC', 'Under 43.5') is None
