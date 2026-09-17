import pandas as pd
from modules.nfl_pbp_engine import features_pbp_actuales

def _data():
    rows=[]
    for team,base in [('A',0),('B',100)]:
        for week in range(1,7):
            row={'game_id':f'{team}{week}','season':2026,'week':week,'team':team}
            for m in ['off_epa_play','off_success_rate','pass_epa','rush_epa','explosive_rate','sack_rate_allowed','plays','def_epa_allowed','def_success_allowed','def_explosive_allowed','pressure_rate']:
                row[m]=base+week
            rows.append(row)
    return pd.DataFrame(rows)

def test_asof_excludes_target_and_future():
    df=_data()
    x=features_pbp_actuales(df,'A','B',windows=(4,),as_of_season=2026,as_of_week=5)
    assert x is not None
    assert x['home_off_epa_play_4']==2.5
    assert x['away_off_epa_play_4']==102.5

def test_future_mutation_cannot_change_historical_replay():
    df=_data(); x=features_pbp_actuales(df,'A','B',windows=(4,),as_of_season=2026,as_of_week=5)
    df.loc[df.week>=5,'off_epa_play']=999999
    y=features_pbp_actuales(df,'A','B',windows=(4,),as_of_season=2026,as_of_week=5)
    assert x['home_off_epa_play_4']==y['home_off_epa_play_4']
    assert x['away_off_epa_play_4']==y['away_off_epa_play_4']

def test_live_mode_keeps_latest_behavior():
    df=_data(); x=features_pbp_actuales(df,'A','B',windows=(4,))
    assert x['home_off_epa_play_4']==4.5
    assert x['away_off_epa_play_4']==104.5
