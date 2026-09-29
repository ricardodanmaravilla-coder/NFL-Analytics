"""Experimental pregame feature expansion. Does not modify production.
All rolling features are built before the target week is added to history.
QB starter is a *proxy* (prior-week primary passer), never claimed confirmed.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_pbp_engine import PBP_METRICS, SITUATIONAL_METRICS, construir_pbp_pregame

QB_STATS = ("attempts", "completions", "passing_yards", "passing_tds",
            "interceptions", "sacks", "passing_air_yards",
            "passing_yards_after_catch", "passing_first_downs", "passing_epa")
QB_RATES = ("completion_pct", "yards_per_attempt", "td_per_attempt",
            "int_per_attempt", "sacks_per_attempt", "epa_per_attempt")
VENUE_STATS = ("pf", "pa", "margin", "total")
EXTRA_PBP = tuple(SITUATIONAL_METRICS)

def _numeric(frame, col):
    return pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(np.nan, index=frame.index)

def _qb_week_rows(qbs):
    if qbs is None or qbs.empty:
        return pd.DataFrame()
    q=qbs.copy()
    needed={"season","week","recent_team","player_id","attempts"}
    if not needed.issubset(q.columns):
        return pd.DataFrame()
    for col in QB_STATS:
        q[col]=_numeric(q,col)
    q["season"]=pd.to_numeric(q.season,errors="coerce")
    q["week"]=pd.to_numeric(q.week,errors="coerce")
    q=q.dropna(subset=["season","week","recent_team","player_id"])
    # Multiple QBs can play; primary passer is chosen from actual attempts in
    # *past* games only. Never use the target game's QB to identify its starter.
    q=q.sort_values(["season","week","recent_team","attempts"],ascending=[True,True,True,False])
    return q.drop_duplicates(["season","week","recent_team"],keep="first")

def build_expanded_pregame(games, pbp=None, qbs=None):
    """Return one row per completed game with strictly prior-week features."""
    base=PredictorNFL_ML()
    f=base.construir_features_pregame(games)
    if f.empty:
        return f
    g=games.copy()
    g["season"]=pd.to_numeric(g.season,errors="coerce")
    g["week"]=pd.to_numeric(g.week,errors="coerce")
    g=g.dropna(subset=["season","week","home_score","away_score"]).sort_values(["season","week","game_id"])
    qb=_qb_week_rows(qbs)
    qb_index={}
    if not qb.empty:
        for (s,w),group in qb.groupby(["season","week"],sort=True):
            qb_index[(int(s),int(w))]=group
    pbp_pre=construir_pbp_pregame(g,pbp,metrics=list(PBP_METRICS)+list(EXTRA_PBP)) if pbp is not None and not pbp.empty else pd.DataFrame()
    if not pbp_pre.empty:
        f=f.merge(pbp_pre,on="game_id",how="left",validate="one_to_one")
    qhist=defaultdict(list)
    team_venue=defaultdict(list)
    team_opponent=defaultdict(list)
    rows=[]
    for (season,week),wg in g.groupby(["season","week"],sort=True):
        for _,r in wg.iterrows():
            row={"game_id":r.game_id}
            time=str(r.get("gametime",""))
            try:
                hour=int(time.split(":")[0])
                row["night_game"]=int(hour>=18)
                row["night_missing"]=0
            except (ValueError,IndexError):
                row["night_game"]=np.nan
                row["night_missing"]=1
            row["neutral_site"]=int(str(r.get("location","")).lower()=="neutral")
            for side,team,opp,venue in (("home",r.home_team,r.away_team,"home"),("away",r.away_team,r.home_team,"away")):
                prefix=side+"_"
                past=qhist[team]
                # Last known primary passer: proxy only; actual confirmed starters
                # require an independent timestamped pregame roster feed.
                starter=past[-1]["player_id"] if past else None
                player_games=[x for x in past if x["player_id"]==starter][-8:]
                row[prefix+"qb_known"]=int(bool(player_games))
                row[prefix+"qb_games"]=len(player_games)
                row[prefix+"qb_changed"]=int(bool(len(past)>1 and past[-1]["player_id"]!=past[-2]["player_id"]))
                for stat in QB_STATS:
                    vals=[x[stat] for x in player_games if pd.notna(x[stat])]
                    row[prefix+"qb_"+stat+"_4"]=float(np.mean(vals[-4:])) if vals else np.nan
                    row[prefix+"qb_"+stat+"_8"]=float(np.mean(vals)) if vals else np.nan
                if player_games:
                    sums={stat:sum(x[stat] for x in player_games if pd.notna(x[stat])) for stat in QB_STATS}
                    attempts=sums["attempts"]
                    for label,numer in (("completion_pct","completions"),("yards_per_attempt","passing_yards"),
                                        ("td_per_attempt","passing_tds"),("int_per_attempt","interceptions"),
                                        ("sacks_per_attempt","sacks"),("epa_per_attempt","passing_epa")):
                        row[prefix+"qb_"+label]=sums[numer]/attempts if attempts>0 else np.nan
                    matchup=[x for x in player_games if x["opponent"]==opp]
                    row[prefix+"qb_vs_opponent_games"]=len(matchup)
                    row[prefix+"qb_vs_opponent_epa"]=float(np.mean([x["passing_epa"] for x in matchup])) if matchup else np.nan
                else:
                    row[prefix+"qb_vs_opponent_games"]=0
                    row[prefix+"qb_vs_opponent_epa"]=np.nan
                    for label in QB_RATES:row[prefix+"qb_"+label]=np.nan
                venue_hist=[x for x in team_venue[team] if x["venue"]==venue][-8:]
                row[prefix+"venue_games"]=len(venue_hist)
                for stat in VENUE_STATS:
                    row[prefix+"venue_"+stat+"_8"]=float(np.mean([x[stat] for x in venue_hist])) if venue_hist else np.nan
                matchup_hist=[x for x in team_opponent[team] if x["opponent"]==opp][-4:]
                row[prefix+"vs_opponent_games"]=len(matchup_hist)
                row[prefix+"vs_opponent_margin"]=float(np.mean([x["margin"] for x in matchup_hist])) if matchup_hist else np.nan
            rows.append(row)
        # Update *after* generating all games in the week. Same-week leakage prohibited.
        current=qb_index.get((int(season),int(week)))
        if current is not None:
            game_opponents={}
            for _,r in wg.iterrows():
                game_opponents[r.home_team]=r.away_team
                game_opponents[r.away_team]=r.home_team
            for _,r in current.iterrows():
                team=r.recent_team
                if team not in game_opponents:continue
                entry={"player_id":r.player_id,"opponent":game_opponents[team]}
                entry.update({stat:r[stat] for stat in QB_STATS})
                qhist[team].append(entry)
        for _,r in wg.iterrows():
            hs=float(r.home_score);aws=float(r.away_score)
            for team,opp,venue,pf,pa in ((r.home_team,r.away_team,"home",hs,aws),(r.away_team,r.home_team,"away",aws,hs)):
                entry={"venue":venue,"opponent":opp,"pf":pf,"pa":pa,"margin":pf-pa,"total":pf+pa}
                team_venue[team].append(entry)
                team_opponent[team].append(entry)
    extras=pd.DataFrame(rows)
    # f only contains games with >=4 prior team games.
    return f.merge(extras,on="game_id",how="left",validate="one_to_one")
