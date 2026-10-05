from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote
import pandas as pd
import nfl_data_py as nfl
from google.auth.transport.requests import AuthorizedSession
from modules.nfl_google_sheets import SHEET_ID,_credentials,_request_json,_normalize_team,_parse_pick,_grade,_profit,_format_final_score,_espn_final_scores,_nflverse_final

WORKSHEET="NFL_Independent_Picks"

def settle_independent_pending():
    credentials,project_id=_credentials(); session=AuthorizedSession(credentials)
    base=f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values"
    values=_request_json(session,"GET",f"{base}/{quote(WORKSHEET+'!A:AA',safe='')}").get("values",[])
    pending=[]; seasons=set(); pairs=set()
    for rn,row in enumerate(values[1:],2):
        status=row[21].strip().upper() if len(row)>21 and row[21] else "PENDING"
        if status!="PENDING": continue
        try:
            season=int(float(row[1])); week=int(float(row[2])); odds=float(row[7]); stake=float(row[19] or 0)
            away,home=[_normalize_team(x) for x in row[3].split(" @ ",1)]
            parsed=_parse_pick(row[5],home,away)
        except Exception: continue
        if not parsed: continue
        pending.append((rn,season,week,away,home,parsed,odds,stake)); seasons.add(season); pairs.add((season,week))
    if not pending:return {"ok":True,"settled":0,"pending":0,"worksheet":WORKSHEET}
    try:schedules=nfl.import_schedules(sorted(seasons))
    except Exception:schedules=pd.DataFrame()
    fallback=_espn_final_scores(pairs); updates=[]; settled=0; left=0
    now=datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
    for rn,season,week,away,home,parsed,odds,stake in pending:
        hs=aws=None
        if not schedules.empty:
            m=schedules[(schedules.week==week)&(schedules.home_team.map(_normalize_team)==home)&(schedules.away_team.map(_normalize_team)==away)]
            if "season" in schedules:m=m[pd.to_numeric(m.season,errors="coerce")==season]
            if not m.empty and _nflverse_final(m.iloc[-1]):hs,aws=float(m.iloc[-1].home_score),float(m.iloc[-1].away_score)
        if hs is None:
            s=fallback.get((season,week,home,away))
            if s is not None:hs,aws=s
        if hs is None:left+=1;continue
        status=_grade(parsed,home,away,hs,aws)
        profit=_profit(stake,odds,status=="GANADA",push=status=="PUSH")
        score=_format_final_score(away,home,aws,hs)
        updates += [{"range":f"{WORKSHEET}!V{rn}:Y{rn}","values":[[status,score,profit,now]]},{"range":f"{WORKSHEET}!AA{rn}","values":[[now]]}]
        settled+=1
    if updates:_request_json(session,"POST",f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values:batchUpdate",json={"valueInputOption":"USER_ENTERED","data":updates})
    return {"ok":True,"settled":settled,"pending":left,"worksheet":WORKSHEET,"adc_project":project_id}
