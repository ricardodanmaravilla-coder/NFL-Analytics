"""Three independent research engines: ML, spread, total.
Train-only feature selection, market-specific targets and independent empirical MC.
No production deployment or betting actions.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, log_loss
from modules.nfl_ml_engine import PredictorNFL_ML
from backtest_nfl_feature_ablation import families

def feature_groups(frame):
    base=PredictorNFL_ML()._base_feature_names()
    groups=families(frame,base)
    # Candidate pools are market-specific, not a single global ranking.
    plans={
      "ML":("qb_efficiency","team_pbp","situational_pbp","venue","kickoff"),
      "SPREAD":("qb_efficiency","team_pbp","situational_pbp","venue","team_matchup"),
      "TOTAL":("qb_efficiency","team_pbp","situational_pbp","venue","kickoff"),
    }
    return {market:{"BASELINE":base,**{"BASE+"+name:sorted(set(base+groups[name]))
        for name in names if name in groups},
        "FULL":sorted(set(base+sum((groups.get(name,[]) for name in names),[])))}
        for market,names in plans.items()}

class IndependentMarketEngine:
    def __init__(self,market,features,seed=42):
        if market not in ("ML","SPREAD","TOTAL"):raise ValueError(market)
        self.market=market;self.features=list(features);self.seed=seed
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        self.model=(RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=8,
                    random_state=seed,n_jobs=2) if market=="ML" else
                    RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,
                    random_state=seed,n_jobs=2))
        self.residuals=None;self.fitted=False
    def target(self,frame):
        if self.market=="ML":return (frame.margen_local>0).astype(int)
        if self.market=="SPREAD":return frame.margen_local.astype(float)
        return frame.puntos_totales.astype(float)
    def fit(self,train):
        x=train[self.features].replace([np.inf,-np.inf],np.nan)
        self.usable=[c for c in self.features if x[c].notna().sum()>=20]
        if not self.usable:raise ValueError("No usable features")
        xx=self.imputer.fit_transform(x[self.usable]);y=self.target(train)
        self.model.fit(xx,y)
        self.fitted=True
        return self
    def predict(self,frame):
        if not self.fitted:raise ValueError("Engine not fitted")
        xx=self.imputer.transform(frame[self.usable].replace([np.inf,-np.inf],np.nan))
        if self.market=="ML":return self.model.predict_proba(xx)[:,list(self.model.classes_).index(1)]
        return self.model.predict(xx)
    def mc(self,predictions,calibration_residuals=None,line=None):
        """Independent market distributions. Residuals MUST be prior OOS errors."""
        p=np.asarray(predictions,dtype=float)
        if self.market=="ML":
            # Bernoulli MC with model-specific win probabilities; no shared margin MC.
            rng=np.random.default_rng(self.seed)
            return {"prob_home":p,"samples":rng.binomial(1,p[:,None],size=(len(p),5000))}
        if calibration_residuals is None or len(calibration_residuals)<30:
            raise ValueError("At least 30 earlier OOS residuals required for market MC")
        residuals=np.asarray(calibration_residuals,dtype=float)
        residuals=residuals[np.isfinite(residuals)]
        if len(residuals)<30:raise ValueError("Insufficient finite residuals")
        rng=np.random.default_rng(self.seed)
        draws=p[:,None]+rng.choice(residuals,size=(len(p),5000),replace=True)
        return {"samples":draws,"over_line_probability":None if line is None else
                np.mean(draws>np.asarray(line,dtype=float).reshape(-1,1),axis=1)}

def select_market_features(train,validation,market,candidates,minimum_gain=0.10):
    """Selection only on historical validation; caller holds out final season."""
    target=IndependentMarketEngine(market,[]).target(validation).to_numpy()
    scores={}
    for name,cols in candidates.items():
        model=IndependentMarketEngine(market,cols).fit(train)
        pred=model.predict(validation)
        score=(float(log_loss(target,np.clip(pred,1e-6,1-1e-6),labels=[0,1]))
               if market=="ML" else float(mean_absolute_error(target,pred)))
        scores[name]=score
    base=scores["BASELINE"]
    # Log-loss is on a different scale than points MAE.
    threshold=0.005 if market=="ML" else minimum_gain
    eligible=[k for k,v in scores.items() if k!="BASELINE" and base-v>=threshold]
    chosen=min(eligible,key=lambda k:(scores[k],len(candidates[k]))) if eligible else "BASELINE"
    return chosen,scores
