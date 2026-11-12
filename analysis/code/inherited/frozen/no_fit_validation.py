
"""Reconstruct design information and compare it without creating model objects."""
from contextlib import ExitStack
from hashlib import sha256
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
from patsy import dmatrices

from education_audit import education_revision

GROUPS = ("both_persist","sleep_only","pain_only","both_resolve")
CORE = ["Bgroup","age_c","sex","education_R","rural","prior_fall"]
EXTRA = ["mood9","chronic12","walk100","vision","memory"]
ANALYSIS = [
    "p0","s0","p1","s1","state0","state1","Bgroup","Y","outcome_status",
    "age","age_c","bd001","sex","education","education_R","rural","prior_fall",
    *EXTRA,"cluster","in_demo1","in_demo2","t0_before_t1","t1_before_t2","exact_t1_anchor",
]
MODELS = {"B_M1_R":CORE,"B_M2_R":CORE+EXTRA}


class FitBlocked(RuntimeError):
    """Deliberate trap raised before any regression estimation can begin."""


class NoFit:
    def __init__(self):
        self.calls = 0

    def reject(self,*args,**kwargs):
        self.calls += 1
        raise FitBlocked("Estimation API disabled in PRE-FEASIBILITY-1.2.1")

    def __enter__(self):
        from statsmodels.base.model import LikelihoodModel
        from statsmodels.discrete.discrete_model import DiscreteModel
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.regression.linear_model import RegressionModel
        self.stack = ExitStack()
        for obj,names in [(LikelihoodModel,("fit",)),(DiscreteModel,("fit",)),
                          (GLM,("fit","fit_regularized","_fit_irls")),
                          (RegressionModel,("fit",))]:
            for name in names:
                self.stack.enter_context(patch.object(obj,name,self.reject))
        return self

    def __exit__(self,*args):
        return self.stack.__exit__(*args)


def add_frozen_covariates(frame,cfg):
    out = frame.copy()
    out["education_R"] = education_revision(out.bd001,cfg["education_revision"]["mapping"])
    out["age_c"] = out.age - cfg["definitions"]["age_center"]
    return out


def derive_masks(d, invalid_t1=None):
    """Rederive from recorded states; do not inherit an archived entry mask."""
    p0s0 = d.p0.eq(1) & d.s0.eq(1)
    start = d.age.ge(60) & p0s0
    classified = start & d.in_demo1.eq(True) & d.p1.notna() & d.s1.notna()
    if invalid_t1 is not None:
        classified &= ~invalid_t1
    available = classified & d.Y.notna() & d.cluster.notna()
    m1 = available & d[CORE].notna().all(axis=1)
    m2 = available & d[CORE+EXTRA].notna().all(axis=1)
    return pd.DataFrame(dict(B_start=start,B_class=classified,B_M1_R=m1,B_M2_R=m2))


def make_design(d,mask,fields,formula):
    frame = d.loc[mask,["Y","cluster",*fields]].copy()
    frame["Bgroup"] = pd.Categorical(frame.Bgroup,categories=GROUPS)
    y,x = dmatrices(formula,frame,NA_action="raise",return_type="dataframe")
    assert x.index.equals(frame.index) and y.index.equals(frame.index)
    assert np.array_equal(y.to_numpy().ravel(),frame.Y.to_numpy())
    return frame,y,x


def same_columns(before,after):
    assert before.index.equals(after.index) and before.columns.equals(after.columns)
    out = []
    for name in before:
        a,b = before[name],after[name]
        equal = a.eq(b).fillna(False) | (a.isna() & b.isna())
        out.append(dict(field=name,rows=len(a),different_values=int((~equal).sum()),
                        different_missing_positions=int(a.isna().ne(b.isna()).sum()),
                        dtype_unchanged=str(a.dtype)==str(b.dtype)))
    return out


def matrix_digest(x):
    # No subject key or row of the matrix is written.
    h=sha256()
    h.update("\n".join(x.columns).encode("utf-8"))
    h.update(np.ascontiguousarray(x.to_numpy(),dtype="<f8").tobytes())
    return h.hexdigest()


def baseline_matches_history(name,frame,x,cfg,archived):
    counts = {"n":len(frame),"events":int(frame.Y.eq(1).sum()),
              "non_events":int(frame.Y.eq(0).sum()),"clusters":int(frame.cluster.nunique()),
              "p":x.shape[1],"rank":int(np.linalg.matrix_rank(x.to_numpy()))}
    rows = archived["model_inputs"]
    original = next(r for r in rows if r["model"]==name and r["phase"]=="REAL")
    counts_ok=all(int(original[k])==v for k,v in counts.items())
    fitted=next(r for r in archived["model_fits"] if r["model"]==name and r["attempt"]=="strict")
    f_ok=fitted["formula"]==cfg["model"]["formulas"][name] and fitted["fit_invoked"]=="True"
    # Evaluate archived coefficients on reconstructed X. This is a fixed
    # matrix multiplication, not a likelihood optimization or fit.
    coef=[r for r in archived["model_coefficients"] if r["model"]==name and r["attempt"]=="strict"]
    labels=[r["term"] for r in coef]
    columns_ok=labels==x.columns.tolist()
    cov=cfg["model"]["covariance"]
    cov_ok=all(str(cov[k])==original[k] for k in cov)
    deviations=[]
    prediction_ok=False
    if columns_ok:
        beta=np.array([float(r["coefficient"]) for r in coef])
        pred=pd.Series(np.exp(x.to_numpy() @ beta),index=frame.index)
        known=[r for r in archived["model_predictions"] if r["model"]==name and r["attempt"]=="strict"]
        assert {r["Bgroup"] for r in known}=={"ALL",*GROUPS}
        n_ok=True
        for row in known:
            values=pred if row["Bgroup"]=="ALL" else pred.loc[frame.Bgroup.eq(row["Bgroup"])]
            n_ok &= len(values)==int(row["n"])
            for stat, measured in [("min",values.min()),("median",values.median()),
                                   ("max",values.max()),("mean",values.mean())]:
                deviations.append(abs(float(row[stat])-measured))
        prediction_ok=bool(n_ok and max(deviations)<=1e-12)
    return dict(model=name,**counts,counts_match_archived=counts_ok,
                formula_match_archived=f_ok,terms_match_archived=columns_ok,
                covariance_options_match_archived=cov_ok,
                prediction_aggregates_match_archived=prediction_ok,
                max_prediction_stat_diff=float(max(deviations)) if deviations else None,
                status="PASS" if all((counts_ok,f_ok,columns_ok,cov_ok,prediction_ok)) else "FAIL")


def freeze_baseline(before,cfg,archived):
    masks=derive_masks(before)
    assert before[["B_start","B_class"]].equals(masks[["B_start","B_class"]])
    designs={name:make_design(before,masks[name],fields,cfg["model"]["formulas"][name])
             for name,fields in MODELS.items()}
    archive_checks=[baseline_matches_history(name,f,x,cfg,archived)
                    for name,(f,y,x) in designs.items()]
    return masks,designs,archive_checks


def compare_analysis(before,after,cfg,baseline,confirmed_death,conflict):
    left,designs,archive_checks=baseline
    right = derive_masks(after,confirmed_death | conflict)
    assert left.index.equals(right.index) and left.index.is_unique
    assert before[["B_start","B_class"]].equals(left[["B_start","B_class"]])
    relevant=left.B_start
    impact=left.B_class | left.B_M1_R | left.B_M2_R | (left.B_start & before.Y.notna())
    violations=dict(
        death_among_baseline_with_classification_or_valid_Y=int((confirmed_death & impact).sum()),
        conflict_among_baseline_with_classification_or_valid_Y=int((conflict & impact).sum()),
    )
    member_rows=[]
    for key in left:
        a,b=left[key],right[key]
        member_rows.append(dict(stage=key,before=int(a.sum()),after=int(b.sum()),
            only_before=int((a&~b).sum()),only_after=int((b&~a).sum()),
            flag_differences=int(a.ne(b).sum()),same_ordered_member_index=before.index[a].equals(after.index[b])))
    field_rows=[dict(scope="B_start",**r)
                for r in same_columns(before.loc[relevant,ANALYSIS],after.loc[relevant,ANALYSIS])]
    matrix_rows=[]
    for name,fields in MODELS.items():
        formula=cfg["model"]["formulas"][name]
        f0,y0,x0=designs[name]
        f1,y1,x1=make_design(after,right[name],fields,formula)
        intersection=f0.index.intersection(f1.index,sort=False)
        for row in same_columns(f0.loc[intersection],f1.loc[intersection]):
            field_rows.append(dict(scope=name,**row))
        col_equal=x0.columns.equals(x1.columns)
        row_equal=x0.index.equals(x1.index)
        same_shape=x0.shape==x1.shape
        matrix_rows.append(dict(model=name,formula=formula,reference="both_persist",
            category_order="|".join(GROUPS),
            covariance_settings=cfg["model"]["covariance"],
            rows_before=len(x0),rows_after=len(x1),parameters=x0.shape[1],
            columns_equal=col_equal,ordered_indices_equal=row_equal,
            design_values_equal=bool(same_shape and np.array_equal(x0.to_numpy(),x1.to_numpy())),
            response_values_equal=bool(np.array_equal(y0.to_numpy(),y1.to_numpy())),
            factorized_cluster_codes_equal=bool(np.array_equal(pd.factorize(f0.cluster,sort=False)[0],
                                                               pd.factorize(f1.cluster,sort=False)[0])),
            before_design_sha256=matrix_digest(x0),after_design_sha256=matrix_digest(x1),
            columns=x0.columns.tolist()))
    unchanged=(not any(violations.values())
        and all(r["flag_differences"]==0 and r["same_ordered_member_index"] for r in member_rows)
        and all(r["different_values"]==r["different_missing_positions"]==0 and r["dtype_unchanged"]
                for r in field_rows)
        and all(all(r[k] for k in ("columns_equal","ordered_indices_equal","design_values_equal",
                                  "response_values_equal","factorized_cluster_codes_equal")) for r in matrix_rows)
        and all(r["status"]=="PASS" for r in archive_checks))
    return right,dict(status="PASS" if unchanged else "FAIL",safe_to_reuse=unchanged,
                      member_checks=member_rows,field_checks=field_rows,matrix_checks=matrix_rows,
                      historical_model_checks=archive_checks,violations=violations)


def synthetic_guard_check():
    from statsmodels.genmod.generalized_linear_model import GLM
    with NoFit() as guard:
        try:
            GLM.fit(None)
        except FitBlocked:
            return guard.calls==1
    return False




