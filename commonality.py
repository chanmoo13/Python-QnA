import pandas as pd
import numpy as np
from pathlib import Path

CAT_COLS = ["ein_ecn_no", "ppid", "reticle_id", "Tag"]
STEP_COLS = ["step_seq", "LAYER"]   # 필요하면 ["step_seq"]로 변경


def prep(df, step_cols=STEP_COLS):
    d = df.copy()

    d["root_lot_id"] = d["root_lot_id"].astype(str)
    d["wafer_id"] = d["wafer_id"].astype(str)
    # wafer_id가 1, 2, 3 형태인데 실제로 01, 02, 03이면 아래처럼 맞춰줘
    # d["wafer_id"] = d["wafer_id"].astype(str).str.zfill(2)

    d["wafer_key"] = d["root_lot_id"] + "||" + d["wafer_id"]
    d["lotwf_key"] = d["lotwf"].astype(str) if "lotwf" in d.columns else d["wafer_key"]

    d["step_key"] = d[step_cols].astype(str).agg(" / ".join, axis=1)

    d["tkin_time"] = pd.to_datetime(d["tkin_time"], errors="coerce")
    d["tkout_time"] = pd.to_datetime(d["tkout_time"], errors="coerce")

    d["duration_hr"] = (d["tkout_time"] - d["tkin_time"]).dt.total_seconds() / 3600

    d = d.sort_values(
        ["wafer_key", "tkin_time", "tkout_time", "step_seq"],
        na_position="last"
    ).reset_index(drop=True)

    d["prev_tkout_time"] = d.groupby("wafer_key")["tkout_time"].shift(1)
    d["wait_hr"] = (d["tkin_time"] - d["prev_tkout_time"]).dt.total_seconds() / 3600

    return d


def mark_bad(d, bad_wafers):
    """
    bad_wafers 입력 예시

    1) tuple list
       bad_wafers = [("LOT_A", "01"), ("LOT_A", "07")]

    2) DataFrame
       bad_wafers = pd.DataFrame({
           "root_lot_id": ["LOT_A", "LOT_A"],
           "wafer_id": ["01", "07"]
       })

    3) lotwf 값 list
       bad_wafers = ["LOT_A_01", "LOT_A_07"]
    """
    bad_keys, bad_lotwf = set(), set()

    if isinstance(bad_wafers, pd.DataFrame):
        b = bad_wafers.copy()

        if {"root_lot_id", "wafer_id"}.issubset(b.columns):
            bad_keys |= set(
                b["root_lot_id"].astype(str) + "||" + b["wafer_id"].astype(str)
            )

        if "lotwf" in b.columns:
            bad_lotwf |= set(b["lotwf"].astype(str))

    else:
        for x in bad_wafers:
            if isinstance(x, tuple):
                bad_keys.add(str(x[0]) + "||" + str(x[1]))
            else:
                s = str(x)
                if "||" in s:
                    bad_keys.add(s)
                else:
                    bad_lotwf.add(s)

    d = d.copy()
    d["is_bad"] = d["wafer_key"].isin(bad_keys) | d["lotwf_key"].isin(bad_lotwf)

    if d.loc[d["is_bad"], "wafer_key"].nunique() == 0:
        raise ValueError("bad wafer가 매칭되지 않았습니다. lot/wafer 포맷 또는 lotwf 값을 확인하세요.")

    return d


def categorical_rare(
    d,
    cat_cols=CAT_COLS,
    max_global_lots=3,
    rare_global_lot_rate=0.05,
    max_lot_wafers=1,
    rare_within_lot_rate=0.20,
):
    """
    bad wafer가 특정 step에서
    1) 전체 lot 기준 rare value를 탔는지
    2) 같은 lot 내부에서 단독/소수 value를 탔는지
    탐지한다.
    """
    outs = []

    for attr in cat_cols:
        if attr not in d.columns:
            continue

        base = d[
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", attr, "is_bad"]
        ].copy()

        base[attr] = (
            base[attr]
            .astype(str)
            .replace({"nan": "<NA>", "None": "<NA>", "": "<BLANK>"})
        )
        base = base[base[attr].ne("<NA>")]

        if base.empty:
            continue

        # 같은 wafer가 같은 step/value를 여러 번 가진 경우 과대계산 방지
        bsv = base.drop_duplicates(["wafer_key", "step_key", attr])

        gval = bsv.groupby(["step_key", attr]).agg(
            global_wafers=("wafer_key", "nunique"),
            global_lots=("root_lot_id", "nunique"),
        ).reset_index()

        gtot = bsv.drop_duplicates(["wafer_key", "step_key"]).groupby("step_key").agg(
            global_step_wafers=("wafer_key", "nunique"),
            global_step_lots=("root_lot_id", "nunique"),
        ).reset_index()

        gval = gval.merge(gtot, on="step_key", how="left")
        gval["global_wafer_rate"] = gval["global_wafers"] / gval["global_step_wafers"]
        gval["global_lot_rate"] = gval["global_lots"] / gval["global_step_lots"]

        lval = bsv.groupby(["root_lot_id", "step_key", attr]).agg(
            lot_wafers=("wafer_key", "nunique"),
        ).reset_index()

        ltot = (
            bsv.drop_duplicates(["root_lot_id", "wafer_key", "step_key"])
            .groupby(["root_lot_id", "step_key"])
            .agg(lot_step_wafers=("wafer_key", "nunique"))
            .reset_index()
        )

        lval = lval.merge(ltot, on=["root_lot_id", "step_key"], how="left")
        lval["within_lot_wafer_rate"] = lval["lot_wafers"] / lval["lot_step_wafers"]

        badv = bsv[bsv["is_bad"]].drop_duplicates(
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", attr]
        )

        rep = badv.merge(gval, on=["step_key", attr], how="left")
        rep = rep.merge(lval, on=["root_lot_id", "step_key", attr], how="left")

        rep["attr"] = attr
        rep["value"] = rep[attr]

        rep["reason"] = ""

        rep.loc[
            rep["global_lots"].eq(1),
            "reason",
        ] += "전체 lot 중 이 lot에서만 관측; "

        rep.loc[
            (rep["global_lots"].le(max_global_lots))
            | (rep["global_lot_rate"].le(rare_global_lot_rate)),
            "reason",
        ] += "전체 lot 기준 rare value; "

        rep.loc[
            rep["lot_wafers"].le(max_lot_wafers),
            "reason",
        ] += "lot 내부 단독/소수 wafer value; "

        rep.loc[
            rep["within_lot_wafer_rate"].le(rare_within_lot_rate),
            "reason",
        ] += "lot 내부 wafer 비율 낮음; "

        rep["reason"] = rep["reason"].str.strip("; ")

        rep["score"] = 0
        rep.loc[rep["global_lots"].eq(1), "score"] += 5
        rep.loc[rep["global_lots"].le(max_global_lots), "score"] += 2
        rep.loc[rep["global_lot_rate"].le(rare_global_lot_rate), "score"] += 2
        rep.loc[rep["lot_wafers"].le(max_lot_wafers), "score"] += 4
        rep.loc[rep["within_lot_wafer_rate"].le(rare_within_lot_rate), "score"] += 2

        keep = [
            "score",
            "reason",
            "root_lot_id",
            "wafer_id",
            "wafer_key",
            "step_key",
            "attr",
            "value",
            "global_lots",
            "global_step_lots",
            "global_lot_rate",
            "global_wafers",
            "global_step_wafers",
            "global_wafer_rate",
            "lot_wafers",
            "lot_step_wafers",
            "within_lot_wafer_rate",
        ]

        outs.append(rep[rep["reason"].ne("")][keep])

    if not outs:
        return pd.DataFrame()

    return pd.concat(outs, ignore_index=True).sort_values("score", ascending=False)


def bad_value_enrichment(
    d,
    cat_cols=CAT_COLS,
    min_bad_support=1,
    min_lift=2.0,
    max_good_rate=0.30,
):
    """
    bad wafer들에 많이 나오고 good wafer들에는 적게 나오는 step/value 조합 탐지.
    bad wafer가 여러 장일 때 유용.
    """
    outs = []

    for attr in cat_cols:
        if attr not in d.columns:
            continue

        base = d[["root_lot_id", "wafer_key", "step_key", attr, "is_bad"]].copy()

        base[attr] = (
            base[attr]
            .astype(str)
            .replace({"nan": "<NA>", "None": "<NA>", "": "<BLANK>"})
        )
        base = base[base[attr].ne("<NA>")]

        bsv = base.drop_duplicates(["wafer_key", "step_key", attr])

        if bsv.empty:
            continue

        total = (
            bsv.drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(
                bad_wafers_at_step=("is_bad", "sum"),
                all_wafers_at_step=("wafer_key", "nunique"),
            )
            .reset_index()
        )

        total["good_wafers_at_step"] = (
            total["all_wafers_at_step"] - total["bad_wafers_at_step"]
        )

        val = bsv.groupby(["step_key", attr]).agg(
            bad_wafers=("is_bad", "sum"),
            all_wafers=("wafer_key", "nunique"),
            lots=("root_lot_id", "nunique"),
        ).reset_index()

        val["good_wafers"] = val["all_wafers"] - val["bad_wafers"]

        rep = val.merge(total, on="step_key", how="left")

        eps = 1e-9
        rep["bad_rate"] = rep["bad_wafers"] / (rep["bad_wafers_at_step"] + eps)
        rep["good_rate"] = rep["good_wafers"] / (rep["good_wafers_at_step"] + eps)
        rep["lift_bad_vs_good"] = (rep["bad_rate"] + eps) / (rep["good_rate"] + eps)

        rep["score"] = rep["bad_wafers"] * np.log1p(rep["lift_bad_vs_good"])
        rep["attr"] = attr
        rep["value"] = rep[attr]

        rep = rep[
            (rep["bad_wafers"] >= min_bad_support)
            & (rep["lift_bad_vs_good"] >= min_lift)
            & (rep["good_rate"] <= max_good_rate)
        ]

        keep = [
            "score",
            "step_key",
            "attr",
            "value",
            "bad_wafers",
            "bad_wafers_at_step",
            "bad_rate",
            "good_wafers",
            "good_wafers_at_step",
            "good_rate",
            "lift_bad_vs_good",
            "lots",
        ]

        outs.append(rep[keep])

    if not outs:
        return pd.DataFrame()

    return pd.concat(outs, ignore_index=True).sort_values("score", ascending=False)


def _q25(x):
    return x.quantile(0.25)


def _q75(x):
    return x.quantile(0.75)


def _rz(x, med, q1, q3):
    iqr = q3 - q1
    scale = iqr / 1.349
    z = (x - med) / scale
    return z.where(iqr > 0, np.where(x == med, 0, np.sign(x - med) * np.inf))


def time_outlier(
    d,
    metrics=("duration_hr", "wait_hr"),
    min_global_n=20,
    min_lot_n=5,
    z_cut=3.0,
):
    """
    bad wafer의 duration/wait이 전체 또는 lot 내부 대비 outlier인지 탐지.
    """
    outs = []

    for m in metrics:
        base = d[np.isfinite(d[m])].copy()

        if base.empty:
            continue

        g = base.groupby("step_key").agg(
            global_n=(m, "count"),
            global_median=(m, "median"),
            global_q1=(m, _q25),
            global_q3=(m, _q75),
        ).reset_index()

        l = base.groupby(["root_lot_id", "step_key"]).agg(
            lot_n=(m, "count"),
            lot_median=(m, "median"),
            lot_q1=(m, _q25),
            lot_q3=(m, _q75),
        ).reset_index()

        rep = base[base["is_bad"]][
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", m]
        ]

        rep = (
            rep.merge(g, on="step_key", how="left")
            .merge(l, on=["root_lot_id", "step_key"], how="left")
        )

        rep["global_robust_z"] = _rz(
            rep[m],
            rep["global_median"],
            rep["global_q1"],
            rep["global_q3"],
        )

        rep["lot_robust_z"] = _rz(
            rep[m],
            rep["lot_median"],
            rep["lot_q1"],
            rep["lot_q3"],
        )

        rep["metric"] = m
        rep["value"] = rep[m]

        mask_g = (rep["global_n"] >= min_global_n) & (
            rep["global_robust_z"].abs() >= z_cut
        )
        mask_l = (rep["lot_n"] >= min_lot_n) & (
            rep["lot_robust_z"].abs() >= z_cut
        )
        mask_neg = rep[m] < 0

        rep["reason"] = ""
        rep.loc[mask_g, "reason"] += "전체 기준 time outlier; "
        rep.loc[mask_l, "reason"] += "lot 내부 기준 time outlier; "
        rep.loc[mask_neg, "reason"] += "음수 시간: 순서/rework/time stamp 확인; "
        rep["reason"] = rep["reason"].str.strip("; ")

        rep["score"] = 0.0
        rep.loc[mask_g, "score"] += rep.loc[mask_g, "global_robust_z"].abs().clip(upper=20)
        rep.loc[mask_l, "score"] += rep.loc[mask_l, "lot_robust_z"].abs().clip(upper=20)
        rep.loc[mask_neg, "score"] += 5

        keep = [
            "score",
            "reason",
            "root_lot_id",
            "wafer_id",
            "wafer_key",
            "step_key",
            "metric",
            "value",
            "global_n",
            "global_median",
            "global_q1",
            "global_q3",
            "global_robust_z",
            "lot_n",
            "lot_median",
            "lot_q1",
            "lot_q3",
            "lot_robust_z",
        ]

        outs.append(rep[rep["reason"].ne("")][keep])

    if not outs:
        return pd.DataFrame()

    return pd.concat(outs, ignore_index=True).sort_values("score", ascending=False)


def route_anomaly(
    d,
    rare_step_within_lot_rate=0.20,
    expected_step_within_lot_rate=0.80,
):
    """
    bad wafer의 rare step, repeat/rework, missing expected step 탐지.
    """
    ws = (
        d.groupby(["root_lot_id", "wafer_id", "wafer_key", "step_key"])
        .size()
        .rename("visit_count")
        .reset_index()
    )

    lot_total = (
        d.groupby("root_lot_id")
        .agg(lot_total_wafers=("wafer_key", "nunique"))
        .reset_index()
    )

    lot_step = (
        ws.groupby(["root_lot_id", "step_key"])
        .agg(
            lot_wafers_with_step=("wafer_key", "nunique"),
            lot_visit_median=("visit_count", "median"),
            lot_visit_max=("visit_count", "max"),
        )
        .reset_index()
        .merge(lot_total, on="root_lot_id", how="left")
    )

    lot_step["lot_step_rate"] = (
        lot_step["lot_wafers_with_step"] / lot_step["lot_total_wafers"]
    )

    gtot_w = d["wafer_key"].nunique()
    gtot_l = d["root_lot_id"].nunique()

    global_step = ws.groupby("step_key").agg(
        global_wafers_with_step=("wafer_key", "nunique"),
        global_lots_with_step=("root_lot_id", "nunique"),
    ).reset_index()

    global_step["global_step_wafer_rate"] = (
        global_step["global_wafers_with_step"] / gtot_w
    )
    global_step["global_step_lot_rate"] = (
        global_step["global_lots_with_step"] / gtot_l
    )

    bad_keys = d[d["is_bad"]][
        ["root_lot_id", "wafer_id", "wafer_key"]
    ].drop_duplicates()

    bad_present = ws.merge(
        bad_keys,
        on=["root_lot_id", "wafer_id", "wafer_key"],
        how="inner",
    )

    present = (
        bad_present.merge(lot_step, on=["root_lot_id", "step_key"], how="left")
        .merge(global_step, on="step_key", how="left")
    )

    mask_repeat = (present["visit_count"] > 1) & (
        present["visit_count"] > present["lot_visit_median"]
    )
    mask_rare_lot = present["lot_step_rate"] <= rare_step_within_lot_rate
    mask_rare_global = present["global_step_lot_rate"] <= 0.05

    present["reason"] = ""
    present.loc[mask_repeat, "reason"] += "반복/rework 의심; "
    present.loc[mask_rare_lot, "reason"] += "lot 내부 소수 wafer만 진행한 step; "
    present.loc[mask_rare_global, "reason"] += "전체 lot 기준 rare step; "
    present["reason"] = present["reason"].str.strip("; ")

    present["issue_type"] = "present_step_anomaly"
    present["score"] = 0.0
    present.loc[mask_repeat, "score"] += (
        4 + present.loc[mask_repeat, "visit_count"] - present.loc[mask_repeat, "lot_visit_median"]
    )
    present.loc[mask_rare_lot, "score"] += 3
    present.loc[mask_rare_global, "score"] += 2

    present = present[present["reason"].ne("")].copy()

    # lot 대다수가 진행한 step을 bad wafer가 안 탄 경우
    expected = lot_step[
        lot_step["lot_step_rate"] >= expected_step_within_lot_rate
    ]

    miss = bad_keys.merge(expected, on="root_lot_id", how="left")

    miss = miss.merge(
        bad_present[["wafer_key", "step_key", "visit_count"]],
        on=["wafer_key", "step_key"],
        how="left",
    )

    miss = miss[miss["visit_count"].isna()].copy()

    if not miss.empty:
        miss = miss.merge(global_step, on="step_key", how="left")
        miss["visit_count"] = 0
        miss["issue_type"] = "missing_expected_step"
        miss["reason"] = "lot 대다수 wafer가 진행한 step을 bad wafer가 미진행/누락"
        miss["score"] = 4 + miss["lot_step_rate"]

    cols = [
        "score",
        "issue_type",
        "reason",
        "root_lot_id",
        "wafer_id",
        "wafer_key",
        "step_key",
        "visit_count",
        "lot_wafers_with_step",
        "lot_total_wafers",
        "lot_step_rate",
        "lot_visit_median",
        "lot_visit_max",
        "global_wafers_with_step",
        "global_lots_with_step",
        "global_step_wafer_rate",
        "global_step_lot_rate",
    ]

    frames = []

    if not present.empty:
        frames.append(present[cols])

    if not miss.empty:
        frames.append(miss[cols])

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values("score", ascending=False)


def run_anomaly(df, bad_wafers, out_dir="./wafer_anomaly_reports"):
    d = mark_bad(prep(df), bad_wafers)

    reports = {
        "categorical_rare": categorical_rare(d),
        "bad_value_enrichment": bad_value_enrichment(d),
        "time_outlier": time_outlier(d),
        "route_anomaly": route_anomaly(d),
    }

    top = []

    for name, r in reports.items():
        if r is not None and not r.empty:
            x = r.copy()
            x.insert(0, "report", name)
            top.append(x)

    reports["top_all"] = (
        pd.concat(top, ignore_index=True, sort=False).sort_values("score", ascending=False)
        if top
        else pd.DataFrame()
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for name, r in reports.items():
        if r is not None and not r.empty:
            r.to_csv(
                Path(out_dir) / f"{name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    return reports
    
    
    
# df = pd.read_csv("process_history.csv")

bad_wafers = [
    ("LOT_A", "01"),
    ("LOT_A", "07"),
    ("LOT_B", "13"),
]

reports = run_anomaly(df, bad_wafers)

# 전체 의심 이력 ranking
reports["top_all"].head(100)

# PPID / reticle / ECN / Tag가 rare한 경우
reports["categorical_rare"].head(50)

# bad wafer들에 공통적으로 많이 나온 특이 value
reports["bad_value_enrichment"].head(50)

# duration / wait time 이상치
reports["time_outlier"].head(50)

# repeat / rework / missing / rare step
reports["route_anomaly"].head(50)