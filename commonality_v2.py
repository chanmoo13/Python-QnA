# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

CAT_COLS = ["ein_ecn_no", "ppid", "reticle_id", "Tag"]
STEP_COLS = ["step_seq", "LAYER"]
EPS = 1e-9


def normalize_wafer_id(x, zfill=None):
    """
    wafer_id 비교용 정규화.
    - 1, 1.0 -> "1"
    - zfill=2이면 1 -> "01"
    """
    s = pd.Series(x).astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    if zfill is not None:
        s = s.str.zfill(zfill)

    return s.astype(str)


def _clean_value(x):
    s = pd.Series(x).astype("string").str.strip()
    s = s.fillna("<MISSING>")
    s = s.mask(
        s.str.lower().isin(["", "nan", "none", "nat", "<na>"]),
        "<MISSING>",
    )
    return s.astype(str)


def _join_unique(values, max_n=10):
    vals = []

    for v in values:
        if pd.isna(v):
            continue

        sv = str(v)

        if sv not in vals:
            vals.append(sv)

        if len(vals) >= max_n:
            break

    suffix = "" if len(vals) < max_n else " ..."
    return ", ".join(vals) + suffix


def _safe_rate(num, den):
    num = pd.Series(num, dtype="float64")
    den = pd.Series(den, dtype="float64")
    return num.div(den.where(den > 0, np.nan))


def _q25(x):
    return x.quantile(0.25)


def _q75(x):
    return x.quantile(0.75)


def _robust_z(x, med, q1, q3):
    x = pd.Series(x, dtype="float64")
    med = pd.Series(med, dtype="float64")
    q1 = pd.Series(q1, dtype="float64")
    q3 = pd.Series(q3, dtype="float64")

    iqr = q3 - q1
    scale = iqr / 1.349
    z = (x - med) / scale

    no_scale = scale.isna() | (scale <= 0)
    z = z.mask(no_scale & (x == med), 0.0)
    z = z.mask(no_scale & (x > med), np.inf)
    z = z.mask(no_scale & (x < med), -np.inf)

    return z


def prep(df, step_cols=STEP_COLS, wafer_zfill=None):
    d = df.copy()

    required = ["root_lot_id", "wafer_id", "tkin_time", "tkout_time"]
    missing = [c for c in required if c not in d.columns]

    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    missing_step = [c for c in step_cols if c not in d.columns]

    if missing_step:
        raise ValueError(f"step_cols에 지정한 컬럼이 없습니다: {missing_step}")

    d["root_lot_id"] = d["root_lot_id"].astype(str).str.strip()
    d["wafer_id"] = normalize_wafer_id(d["wafer_id"], zfill=wafer_zfill)

    d["wafer_key"] = d["root_lot_id"] + "||" + d["wafer_id"]

    if "lotwf" in d.columns:
        d["lotwf_key"] = d["lotwf"].astype(str).str.strip()
    else:
        d["lotwf_key"] = d["wafer_key"]

    d["step_key"] = d[list(step_cols)].astype(str).agg(" / ".join, axis=1)

    d["tkin_time"] = pd.to_datetime(d["tkin_time"], errors="coerce")
    d["tkout_time"] = pd.to_datetime(d["tkout_time"], errors="coerce")

    d["duration_hr"] = (
        d["tkout_time"] - d["tkin_time"]
    ).dt.total_seconds() / 3600

    sort_cols = [
        c for c in ["wafer_key", "tkin_time", "tkout_time", "step_seq"]
        if c in d.columns
    ]

    d = d.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    d["prev_tkout_time"] = d.groupby("wafer_key")["tkout_time"].shift(1)

    d["wait_hr"] = (
        d["tkin_time"] - d["prev_tkout_time"]
    ).dt.total_seconds() / 3600

    return d


def mark_bad(d, bad_wafers, wafer_zfill=None):
    """
    bad_wafers 입력 가능 형태

    1) tuple list
       [("LOT_A", 1), ("LOT_A", 7)]

    2) DataFrame
       columns: root_lot_id, wafer_id
       또는 lotwf

    3) lotwf list
       ["LOT_A_01", "LOT_A_07"]

    4) wafer_key list
       ["LOT_A||1", "LOT_A||7"]
    """
    bad_keys, bad_lotwf = set(), set()

    if isinstance(bad_wafers, pd.DataFrame):
        b = bad_wafers.copy()

        if {"root_lot_id", "wafer_id"}.issubset(b.columns):
            lot = b["root_lot_id"].astype(str).str.strip()
            wf = normalize_wafer_id(b["wafer_id"], zfill=wafer_zfill)
            bad_keys |= set(lot + "||" + wf)

        if "lotwf" in b.columns:
            bad_lotwf |= set(b["lotwf"].astype(str).str.strip())

    else:
        for x in bad_wafers:
            if isinstance(x, tuple):
                lot = str(x[0]).strip()
                wf = normalize_wafer_id([x[1]], zfill=wafer_zfill).iloc[0]
                bad_keys.add(lot + "||" + wf)
            else:
                s = str(x).strip()

                if "||" in s:
                    bad_keys.add(s)
                else:
                    bad_lotwf.add(s)

    d = d.copy()

    d["is_bad"] = (
        d["wafer_key"].isin(bad_keys)
        | d["lotwf_key"].isin(bad_lotwf)
    )

    matched = d.loc[d["is_bad"], "wafer_key"].nunique()

    if matched == 0:
        raise ValueError(
            "bad wafer가 매칭되지 않았습니다. lot/wafer 포맷 또는 lotwf 값을 확인하세요."
        )

    return d


def common_exact_value_anomaly(
    d,
    cat_cols=CAT_COLS,
    min_bad_support=2,
    min_bad_coverage=0.20,
    min_bad_step_rate=0.50,
    min_lift=2.0,
    max_good_step_rate=0.30,
    max_good_wafers=3,
    lift_cap_for_score=50,
    max_examples=10,
):
    """
    입력 bad wafer들에 공통으로 나타나는 exact value anomaly 탐지.

    예:
    같은 step에서 bad wafer 다수가 동일 PPID / reticle / ECN / Tag를 탔고,
    good wafer에서는 드문 경우.
    """
    total_bad_wafers = d.loc[d["is_bad"], "wafer_key"].nunique()
    total_good_wafers = d.loc[~d["is_bad"], "wafer_key"].nunique()
    total_bad_lots = d.loc[d["is_bad"], "root_lot_id"].nunique()

    eff_min_bad_support = max(1, min(min_bad_support, total_bad_wafers))

    outs = []

    for attr in cat_cols:
        if attr not in d.columns:
            continue

        base = d[
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", attr, "is_bad"]
        ].copy()

        base["value"] = _clean_value(base[attr])

        base = base[
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", "value", "is_bad"]
        ]

        bsv = base.drop_duplicates(["wafer_key", "step_key", "value"])

        if bsv.empty:
            continue

        keys = ["step_key", "value"]

        bad_counts = (
            bsv[bsv["is_bad"]]
            .groupby(keys)
            .agg(
                bad_wafers=("wafer_key", "nunique"),
                bad_lots=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        if bad_counts.empty:
            continue

        good_counts = (
            bsv[~bsv["is_bad"]]
            .groupby(keys)
            .agg(
                good_wafers=("wafer_key", "nunique"),
                good_lots=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        bad_step = (
            bsv[bsv["is_bad"]]
            .drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(
                bad_wafers_at_step=("wafer_key", "nunique"),
                bad_lots_at_step=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        good_step = (
            bsv[~bsv["is_bad"]]
            .drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(
                good_wafers_at_step=("wafer_key", "nunique"),
                good_lots_at_step=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        rep = bad_counts.merge(good_counts, on=keys, how="left")
        rep = rep.merge(bad_step, on="step_key", how="left")
        rep = rep.merge(good_step, on="step_key", how="left")

        count_cols = [
            "bad_wafers",
            "bad_lots",
            "good_wafers",
            "good_lots",
            "bad_wafers_at_step",
            "bad_lots_at_step",
            "good_wafers_at_step",
            "good_lots_at_step",
        ]

        for c in count_cols:
            if c in rep.columns:
                rep[c] = rep[c].fillna(0).astype(int)

        rep["coverage_bad_all"] = rep["bad_wafers"] / max(total_bad_wafers, 1)
        rep["bad_lot_coverage"] = rep["bad_lots"] / max(total_bad_lots, 1)

        rep["bad_step_rate"] = _safe_rate(
            rep["bad_wafers"],
            rep["bad_wafers_at_step"],
        )

        rep["good_step_rate"] = _safe_rate(
            rep["good_wafers"],
            rep["good_wafers_at_step"],
        )

        rep["good_all_rate"] = rep["good_wafers"] / max(total_good_wafers, 1)

        good_rate_for_score = rep["good_step_rate"].fillna(0.0)

        lift_for_score = (
            rep["bad_step_rate"].fillna(0.0) + EPS
        ) / (
            good_rate_for_score + EPS
        )

        rep["lift_bad_vs_good"] = np.where(
            rep["good_step_rate"].fillna(0) > 0,
            rep["bad_step_rate"] / rep["good_step_rate"],
            np.inf,
        )

        rep["lift_for_score"] = lift_for_score.clip(upper=lift_cap_for_score)

        # score는 원인 확률이 아니라 "공통 anomaly 우선순위 점수"
        rep["score"] = (
            60 * rep["coverage_bad_all"]
            + 35 * rep["bad_step_rate"].fillna(0)
            + 15 * np.log1p(rep["lift_for_score"])
            + 15 * (1 - good_rate_for_score.clip(0, 1))
            + 10 * rep["bad_lot_coverage"]
            + np.where(rep["good_wafers"].eq(0), 10, 0)
            + np.where(rep["bad_lots"].ge(2), 5, 0)
        )

        sample = bsv[bsv["is_bad"]].copy()

        sample["example_bad_wafers"] = (
            sample["root_lot_id"].astype(str)
            + ":"
            + sample["wafer_id"].astype(str)
        )

        sample_agg = (
            sample.groupby(keys)["example_bad_wafers"]
            .agg(lambda x: _join_unique(x, max_examples))
            .reset_index()
        )

        lot_agg = (
            sample.groupby(keys)["root_lot_id"]
            .agg(lambda x: _join_unique(x, max_examples))
            .reset_index()
            .rename(columns={"root_lot_id": "bad_lot_ids"})
        )

        rep = (
            rep.merge(sample_agg, on=keys, how="left")
            .merge(lot_agg, on=keys, how="left")
        )

        rep["attr"] = attr
        rep["issue_type"] = "common_exact_value"

        def _reason(row):
            parts = [
                f"입력 bad {int(row['bad_wafers'])}/{total_bad_wafers} wafer 공통({row['coverage_bad_all']:.1%})",
                f"해당 step을 탄 bad 기준 {row['bad_step_rate']:.1%}",
            ]

            if row["bad_lots"] >= 2:
                parts.append(f"{int(row['bad_lots'])}개 bad lot에 걸침")

            if np.isfinite(row["lift_bad_vs_good"]):
                parts.append(f"good 대비 {row['lift_bad_vs_good']:.1f}배")
            elif row["good_wafers_at_step"] == 0:
                parts.append("해당 step good baseline 없음")
            else:
                parts.append("good에서는 0%")

            if row["good_wafers"] == 0:
                parts.append("good wafer 관측 0")

            return "; ".join(parts)

        rep["reason"] = rep.apply(_reason, axis=1)

        keep_mask = (
            (rep["bad_wafers"] >= eff_min_bad_support)
            & (
                (rep["coverage_bad_all"] >= min_bad_coverage)
                | (rep["bad_step_rate"] >= min_bad_step_rate)
            )
            & (
                (rep["lift_for_score"] >= min_lift)
                | (good_rate_for_score <= max_good_step_rate)
                | (rep["good_wafers"] <= max_good_wafers)
            )
        )

        keep_cols = [
            "score",
            "issue_type",
            "reason",
            "step_key",
            "attr",
            "value",
            "bad_wafers",
            "bad_lots",
            "coverage_bad_all",
            "bad_lot_coverage",
            "bad_wafers_at_step",
            "bad_step_rate",
            "good_wafers",
            "good_lots",
            "good_wafers_at_step",
            "good_step_rate",
            "good_all_rate",
            "lift_bad_vs_good",
            "example_bad_wafers",
            "bad_lot_ids",
        ]

        outs.append(rep.loc[keep_mask, keep_cols])

    if not outs:
        return pd.DataFrame()

    return pd.concat(outs, ignore_index=True).sort_values(
        "score",
        ascending=False,
    )


def common_within_lot_rare_pattern(
    d,
    cat_cols=CAT_COLS,
    min_bad_support=2,
    min_bad_coverage=0.20,
    min_bad_step_flag_rate=0.50,
    max_lot_wafers=1,
    rare_within_lot_rate=0.20,
    max_global_lots=3,
    rare_global_lot_rate=0.05,
    max_examples=10,
):
    """
    exact value가 서로 달라도 잡기 위한 report.

    예:
    여러 bad wafer가 같은 step/attr에서 각각 자기 lot 내부에서는 단독/소수 value를 탔다.
    즉, value 자체보다 "lot 내부 rare value를 탔다는 패턴"이 공통인지 본다.
    """
    total_bad_wafers = d.loc[d["is_bad"], "wafer_key"].nunique()
    total_bad_lots = d.loc[d["is_bad"], "root_lot_id"].nunique()

    eff_min_bad_support = max(1, min(min_bad_support, total_bad_wafers))

    outs = []

    for attr in cat_cols:
        if attr not in d.columns:
            continue

        base = d[
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", attr, "is_bad"]
        ].copy()

        base["value"] = _clean_value(base[attr])

        base = base[
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", "value", "is_bad"]
        ]

        bsv = base.drop_duplicates(["wafer_key", "step_key", "value"])

        if bsv.empty or bsv[bsv["is_bad"]].empty:
            continue

        gv = (
            bsv.groupby(["step_key", "value"])
            .agg(
                global_wafers=("wafer_key", "nunique"),
                global_lots=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        gtot = (
            bsv.drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(
                global_step_wafers=("wafer_key", "nunique"),
                global_step_lots=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        lv = (
            bsv.groupby(["root_lot_id", "step_key", "value"])
            .agg(lot_wafers=("wafer_key", "nunique"))
            .reset_index()
        )

        ltot = (
            bsv.drop_duplicates(["root_lot_id", "wafer_key", "step_key"])
            .groupby(["root_lot_id", "step_key"])
            .agg(lot_step_wafers=("wafer_key", "nunique"))
            .reset_index()
        )

        m = bsv.merge(gv, on=["step_key", "value"], how="left")
        m = m.merge(gtot, on="step_key", how="left")
        m = m.merge(lv, on=["root_lot_id", "step_key", "value"], how="left")
        m = m.merge(ltot, on=["root_lot_id", "step_key"], how="left")

        m["within_lot_wafer_rate"] = _safe_rate(
            m["lot_wafers"],
            m["lot_step_wafers"],
        )

        m["global_lot_rate"] = _safe_rate(
            m["global_lots"],
            m["global_step_lots"],
        )

        m["flag_lot_single"] = m["lot_wafers"].le(max_lot_wafers)

        m["flag_lot_rare"] = m["within_lot_wafer_rate"].le(
            rare_within_lot_rate
        )

        m["flag_global_rare"] = (
            m["global_lots"].le(max_global_lots)
            | m["global_lot_rate"].le(rare_global_lot_rate)
        )

        m["flag_any"] = (
            m["flag_lot_single"]
            | m["flag_lot_rare"]
            | m["flag_global_rare"]
        )

        bad_flag = m[m["is_bad"] & m["flag_any"]].copy()

        if bad_flag.empty:
            continue

        bad_flag_unique = bad_flag.drop_duplicates(["wafer_key", "step_key"])

        rep = (
            bad_flag_unique.groupby("step_key")
            .agg(
                bad_wafers=("wafer_key", "nunique"),
                bad_lots=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        step_den = (
            bsv[bsv["is_bad"]]
            .drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(
                bad_wafers_at_step=("wafer_key", "nunique"),
                bad_lots_at_step=("root_lot_id", "nunique"),
            )
            .reset_index()
        )

        stats = (
            bad_flag.groupby("step_key")
            .agg(
                lot_single_events=("flag_lot_single", "sum"),
                lot_rare_events=("flag_lot_rare", "sum"),
                global_rare_events=("flag_global_rare", "sum"),
                median_within_lot_rate=("within_lot_wafer_rate", "median"),
                min_within_lot_rate=("within_lot_wafer_rate", "min"),
                median_global_lot_rate=("global_lot_rate", "median"),
                min_global_lot_rate=("global_lot_rate", "min"),
            )
            .reset_index()
        )

        sample = bad_flag.copy()

        sample["example_bad_wafers"] = (
            sample["root_lot_id"].astype(str)
            + ":"
            + sample["wafer_id"].astype(str)
            + "="
            + sample["value"].astype(str)
        )

        sample_agg = (
            sample.groupby("step_key")["example_bad_wafers"]
            .agg(lambda x: _join_unique(x, max_examples))
            .reset_index()
        )

        lot_agg = (
            bad_flag_unique.groupby("step_key")["root_lot_id"]
            .agg(lambda x: _join_unique(x, max_examples))
            .reset_index()
            .rename(columns={"root_lot_id": "bad_lot_ids"})
        )

        rep = rep.merge(step_den, on="step_key", how="left")
        rep = rep.merge(stats, on="step_key", how="left")
        rep = rep.merge(sample_agg, on="step_key", how="left")
        rep = rep.merge(lot_agg, on="step_key", how="left")

        rep["coverage_bad_all"] = rep["bad_wafers"] / max(total_bad_wafers, 1)
        rep["bad_lot_coverage"] = rep["bad_lots"] / max(total_bad_lots, 1)

        rep["bad_step_flag_rate"] = _safe_rate(
            rep["bad_wafers"],
            rep["bad_wafers_at_step"],
        )

        rep["score"] = (
            65 * rep["coverage_bad_all"]
            + 35 * rep["bad_step_flag_rate"].fillna(0)
            + 15 * rep["bad_lot_coverage"]
            + 10 * (1 - rep["median_within_lot_rate"].fillna(1).clip(0, 1))
            + 5 * (1 - rep["median_global_lot_rate"].fillna(1).clip(0, 1))
            + np.where(rep["bad_lots"].ge(2), 5, 0)
        )

        rep["attr"] = attr
        rep["issue_type"] = "common_within_lot_rare_pattern"
        rep["value"] = "<pattern: value may differ by lot/wafer>"

        def _reason(row):
            parts = [
                f"입력 bad {int(row['bad_wafers'])}/{total_bad_wafers} wafer가 같은 step/{attr}에서 lot 내부 rare value 패턴({row['coverage_bad_all']:.1%})",
                f"해당 step을 탄 bad 기준 {row['bad_step_flag_rate']:.1%}",
                "exact value가 서로 달라도 잡는 패턴",
            ]

            if row["bad_lots"] >= 2:
                parts.append(f"{int(row['bad_lots'])}개 bad lot에 걸침")

            return "; ".join(parts)

        rep["reason"] = rep.apply(_reason, axis=1)

        keep_mask = (
            (rep["bad_wafers"] >= eff_min_bad_support)
            & (
                (rep["coverage_bad_all"] >= min_bad_coverage)
                | (rep["bad_step_flag_rate"] >= min_bad_step_flag_rate)
            )
        )

        keep_cols = [
            "score",
            "issue_type",
            "reason",
            "step_key",
            "attr",
            "value",
            "bad_wafers",
            "bad_lots",
            "coverage_bad_all",
            "bad_lot_coverage",
            "bad_wafers_at_step",
            "bad_step_flag_rate",
            "lot_single_events",
            "lot_rare_events",
            "global_rare_events",
            "median_within_lot_rate",
            "min_within_lot_rate",
            "median_global_lot_rate",
            "min_global_lot_rate",
            "example_bad_wafers",
            "bad_lot_ids",
        ]

        outs.append(rep.loc[keep_mask, keep_cols])

    if not outs:
        return pd.DataFrame()

    return pd.concat(outs, ignore_index=True).sort_values(
        "score",
        ascending=False,
    )


def _agg_route_issue(
    subset,
    issue_type,
    reason_text,
    total_bad_wafers,
    total_bad_lots,
    den,
    min_bad_support,
    min_bad_coverage,
    min_issue_rate,
    max_examples,
):
    if subset.empty:
        return pd.DataFrame()

    s = subset.drop_duplicates(["wafer_key", "step_key"]).copy()

    rep = (
        s.groupby("step_key")
        .agg(
            bad_wafers=("wafer_key", "nunique"),
            bad_lots=("root_lot_id", "nunique"),
            median_visit_count=("visit_count", "median"),
            max_visit_count=("visit_count", "max"),
            median_lot_step_rate=("lot_step_rate", "median"),
            median_global_step_lot_rate=("global_step_lot_rate", "median"),
            median_lot_visit_median=("lot_visit_median", "median"),
        )
        .reset_index()
    )

    sample = s.copy()

    sample["example_bad_wafers"] = (
        sample["root_lot_id"].astype(str)
        + ":"
        + sample["wafer_id"].astype(str)
    )

    sample_agg = (
        sample.groupby("step_key")["example_bad_wafers"]
        .agg(lambda x: _join_unique(x, max_examples))
        .reset_index()
    )

    lot_agg = (
        sample.groupby("step_key")["root_lot_id"]
        .agg(lambda x: _join_unique(x, max_examples))
        .reset_index()
        .rename(columns={"root_lot_id": "bad_lot_ids"})
    )

    rep = rep.merge(den, on="step_key", how="left")
    rep = rep.merge(sample_agg, on="step_key", how="left")
    rep = rep.merge(lot_agg, on="step_key", how="left")

    rep["bad_wafers_den"] = rep["bad_wafers_den"].fillna(
        rep["bad_wafers"]
    ).astype(int)

    rep["coverage_bad_all"] = rep["bad_wafers"] / max(total_bad_wafers, 1)
    rep["bad_lot_coverage"] = rep["bad_lots"] / max(total_bad_lots, 1)

    rep["bad_issue_rate"] = _safe_rate(
        rep["bad_wafers"],
        rep["bad_wafers_den"],
    )

    rep["issue_type"] = issue_type

    rep["score"] = (
        60 * rep["coverage_bad_all"]
        + 30 * rep["bad_issue_rate"].fillna(0)
        + 15 * rep["bad_lot_coverage"]
    )

    if issue_type == "repeat_or_rework":
        rep["score"] += 5 * (
            rep["median_visit_count"] - rep["median_lot_visit_median"]
        ).fillna(0).clip(lower=0, upper=10)

    elif issue_type == "missing_expected_step":
        rep["score"] += 20 * rep["median_lot_step_rate"].fillna(0).clip(0, 1)

    elif issue_type == "rare_step_within_lot":
        rep["score"] += 15 * (
            1 - rep["median_lot_step_rate"].fillna(1).clip(0, 1)
        )

    elif issue_type == "rare_step_global":
        rep["score"] += 10 * (
            1 - rep["median_global_step_lot_rate"].fillna(1).clip(0, 1)
        )

    def _reason(row):
        parts = [
            reason_text,
            f"입력 bad {int(row['bad_wafers'])}/{total_bad_wafers} wafer 해당({row['coverage_bad_all']:.1%})",
            f"적용 가능 bad 기준 {row['bad_issue_rate']:.1%}",
        ]

        if row["bad_lots"] >= 2:
            parts.append(f"{int(row['bad_lots'])}개 bad lot에 걸침")

        return "; ".join(parts)

    rep["reason"] = rep.apply(_reason, axis=1)

    keep_mask = (
        (rep["bad_wafers"] >= min_bad_support)
        & (
            (rep["coverage_bad_all"] >= min_bad_coverage)
            | (rep["bad_issue_rate"] >= min_issue_rate)
        )
    )

    keep_cols = [
        "score",
        "issue_type",
        "reason",
        "step_key",
        "bad_wafers",
        "bad_lots",
        "coverage_bad_all",
        "bad_lot_coverage",
        "bad_wafers_den",
        "bad_issue_rate",
        "median_visit_count",
        "max_visit_count",
        "median_lot_visit_median",
        "median_lot_step_rate",
        "median_global_step_lot_rate",
        "example_bad_wafers",
        "bad_lot_ids",
    ]

    return rep.loc[keep_mask, keep_cols]


def common_route_anomaly(
    d,
    min_bad_support=2,
    min_bad_coverage=0.20,
    min_issue_rate=0.50,
    rare_step_within_lot_rate=0.20,
    expected_step_within_lot_rate=0.80,
    rare_step_global_lot_rate=0.05,
    max_examples=10,
):
    """
    입력 bad wafer들에 공통적인 route anomaly 탐지.

    - 반복/rework
    - lot 내부 소수 wafer만 진행한 step
    - 전체 lot 기준 rare step
    - lot 대다수가 진행한 step을 bad wafer들이 공통적으로 누락
    """
    total_bad_wafers = d.loc[d["is_bad"], "wafer_key"].nunique()
    total_bad_lots = d.loc[d["is_bad"], "root_lot_id"].nunique()

    eff_min_bad_support = max(1, min(min_bad_support, total_bad_wafers))

    ws = (
        d.groupby(["root_lot_id", "wafer_id", "wafer_key", "step_key"])
        .size()
        .rename("visit_count")
        .reset_index()
    )

    wafer_status = d[["wafer_key", "is_bad"]].drop_duplicates()
    ws = ws.merge(wafer_status, on="wafer_key", how="left")

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
        lot_step["lot_wafers_with_step"]
        / lot_step["lot_total_wafers"].clip(lower=1)
    )

    gtot_w = max(ws["wafer_key"].nunique(), 1)
    gtot_l = max(ws["root_lot_id"].nunique(), 1)

    global_step = (
        ws.groupby("step_key")
        .agg(
            global_wafers_with_step=("wafer_key", "nunique"),
            global_lots_with_step=("root_lot_id", "nunique"),
        )
        .reset_index()
    )

    global_step["global_step_wafer_rate"] = (
        global_step["global_wafers_with_step"] / gtot_w
    )

    global_step["global_step_lot_rate"] = (
        global_step["global_lots_with_step"] / gtot_l
    )

    bad_present = ws[ws["is_bad"]].copy()

    bad_present = bad_present.merge(
        lot_step,
        on=["root_lot_id", "step_key"],
        how="left",
    )

    bad_present = bad_present.merge(
        global_step,
        on="step_key",
        how="left",
    )

    bad_present["flag_repeat"] = (
        (bad_present["visit_count"] > 1)
        & (
            bad_present["visit_count"]
            > bad_present["lot_visit_median"].fillna(1)
        )
    )

    bad_present["flag_rare_step_within_lot"] = bad_present[
        "lot_step_rate"
    ].le(rare_step_within_lot_rate)

    bad_present["flag_rare_step_global"] = bad_present[
        "global_step_lot_rate"
    ].le(rare_step_global_lot_rate)

    den_present = (
        bad_present.groupby("step_key")
        .agg(bad_wafers_den=("wafer_key", "nunique"))
        .reset_index()
    )

    frames = []

    frames.append(
        _agg_route_issue(
            bad_present[bad_present["flag_repeat"]],
            "repeat_or_rework",
            "반복/rework 의심 step이 bad wafer들에 공통",
            total_bad_wafers,
            total_bad_lots,
            den_present,
            eff_min_bad_support,
            min_bad_coverage,
            min_issue_rate,
            max_examples,
        )
    )

    frames.append(
        _agg_route_issue(
            bad_present[bad_present["flag_rare_step_within_lot"]],
            "rare_step_within_lot",
            "lot 내부 소수 wafer만 진행한 step이 bad wafer들에 공통",
            total_bad_wafers,
            total_bad_lots,
            den_present,
            eff_min_bad_support,
            min_bad_coverage,
            min_issue_rate,
            max_examples,
        )
    )

    frames.append(
        _agg_route_issue(
            bad_present[bad_present["flag_rare_step_global"]],
            "rare_step_global",
            "전체 lot 기준 rare step이 bad wafer들에 공통",
            total_bad_wafers,
            total_bad_lots,
            den_present,
            eff_min_bad_support,
            min_bad_coverage,
            min_issue_rate,
            max_examples,
        )
    )

    bad_keys = d[d["is_bad"]][
        ["root_lot_id", "wafer_id", "wafer_key"]
    ].drop_duplicates()

    expected = lot_step[
        lot_step["lot_step_rate"] >= expected_step_within_lot_rate
    ].copy()

    if not expected.empty:
        expected_bad = bad_keys.merge(expected, on="root_lot_id", how="inner")
        expected_bad = expected_bad.merge(global_step, on="step_key", how="left")

        present_keys = ws[["wafer_key", "step_key", "visit_count"]].copy()

        miss = expected_bad.merge(
            present_keys,
            on=["wafer_key", "step_key"],
            how="left",
        )

        miss = miss[miss["visit_count"].isna()].copy()
        miss["visit_count"] = 0

        den_missing = (
            expected_bad.groupby("step_key")
            .agg(bad_wafers_den=("wafer_key", "nunique"))
            .reset_index()
        )

        frames.append(
            _agg_route_issue(
                miss,
                "missing_expected_step",
                "lot 대다수가 진행한 expected step을 bad wafer들이 공통적으로 미진행/누락",
                total_bad_wafers,
                total_bad_lots,
                den_missing,
                eff_min_bad_support,
                min_bad_coverage,
                min_issue_rate,
                max_examples,
            )
        )

    frames = [x for x in frames if x is not None and not x.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(
        "score",
        ascending=False,
    )


def _agg_time_flags(
    flagged,
    total_bad_wafers,
    total_bad_lots,
    den,
    min_bad_support,
    min_bad_coverage,
    min_issue_rate,
    max_examples,
):
    if flagged.empty:
        return pd.DataFrame()

    s = flagged.drop_duplicates(
        ["wafer_key", "step_key", "metric", "basis", "direction"]
    ).copy()

    keys = ["step_key", "metric", "basis", "direction"]

    rep = (
        s.groupby(keys)
        .agg(
            bad_wafers=("wafer_key", "nunique"),
            bad_lots=("root_lot_id", "nunique"),
            median_value=("value", "median"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            median_z=("z", "median"),
            median_abs_z=("abs_z", "median"),
            max_abs_z=("abs_z", "max"),
        )
        .reset_index()
    )

    sample = s.copy()

    sample["example_bad_wafers"] = (
        sample["root_lot_id"].astype(str)
        + ":"
        + sample["wafer_id"].astype(str)
        + "="
        + sample["value"].round(3).astype(str)
    )

    sample_agg = (
        sample.groupby(keys)["example_bad_wafers"]
        .agg(lambda x: _join_unique(x, max_examples))
        .reset_index()
    )

    lot_agg = (
        sample.groupby(keys)["root_lot_id"]
        .agg(lambda x: _join_unique(x, max_examples))
        .reset_index()
        .rename(columns={"root_lot_id": "bad_lot_ids"})
    )

    rep = rep.merge(den, on="step_key", how="left")
    rep = rep.merge(sample_agg, on=keys, how="left")
    rep = rep.merge(lot_agg, on=keys, how="left")

    rep["bad_wafers_den"] = rep["bad_wafers_den"].fillna(
        rep["bad_wafers"]
    ).astype(int)

    rep["coverage_bad_all"] = rep["bad_wafers"] / max(total_bad_wafers, 1)
    rep["bad_lot_coverage"] = rep["bad_lots"] / max(total_bad_lots, 1)

    rep["bad_issue_rate"] = _safe_rate(
        rep["bad_wafers"],
        rep["bad_wafers_den"],
    )

    z_component = (
        rep["median_abs_z"]
        .replace([np.inf, -np.inf], 20)
        .fillna(0)
        .clip(0, 20)
    )

    rep["score"] = (
        60 * rep["coverage_bad_all"]
        + 30 * rep["bad_issue_rate"].fillna(0)
        + 15 * rep["bad_lot_coverage"]
        + 4 * z_component
    )

    rep.loc[rep["basis"].eq("negative_time"), "score"] += 20

    rep["issue_type"] = "common_time_outlier"

    def _reason(row):
        if row["basis"] == "negative_time":
            direction_txt = "음수 time"
        else:
            direction_txt = "긴" if row["direction"] == "high" else "짧은"

        return (
            f"{row['metric']} {row['basis']} 기준 {direction_txt} outlier가 "
            f"입력 bad {int(row['bad_wafers'])}/{total_bad_wafers} wafer에 공통({row['coverage_bad_all']:.1%}); "
            f"적용 가능 bad 기준 {row['bad_issue_rate']:.1%}; "
            f"median_abs_z={row['median_abs_z']:.2f}"
        )

    rep["reason"] = rep.apply(_reason, axis=1)

    keep_mask = (
        (rep["bad_wafers"] >= min_bad_support)
        & (
            (rep["coverage_bad_all"] >= min_bad_coverage)
            | (rep["bad_issue_rate"] >= min_issue_rate)
        )
    )

    keep_cols = [
        "score",
        "issue_type",
        "reason",
        "step_key",
        "metric",
        "basis",
        "direction",
        "bad_wafers",
        "bad_lots",
        "coverage_bad_all",
        "bad_lot_coverage",
        "bad_wafers_den",
        "bad_issue_rate",
        "median_value",
        "min_value",
        "max_value",
        "median_z",
        "median_abs_z",
        "max_abs_z",
        "example_bad_wafers",
        "bad_lot_ids",
    ]

    return rep.loc[keep_mask, keep_cols]


def common_time_anomaly(
    d,
    metrics=("duration_hr", "wait_hr"),
    min_bad_support=2,
    min_bad_coverage=0.20,
    min_issue_rate=0.50,
    min_global_n=20,
    min_lot_n=5,
    z_cut=3.0,
    max_examples=10,
):
    """
    입력 bad wafer들에서 공통적으로 duration / wait time outlier가 발생한 step 탐지.
    """
    total_bad_wafers = d.loc[d["is_bad"], "wafer_key"].nunique()
    total_bad_lots = d.loc[d["is_bad"], "root_lot_id"].nunique()

    eff_min_bad_support = max(1, min(min_bad_support, total_bad_wafers))

    frames = []

    for m in metrics:
        if m not in d.columns:
            continue

        base = d.copy()
        base[m] = pd.to_numeric(base[m], errors="coerce")
        base = base[np.isfinite(base[m])].copy()

        if base.empty or base[base["is_bad"]].empty:
            continue

        g = (
            base.groupby("step_key")
            .agg(
                global_n=(m, "count"),
                global_median=(m, "median"),
                global_q1=(m, _q25),
                global_q3=(m, _q75),
            )
            .reset_index()
        )

        l = (
            base.groupby(["root_lot_id", "step_key"])
            .agg(
                lot_n=(m, "count"),
                lot_median=(m, "median"),
                lot_q1=(m, _q25),
                lot_q3=(m, _q75),
            )
            .reset_index()
        )

        rep = base[base["is_bad"]][
            ["root_lot_id", "wafer_id", "wafer_key", "step_key", m]
        ].copy()

        rep = rep.merge(g, on="step_key", how="left")
        rep = rep.merge(l, on=["root_lot_id", "step_key"], how="left")

        rep["global_z"] = _robust_z(
            rep[m],
            rep["global_median"],
            rep["global_q1"],
            rep["global_q3"],
        )

        rep["lot_z"] = _robust_z(
            rep[m],
            rep["lot_median"],
            rep["lot_q1"],
            rep["lot_q3"],
        )

        den = (
            rep.drop_duplicates(["wafer_key", "step_key"])
            .groupby("step_key")
            .agg(bad_wafers_den=("wafer_key", "nunique"))
            .reset_index()
        )

        flags = []

        gf = rep[
            (rep["global_n"] >= min_global_n)
            & (rep["global_z"].abs() >= z_cut)
        ].copy()

        if not gf.empty:
            gf["metric"] = m
            gf["basis"] = "global"
            gf["z"] = gf["global_z"]
            gf["abs_z"] = gf["z"].abs()
            gf["direction"] = np.where(gf["z"] >= 0, "high", "low")
            gf["value"] = gf[m]
            flags.append(gf)

        lf = rep[
            (rep["lot_n"] >= min_lot_n)
            & (rep["lot_z"].abs() >= z_cut)
        ].copy()

        if not lf.empty:
            lf["metric"] = m
            lf["basis"] = "lot"
            lf["z"] = lf["lot_z"]
            lf["abs_z"] = lf["z"].abs()
            lf["direction"] = np.where(lf["z"] >= 0, "high", "low")
            lf["value"] = lf[m]
            flags.append(lf)

        nf = rep[rep[m] < 0].copy()

        if not nf.empty:
            nf["metric"] = m
            nf["basis"] = "negative_time"
            nf["z"] = np.nan
            nf["abs_z"] = 20.0
            nf["direction"] = "negative"
            nf["value"] = nf[m]
            flags.append(nf)

        if flags:
            flagged = pd.concat(flags, ignore_index=True, sort=False)

            frames.append(
                _agg_time_flags(
                    flagged,
                    total_bad_wafers,
                    total_bad_lots,
                    den,
                    eff_min_bad_support,
                    min_bad_coverage,
                    min_issue_rate,
                    max_examples,
                )
            )

    frames = [x for x in frames if x is not None and not x.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(
        "score",
        ascending=False,
    )


def build_top_common(common_reports):
    frames = []

    for name, r in common_reports.items():
        if r is None or r.empty:
            continue

        x = r.copy()
        x.insert(0, "report", name)
        frames.append(x)

    if not frames:
        return pd.DataFrame()

    top = pd.concat(frames, ignore_index=True, sort=False)

    top["score_rank_pct_by_report"] = (
        top.groupby("report")["score"].rank(pct=True, ascending=True)
    )

    top = top.sort_values(
        ["score", "score_rank_pct_by_report"],
        ascending=False,
    ).reset_index(drop=True)

    return top


def _not_missing(v):
    if v is None:
        return False

    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    return str(v) != "nan"


def _label_top_issue(row):
    label = str(row.get("report", ""))

    if _not_missing(row.get("issue_type", None)):
        label += "/" + str(row.get("issue_type"))

    if _not_missing(row.get("attr", None)):
        label += "/" + str(row.get("attr"))

        if _not_missing(row.get("value", None)):
            label += "=" + str(row.get("value"))

    if _not_missing(row.get("metric", None)):
        label += "/" + str(row.get("metric"))

    if _not_missing(row.get("basis", None)):
        label += "/" + str(row.get("basis"))

    if _not_missing(row.get("direction", None)):
        label += "/" + str(row.get("direction"))

    label += f"({row.get('score', 0):.1f})"

    return label


def make_common_step_summary(top_common, max_issues_per_step=5):
    """
    같은 step_key가 여러 common report에서 반복 검출되는지 요약.
    이 report를 먼저 보면 어떤 step을 우선 확인할지 빠르게 볼 수 있다.
    """
    if top_common is None or top_common.empty:
        return pd.DataFrame()

    tmp = top_common.copy()

    for c in ["bad_wafers", "coverage_bad_all", "bad_lots"]:
        if c not in tmp.columns:
            tmp[c] = np.nan

    base = (
        tmp.groupby("step_key")
        .agg(
            max_score=("score", "max"),
            issue_count=("score", "count"),
            report_count=("report", "nunique"),
            max_bad_wafers=("bad_wafers", "max"),
            max_bad_lots=("bad_lots", "max"),
            max_coverage_bad_all=("coverage_bad_all", "max"),
        )
        .reset_index()
    )

    topn = (
        tmp.sort_values("score", ascending=False)
        .groupby("step_key")
        .head(max_issues_per_step)
    )

    topn_score = (
        topn.groupby("step_key")["score"]
        .sum()
        .rename("step_top_issue_score_sum")
        .reset_index()
    )

    issue_labels = topn.copy()
    issue_labels["issue_label"] = issue_labels.apply(_label_top_issue, axis=1)

    issue_labels = (
        issue_labels.groupby("step_key")["issue_label"]
        .agg(lambda x: " | ".join(x))
        .rename("top_issues")
        .reset_index()
    )

    summary = (
        base.merge(topn_score, on="step_key", how="left")
        .merge(issue_labels, on="step_key", how="left")
    )

    summary["step_common_score"] = (
        summary["step_top_issue_score_sum"].fillna(0)
        + 10 * summary["report_count"].fillna(0)
        + 20 * summary["max_coverage_bad_all"].fillna(0)
    )

    summary = summary.sort_values(
        "step_common_score",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def run_common_anomaly(
    df,
    bad_wafers,
    out_dir="./wafer_common_anomaly_reports",
    wafer_zfill=None,
    step_cols=STEP_COLS,
    cat_cols=CAT_COLS,
    min_bad_support=2,
    min_bad_coverage=0.20,
    save_csv=True,
):
    """
    메인 실행 함수.

    핵심 output:
    - common_step_summary
        어떤 step을 먼저 볼지 요약

    - top_common
        모든 common anomaly 후보 ranking

    - common_exact_value
        동일 step/value가 bad wafer들에 공통

    - common_within_lot_rare_pattern
        value는 달라도 같은 step/attr에서 lot 내부 rare 패턴이 공통

    - common_route
        공통 route anomaly

    - common_time
        공통 duration/wait outlier
    """
    d = prep(df, step_cols=step_cols, wafer_zfill=wafer_zfill)
    d = mark_bad(d, bad_wafers, wafer_zfill=wafer_zfill)

    total_bad = d.loc[d["is_bad"], "wafer_key"].nunique()
    eff_min_bad_support = max(1, min(min_bad_support, total_bad))

    reports = {
        "common_exact_value": common_exact_value_anomaly(
            d,
            cat_cols=cat_cols,
            min_bad_support=eff_min_bad_support,
            min_bad_coverage=min_bad_coverage,
        ),
        "common_within_lot_rare_pattern": common_within_lot_rare_pattern(
            d,
            cat_cols=cat_cols,
            min_bad_support=eff_min_bad_support,
            min_bad_coverage=min_bad_coverage,
        ),
        "common_route": common_route_anomaly(
            d,
            min_bad_support=eff_min_bad_support,
            min_bad_coverage=min_bad_coverage,
        ),
        "common_time": common_time_anomaly(
            d,
            min_bad_support=eff_min_bad_support,
            min_bad_coverage=min_bad_coverage,
        ),
    }

    reports["top_common"] = build_top_common(
        {k: v for k, v in reports.items()}
    )

    reports["common_step_summary"] = make_common_step_summary(
        reports["top_common"]
    )

    reports["prepared_data_with_bad_flag"] = d

    if save_csv:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for name, r in reports.items():
            if name == "prepared_data_with_bad_flag":
                continue

            if r is not None and not r.empty:
                r.to_csv(
                    Path(out_dir) / f"{name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    return reports