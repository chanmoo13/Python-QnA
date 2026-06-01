import numpy as np
import pandas as pd
from scipy import stats


def _safe_pvalue_from_scipy_result(res):
    """
    scipy version에 따라 결과가 tuple 또는 object로 올 수 있어서 둘 다 대응.
    """
    if hasattr(res, "pvalue"):
        return float(res.pvalue)
    return float(res[1])


def run_edge_inner_center_tests(
    df: pd.DataFrame,
    iqr_k: float = 1.5,
    ref_zones=None,
    distances=("A", "B", "C", "D", "E"),
    key_cols=("process_id", "step_seq", "item_id"),
    value_col="value",
    zone_col="zone",
    distance_col="distance",
    item_group_col="item_group",
    tie_policy="first",
) -> pd.DataFrame:
    """
    Edge vs Inner 중심치 one-sided test 결과 생성.

    분석 단위:
        process_id / step_seq / item_id

    Reference 방법:
        1) ALL_INNER:
            zone E04~E25 전체
        2) MAX_INNER_ZONE:
            zone E04~E25 중 중심치가 가장 높은 zone 하나

    Compare:
        distance A, B, C, D, E 각각

    Test:
        FBT -> Mann-Whitney U test, alternative='greater'
        SIG -> Welch's t-test, alternative='greater'
        BIN -> Fisher's exact test, alternative='greater'

    Effect size:
        (Compare 중심치 - Reference 중심치) / Reference 중심치

        FBT -> median 기준
        SIG -> mean 기준
        BIN -> value == 1 비율 기준

    Outlier:
        FBT, SIG만 key별 IQR 제거
        BIN은 outlier 제거하지 않음

    Parameters
    ----------
    iqr_k:
        IQR multiplier.
        기본값 1.5.
        예: 더 강하게 제거하고 싶으면 1.0, 더 느슨하게 제거하고 싶으면 3.0.

    tie_policy:
        MAX_INNER_ZONE에서 중심치가 동일한 zone이 여러 개일 때 처리 방식.
        "first" -> E04~E25 순서상 첫 번째 zone만 사용
        "all"   -> 중심치가 max인 zone들을 모두 합쳐 사용
    """

    if ref_zones is None:
        ref_zones = [f"E{i:02d}" for i in range(4, 26)]

    if iqr_k < 0:
        raise ValueError("iqr_k는 0 이상이어야 합니다.")

    if tie_policy not in {"first", "all"}:
        raise ValueError("tie_policy는 'first' 또는 'all'만 사용할 수 있습니다.")

    required_cols = list(
        dict.fromkeys(
            list(key_cols)
            + [value_col, zone_col, distance_col, item_group_col]
        )
    )

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {missing_cols}")

    data = df[required_cols].copy()

    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])

    data[zone_col] = data[zone_col].astype("string").str.strip().str.upper()
    data[distance_col] = data[distance_col].astype("string").str.strip().str.upper()
    data[item_group_col] = data[item_group_col].astype("string").str.strip().str.upper()

    ref_zones = [str(z).strip().upper() for z in ref_zones]
    distances = [str(d).strip().upper() for d in distances]

    method_label = {
        "FBT": "Mann-Whitney U (greater)",
        "SIG": "Welch t-test (greater)",
        "BIN": "Fisher exact (greater)",
    }

    def validate_bin_values(g: pd.DataFrame, key_info: dict):
        vals = set(g[value_col].dropna().unique())
        if not vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"BIN item의 value는 0/1이어야 합니다. "
                f"key={key_info}, 발견된 값 예시={sorted(vals)[:10]}"
            )

    def center_value(s: pd.Series, item_group: str) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()

        if len(s) == 0:
            return np.nan

        if item_group == "FBT":
            return float(s.median())

        if item_group == "SIG":
            return float(s.mean())

        if item_group == "BIN":
            return float((s == 1).mean())

        return np.nan

    def effect_size(comp: pd.Series, ref: pd.Series, item_group: str) -> float:
        comp_center = center_value(comp, item_group)
        ref_center = center_value(ref, item_group)

        if pd.isna(comp_center) or pd.isna(ref_center):
            return np.nan

        if ref_center == 0:
            return np.nan

        return float((comp_center - ref_center) / ref_center)

    def calc_pvalue(comp: pd.Series, ref: pd.Series, item_group: str) -> float:
        comp = pd.to_numeric(comp, errors="coerce").dropna()
        ref = pd.to_numeric(ref, errors="coerce").dropna()

        if len(comp) == 0 or len(ref) == 0:
            return np.nan

        if item_group == "FBT":
            # H1: Edge > Inner
            return _safe_pvalue_from_scipy_result(
                stats.mannwhitneyu(
                    comp,
                    ref,
                    alternative="greater",
                )
            )

        if item_group == "SIG":
            # Welch's t-test는 양쪽 sample이 최소 2개 이상 필요
            if len(comp) < 2 or len(ref) < 2:
                return np.nan

            # H1: Edge mean > Inner mean
            return _safe_pvalue_from_scipy_result(
                stats.ttest_ind(
                    comp,
                    ref,
                    equal_var=False,
                    alternative="greater",
                    nan_policy="omit",
                )
            )

        if item_group == "BIN":
            edge_1 = int((comp == 1).sum())
            edge_0 = int((comp == 0).sum())

            inner_1 = int((ref == 1).sum())
            inner_0 = int((ref == 0).sum())

            # 출력하지 않고 p-value 계산에만 사용
            table = [
                [edge_1, edge_0],
                [inner_1, inner_0],
            ]

            # H1: Edge의 1 비율 > Inner의 1 비율
            return _safe_pvalue_from_scipy_result(
                stats.fisher_exact(
                    table,
                    alternative="greater",
                )
            )

        return np.nan

    def remove_iqr_outlier(g: pd.DataFrame, item_group: str) -> pd.DataFrame:
        # BIN은 outlier 제거하지 않음
        if item_group == "BIN":
            return g.copy()

        x = g[value_col].dropna()

        if len(x) == 0:
            return g.iloc[0:0].copy()

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr

        return g[
            (g[value_col] >= lower)
            & (g[value_col] <= upper)
        ].copy()

    def get_max_inner_ref(g: pd.DataFrame, item_group: str) -> pd.Series:
        inner = g[g[zone_col].isin(ref_zones)]

        if inner.empty:
            return inner[value_col]

        zone_center = {}

        for z in ref_zones:
            vals = inner.loc[inner[zone_col] == z, value_col]

            if len(vals) == 0:
                continue

            c = center_value(vals, item_group)

            if not pd.isna(c):
                zone_center[z] = c

        if not zone_center:
            return inner.iloc[0:0][value_col]

        max_center = max(zone_center.values())

        if tie_policy == "all":
            selected_zones = [
                z for z, c in zone_center.items()
                if np.isclose(c, max_center, equal_nan=False)
            ]
        else:
            selected_zones = [
                z for z in ref_zones
                if z in zone_center
                and np.isclose(zone_center[z], max_center, equal_nan=False)
            ][:1]

        return inner.loc[
            inner[zone_col].isin(selected_zones),
            value_col,
        ]

    result_rows = []

    for key_values, g0 in data.groupby(list(key_cols), dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)

        key_info = dict(zip(key_cols, key_values))

        item_groups = (
            g0[item_group_col]
            .dropna()
            .unique()
            .tolist()
        )

        item_groups = [
            x for x in item_groups
            if x in {"FBT", "SIG", "BIN"}
        ]

        if len(item_groups) != 1:
            raise ValueError(
                f"key={key_info}에서 item_group이 하나로 결정되지 않습니다. "
                f"발견된 item_group={g0[item_group_col].dropna().unique().tolist()}"
            )

        item_group = item_groups[0]

        if item_group == "BIN":
            validate_bin_values(g0, key_info)

        # key별 outlier 제거
        g = remove_iqr_outlier(g0, item_group)

        ref_sets = {
            "ALL_INNER": g.loc[
                g[zone_col].isin(ref_zones),
                value_col,
            ],
            "MAX_INNER_ZONE": get_max_inner_ref(g, item_group),
        }

        for ref_name, ref_values in ref_sets.items():
            row = dict(zip(key_cols, key_values))
            row["Reference방법"] = ref_name

            for d in distances:
                comp_values = g.loc[
                    g[distance_col] == d,
                    value_col,
                ]

                row[f"거리{d}"] = calc_pvalue(
                    comp=comp_values,
                    ref=ref_values,
                    item_group=item_group,
                )

            for d in distances:
                comp_values = g.loc[
                    g[distance_col] == d,
                    value_col,
                ]

                row[f"거리{d}_effectsize"] = effect_size(
                    comp=comp_values,
                    ref=ref_values,
                    item_group=item_group,
                )

            row["Method"] = method_label[item_group]

            result_rows.append(row)

    result = pd.DataFrame(result_rows)

    ordered_cols = (
        list(key_cols)
        + ["Reference방법"]
        + [f"거리{d}" for d in distances]
        + [f"거리{d}_effectsize" for d in distances]
        + ["Method"]
    )

    return result[ordered_cols]