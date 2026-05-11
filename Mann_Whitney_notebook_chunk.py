# =========================
# 1. 설정
# =========================

input_path = "rawdata_sorted.csv"
output_path = "mannwhitney_result.csv"

chunksize = 300_000

cols = ["step_seq", "item_id", "value", "거리 구분", "Group"]

result_cols = [
    "step_seq",
    "item_id",
    "Group",
    "ref_group",
    "comp_group",
    "ref_median",
    "comp_median",
    "p_value"
]

# 기존 결과 파일 삭제
if os.path.exists(output_path):
    os.remove(output_path)
    

# =========================
# 2. 최소 helper - 수정 버전
# =========================

def remove_outlier_iqr(x):
    """
    Rule of thumb outlier 제거:
    Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
    
    Mann-Whitney 오류 방지를 위해 무조건 numeric으로 변환.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    
    if len(x) == 0:
        return x
    
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    return x[(x >= lower) & (x <= upper)]


def make_result_rows(test_df):
    """
    완성된 step_seq + item_id 데이터에 대해서
    Group별로 E4 vs 나머지 거리 구분 Mann-Whitney 검정
    """
    rows = []
    
    # value를 다시 한번 numeric으로 강제 변환
    test_df = test_df.copy()
    test_df["value"] = pd.to_numeric(test_df["value"], errors="coerce")
    test_df = test_df.dropna(subset=["value"])
    
    for (step_seq, item_id), item_df in test_df.groupby(
        ["step_seq", "item_id"],
        sort=False
    ):
        for group_name, group_df in item_df.groupby("Group", sort=False):
            
            ref_raw = group_df.loc[
                group_df["거리 구분"] == "E4",
                "value"
            ]
            
            ref_raw = pd.to_numeric(ref_raw, errors="coerce").dropna()
            
            if len(ref_raw) == 0:
                continue
            
            comp_groups = [
                x for x in group_df["거리 구분"].dropna().unique()
                if x != "E4"
            ]
            
            for comp_group in sorted(comp_groups):
                
                comp_raw = group_df.loc[
                    group_df["거리 구분"] == comp_group,
                    "value"
                ]
                
                comp_raw = pd.to_numeric(comp_raw, errors="coerce").dropna()
                
                if len(comp_raw) == 0:
                    continue
                
                # median은 outlier 제거 전 기준
                ref_median = ref_raw.median()
                comp_median = comp_raw.median()
                
                # p-value는 outlier 제거 후 기준
                ref_clean = remove_outlier_iqr(ref_raw)
                comp_clean = remove_outlier_iqr(comp_raw)
                
                # scipy에 넣기 직전 numpy float array로 강제 변환
                ref_arr = pd.to_numeric(ref_clean, errors="coerce").dropna().to_numpy(dtype="float64")
                comp_arr = pd.to_numeric(comp_clean, errors="coerce").dropna().to_numpy(dtype="float64")
                
                if len(ref_arr) > 0 and len(comp_arr) > 0:
                    p_value = mannwhitneyu(
                        ref_arr,
                        comp_arr,
                        alternative="two-sided"
                    ).pvalue
                else:
                    p_value = None
                
                rows.append({
                    "step_seq": step_seq,
                    "item_id": item_id,
                    "Group": group_name,
                    "ref_group": "E4",
                    "comp_group": comp_group,
                    "ref_median": ref_median,
                    "comp_median": comp_median,
                    "p_value": p_value
                })
    
    return rows
    
# =========================
# 3. Chunk 단위 순차 load + 짜투리 buffer 처리
# =========================

buffer = pd.DataFrame(columns=cols)

header_written = False
total_tests = 0

reader = pd.read_csv(
    input_path,
    usecols=cols,
    chunksize=chunksize,
    encoding="utf-8-sig"
)

for chunk_no, chunk in enumerate(reader, start=1):
    
    # -------------------------
    # 필요한 column만 유지
    # -------------------------
    chunk = chunk[cols].copy()
    
    # -------------------------
    # Basic cleaning
    # -------------------------
    chunk["step_seq"] = chunk["step_seq"].astype(str).str.strip()
    chunk["item_id"] = chunk["item_id"].astype(str).str.strip()
    
    chunk["Group"] = chunk["Group"].fillna("").astype(str).str.strip()
    chunk.loc[chunk["Group"] == "", "Group"] = "EE-type"
    
    chunk["거리 구분"] = (
        chunk["거리 구분"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    
    chunk["value"] = (
        chunk["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    
    chunk = chunk.dropna(subset=["value"])
    chunk = chunk[chunk["거리 구분"] != ""].copy()
    
    # -------------------------
    # 이전 chunk의 짜투리와 현재 chunk 합치기
    # -------------------------
    data = pd.concat([buffer, chunk], ignore_index=True)
    
    if len(data) == 0:
        continue
    
    # -------------------------
    # 현재 data의 마지막 step_seq + item_id는
    # 다음 chunk에 이어질 가능성이 있으므로 buffer에 남김
    # -------------------------
    last_step_seq = data["step_seq"].iloc[-1]
    last_item_id = data["item_id"].iloc[-1]
    
    last_key_mask = (
        (data["step_seq"] == last_step_seq) &
        (data["item_id"] == last_item_id)
    )
    
    test_df = data.loc[~last_key_mask].copy()
    buffer = data.loc[last_key_mask].copy()
    
    # -------------------------
    # 완성된 step_seq + item_id만 검정
    # -------------------------
    result_rows = make_result_rows(test_df)
    
    # -------------------------
    # 결과는 바로 파일에 append 저장
    # -------------------------
    if len(result_rows) > 0:
        result_chunk = pd.DataFrame(result_rows, columns=result_cols)
        
        result_chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=not header_written,
            encoding="utf-8-sig"
        )
        
        header_written = True
        total_tests += len(result_chunk)
    
    print(f"chunk {chunk_no} 완료 / 현재까지 검정 수: {total_tests:,} / buffer row 수: {len(buffer):,}")
    
