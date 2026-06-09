import pandas as pd


NEW_COL = "count_decoded"


def normalize_index_value(x):
    """
    df2['INDEX'] 값을 문자열 key로 정규화합니다.
    예: 1 -> "1", 1.0 -> "1", " A " -> "A"
    """
    if pd.isna(x):
        raise ValueError("df2['INDEX']에 NaN 값이 있습니다.")

    if isinstance(x, float) and x.is_integer():
        return str(int(x))

    return str(x).strip()


# --------------------------------------------------
# 1. df2 mapping 생성
# --------------------------------------------------
df2_map = df2.copy()
df2_map["INDEX"] = df2_map["INDEX"].map(normalize_index_value)

# INDEX 중복 체크
duplicated_index = df2_map.loc[df2_map["INDEX"].duplicated(keep=False), "INDEX"].unique()
if len(duplicated_index) > 0:
    raise ValueError(f"df2['INDEX']에 중복값이 있습니다: {list(duplicated_index)}")

# INDEX는 한 글자여야 함
invalid_index = df2_map.loc[df2_map["INDEX"].str.len() != 1, "INDEX"].unique()
if len(invalid_index) > 0:
    raise ValueError(f"df2['INDEX']에는 한 글자 값만 있어야 합니다: {list(invalid_index)}")

count_map = df2_map.set_index("INDEX")["count_real"].astype(int).to_dict()
count_map_keys = set(count_map.keys())


# --------------------------------------------------
# 2. SOLTIME_RB_COUNT 파싱
# --------------------------------------------------
def parse_rb_count(value, row_idx):
    """
    예:
    R0_ABCDEFGHIJKLMNOPQRSTUV/R1_1234567890ABCDEFGHIJKL

    return:
    ["ABCDEFGHIJKLMNOPQRSTUV", "1234567890ABCDEFGHIJKL"]
    """
    if pd.isna(value):
        raise ValueError(f"row index {row_idx}: SOLTIME_RB_COUNT 값이 NaN입니다.")

    value = str(value).strip()
    parts = value.split("/")

    if len(parts) != 2:
        raise ValueError(
            f"row index {row_idx}: SOLTIME_RB_COUNT 형식이 잘못되었습니다. "
            f"기대 형식은 'R0_문자열/R1_문자열' 입니다. 실제값: {value}"
        )

    rb_dict = {}

    for part in parts:
        if "_" not in part:
            raise ValueError(
                f"row index {row_idx}: '_' 구분자가 없습니다. 실제값: {part}"
            )

        prefix, encoded = part.split("_", 1)

        if prefix not in {"R0", "R1"}:
            raise ValueError(
                f"row index {row_idx}: prefix는 R0 또는 R1이어야 합니다. 실제값: {prefix}"
            )

        if len(encoded) != 22:
            raise ValueError(
                f"row index {row_idx}: {prefix} 문자열 길이는 22여야 합니다. "
                f"실제 길이: {len(encoded)}, 실제 문자열: {encoded}"
            )

        rb_dict[prefix] = encoded

    if set(rb_dict.keys()) != {"R0", "R1"}:
        raise ValueError(
            f"row index {row_idx}: R0와 R1이 모두 있어야 합니다. 실제 prefix: {list(rb_dict.keys())}"
        )

    return [rb_dict["R0"], rb_dict["R1"]]


# --------------------------------------------------
# 3. 제품별 문자열 추출 규칙
# --------------------------------------------------
def extract_chars(encoded, product_type):
    """
    제품A:
    앞 5개 추출      -> encoded[0:5]
    다음 5개 skip
    다음 1개 추출    -> encoded[10]
    다음 5개 추출    -> encoded[11:16]
    다음 5개 skip
    마지막 1개 추출  -> encoded[21]

    총 12개 문자 추출
    """
    if product_type == "A":
        return (
            encoded[0:5]
            + encoded[10]
            + encoded[11:16]
            + encoded[21]
        )

    if product_type == "B":
        return encoded

    raise ValueError(f"알 수 없는 product_type입니다: {product_type}")


# --------------------------------------------------
# 4. row 단위 decode
# --------------------------------------------------
def decode_count_row(row):
    part_no = row["part_no"]

    if pd.isna(part_no):
        return "part_no_error"

    part_no = str(part_no)

    if part_no.startswith("aaaaa"):
        product_type = "A"
    elif part_no.startswith("bbbbb"):
        product_type = "B"
    else:
        return "part_no_error"

    rb_strings = parse_rb_count(
        value=row["SOLTIME_RB_COUNT"],
        row_idx=row.name
    )

    extracted = "".join(
        extract_chars(encoded=s, product_type=product_type)
        for s in rb_strings
    )

    missing_chars = sorted(set(extracted) - count_map_keys)
    if missing_chars:
        raise KeyError(
            f"row index {row.name}: df2['INDEX']에 없는 문자가 있습니다. "
            f"missing_chars={missing_chars}, extracted={extracted}"
        )

    return sum(count_map[ch] for ch in extracted)


# --------------------------------------------------
# 5. 새 컬럼 생성
# --------------------------------------------------
df1[NEW_COL] = df1.apply(decode_count_row, axis=1)