import pandas as pd
import re
from numbers import Real


# 새로 만들 컬럼명
LABEL_COL = "A"
POSITION_COL = "position"
DECODED_COL = "count_decoded"


# --------------------------------------------------
# 0. 새 컬럼명 중복 방지
# --------------------------------------------------
new_cols = [LABEL_COL, POSITION_COL, DECODED_COL]

duplicated_new_cols = [col for col in new_cols if col in df1.columns]
if duplicated_new_cols:
    raise ValueError(f"df1에 이미 새로 만들 컬럼명이 존재합니다: {duplicated_new_cols}")


# --------------------------------------------------
# 1. df2 mapping 생성
# --------------------------------------------------
def normalize_index_value(x):
    """
    df2['INDEX'] 값을 문자 1개짜리 key로 정규화합니다.
    예:
    1     -> "1"
    1.0   -> "1"
    " A " -> "A"
    """
    if pd.isna(x):
        raise ValueError("df2['INDEX']에 NaN 값이 있습니다.")

    if isinstance(x, Real) and float(x).is_integer():
        return str(int(x))

    return str(x).strip()


df2_map = df2.copy()

df2_map["INDEX"] = df2_map["INDEX"].map(normalize_index_value)

duplicated_index = df2_map.loc[
    df2_map["INDEX"].duplicated(keep=False),
    "INDEX"
].unique()

if len(duplicated_index) > 0:
    raise ValueError(f"df2['INDEX']에 중복값이 있습니다: {list(duplicated_index)}")

invalid_index = df2_map.loc[
    df2_map["INDEX"].str.len() != 1,
    "INDEX"
].unique()

if len(invalid_index) > 0:
    raise ValueError(
        f"df2['INDEX']에는 문자 1개짜리 값만 있어야 합니다: {list(invalid_index)}"
    )

if df2_map["count_real"].isna().any():
    raise ValueError("df2['count_real']에 NaN 값이 있습니다.")

df2_map["count_real"] = pd.to_numeric(df2_map["count_real"], errors="raise").astype(int)

count_map = df2_map.set_index("INDEX")["count_real"].to_dict()
count_map_keys = set(count_map.keys())


# --------------------------------------------------
# 2. part_no 기준 제품 구분
# --------------------------------------------------
def get_product_type(part_no):
    """
    return:
    - "A"    : 제품A
    - "B"    : 제품B
    - None   : part_no_error 대상
    """
    if pd.isna(part_no):
        return None

    part_no = str(part_no)

    if part_no.startswith("aaaaa"):
        return "A"

    if part_no.startswith("bbbbb"):
        return "B"

    return None


# --------------------------------------------------
# 3. SOLTIME_RB_COUNT 파싱
# --------------------------------------------------
def parse_rb_count(value, row_idx):
    """
    예:
    R0_ABCDEFGHIJKLMNOPQRSTUV/R1_1234567890ABCDEFGHIJKL

    return:
    [
        R0 문자열,
        R1 문자열
    ]
    """
    if pd.isna(value):
        raise ValueError(f"row index {row_idx}: SOLTIME_RB_COUNT 값이 NaN입니다.")

    value = str(value).strip()
    parts = [x.strip() for x in value.split("/")]

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
        prefix = prefix.strip()
        encoded = encoded.strip()

        if prefix not in {"R0", "R1"}:
            raise ValueError(
                f"row index {row_idx}: prefix는 R0 또는 R1이어야 합니다. 실제값: {prefix}"
            )

        if len(encoded) != 22:
            raise ValueError(
                f"row index {row_idx}: {prefix} 문자열 길이는 22여야 합니다. "
                f"실제 길이: {len(encoded)}, 실제 문자열: {encoded}"
            )

        if re.fullmatch(r"[0-9A-Z]{22}", encoded) is None:
            raise ValueError(
                f"row index {row_idx}: {prefix} 문자열은 숫자 또는 대문자 알파벳 22개여야 합니다. "
                f"실제 문자열: {encoded}"
            )

        rb_dict[prefix] = encoded

    if set(rb_dict.keys()) != {"R0", "R1"}:
        raise ValueError(
            f"row index {row_idx}: R0와 R1이 모두 있어야 합니다. "
            f"실제 prefix: {list(rb_dict.keys())}"
        )

    return [rb_dict["R0"], rb_dict["R1"]]


# --------------------------------------------------
# 4. 제품별 추출 위치와 A 컬럼 label 생성
# --------------------------------------------------
def build_extract_plan(product_type, rb_no):
    """
    rb_no:
    - 0: R0
    - 1: R1

    return:
    [
        ("comp1", 0),
        ("comp2", 1),
        ...
    ]

    tuple의 두 번째 값은 encoded 문자열에서 뽑을 위치입니다.
    """

    plan = []

    if product_type == "A":
        # 제품A는 R0에서 comp1~10, extra1~2
        # 제품A는 R1에서 comp11~20, extra3~4
        comp_start = 1 + rb_no * 10
        extra_start = 1 + rb_no * 2

        # 앞 5개 추출
        for i in range(5):
            plan.append((f"comp{comp_start + i}", i))

        # 다음 5개 skip 후 1개 추출
        plan.append((f"extra{extra_start}", 10))

        # 다시 5개 추출
        for i in range(5):
            plan.append((f"comp{comp_start + 5 + i}", 11 + i))

        # 다음 5개 skip 후 마지막 1개 추출
        plan.append((f"extra{extra_start + 1}", 21))

        return plan

    if product_type == "B":
        # 제품B는 R0에서 comp1~20, extra1~2
        # 제품B는 R1에서 comp21~40, extra3~4
        comp_start = 1 + rb_no * 20
        extra_start = 1 + rb_no * 2

        # 앞 10개 추출
        for i in range(10):
            plan.append((f"comp{comp_start + i}", i))

        # 1개 extra
        plan.append((f"extra{extra_start}", 10))

        # 다음 10개 추출
        for i in range(10):
            plan.append((f"comp{comp_start + 10 + i}", 11 + i))

        # 마지막 1개 extra
        plan.append((f"extra{extra_start + 1}", 21))

        return plan

    raise ValueError(f"알 수 없는 product_type입니다: {product_type}")


# --------------------------------------------------
# 5. df1 row를 여러 row로 펼치기
# --------------------------------------------------
def expand_one_row(row):
    row_idx = row.name
    base = row.to_dict()

    product_type = get_product_type(row["part_no"])

    # 기존 요구사항 유지:
    # part_no가 aaaaa/bbbbb로 시작하지 않으면 에러를 발생시키지 않고
    # 새 컬럼들에 part_no_error를 넣은 row 1개 생성
    if product_type is None:
        return [
            {
                **base,
                LABEL_COL: "part_no_error",
                POSITION_COL: "part_no_error",
                DECODED_COL: "part_no_error",
            }
        ]

    rb_strings = parse_rb_count(
        value=row["SOLTIME_RB_COUNT"],
        row_idx=row_idx
    )

    output_rows = []

    for rb_no, encoded in enumerate(rb_strings):
        extract_plan = build_extract_plan(product_type, rb_no)

        for label, char_idx in extract_plan:
            char_value = encoded[char_idx]

            if char_value not in count_map_keys:
                raise KeyError(
                    f"row index {row_idx}: df2['INDEX']에 없는 문자가 있습니다. "
                    f"문자={char_value}, A={label}, SOLTIME_RB_COUNT={row['SOLTIME_RB_COUNT']}"
                )

            output_rows.append(
                {
                    **base,
                    LABEL_COL: label,
                    POSITION_COL: char_value,
                    DECODED_COL: count_map[char_value],
                }
            )

    return output_rows


# --------------------------------------------------
# 6. 최종 expanded DataFrame 생성
# --------------------------------------------------
records = []

for _, row in df1.iterrows():
    records.extend(expand_one_row(row))

df1_expanded = pd.DataFrame(records)

# 컬럼 순서 정리: 기존 df1 컬럼 + 새 컬럼 3개
df1_expanded = df1_expanded[list(df1.columns) + new_cols]