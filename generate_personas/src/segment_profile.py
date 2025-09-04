from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parents[1]

EXCEL_PATH = ROOT / "data" / "segments" / "food_customer_behavior_adult_2024.xlsx"
OUT_PATH   = ROOT / "data" / "segments" / "segments_profile.json"

# (숫자) 패턴 제거 및 앞뒤 공백제거
def clean_category(x):
    if isinstance(x, str):
        # 앞뒤 공백 제거 + (숫자) 패턴 제거
        return re.sub(r'^\s*\(\d+\)\s*', '', x).strip()
    return x

# 월평균가구소득
def categorize_income(x):
    if '100만원 미만' in x or '200' in x or '300' in x:
        return '저소득(월 300만원 미만)'
    elif '300' in x or '400' in x or '500' in x or '600' in x:
        return '중간소득(월 300~700만원 미만)'
    else:
        return '고소득(월 700만원 이상)'

# 가구원수 → 카테고리 변환
def map_household_size(x):
    if x == 1:
        return "1인가구"
    elif x == 2:
        return "2인가구"
    elif x>=3:
        return "3인가구 이상"
    else:
        return "정의되지 않음"

def generate_segment_df(df):
    cols = ['지역', 'SQ1', 'SQ2_range', 'JJ_SQ3_4', 'JJ_SQ3_5', 
        'J_SQ3N', 'BA5', 'J_SQ7', 'J_BA2', 'J_BA3', 'BA5_1', 'BA4', 'BA7']
    only_main_cols = ['J_A1', 'J_A1_1', 'J_D22_1', 'J_F1', 'J_F2_1']

    filtered_df = df[cols+only_main_cols].copy()

    for col in only_main_cols:
        filtered_df[col] = np.where(df['id'] == df['주구입자id'], df[col], np.nan)

    filtered_df = filtered_df[~filtered_df['BA5_1'].isin([6, 7])].copy()

    for col in filtered_df.columns:
        if filtered_df[col].dtype == 'object':
            filtered_df[col] = filtered_df[col].apply(clean_category)
    
    filtered_df["J_SQ3N"] = filtered_df["J_SQ3N"].astype(int).apply(map_household_size)
    
    filtered_df["J_SQ7"] = filtered_df["J_SQ7"].replace({
            r".*만들어 먹지 않는다.*": "거의 안함",
            r".*만들어 먹기도, 사 먹기도.*": "보통",
            r".*대부분 집에서 직접 만들어.*": "자주",
        }, regex=True).fillna("거의 안함")

    filtered_df['J_BA2']=filtered_df['J_BA2'].apply(categorize_income)

    col_names = ['지역', '성별', '연령대', '교육수준', '직업', '가구원수', '건강관심도','가구요리빈도','가구소득', '주거형태', '건강투자정도', '운동여부', 'sns사용빈도', '식료품구입빈도', '1회평균식료품구입금액', '우유구입기준', '가공식품구입빈도', '가공식품구입기준']
    filtered_df.columns = col_names

    return filtered_df

# ====== 핵심 함수 ======
def build_segments_profile(filtered_df: pd.DataFrame) -> dict:
    """
    입력 DataFrame에서 세그먼트 프로필 JSON(dict)을 생성합니다.
    - seg_cols: 세그먼트 키를 구성하는 컬럼들
    - num_vars: 수치 요약을 계산할 컬럼들
    반환: {"generated_at": ..., "segments": [...]} 형태의 dict
    """
    seg_cols = ["가구소득", "연령대", "가구원수", "성별"]
    num_vars = ["1회평균식료품구입금액"]

    # 안전 복사
    df = filtered_df.copy()

    # 세그먼트 키 생성
    df["segment_key"] = df[seg_cols].astype(str).agg("-".join, axis=1)

    # CAT_VARS: seg + num + 'segment_key' 이외의 모든 컬럼
    cat_vars = [c for c in df.columns if c not in (seg_cols + num_vars + ["segment_key"])]

    # (중요) 범주형 전체 레벨 수집 (일관된 출력용)
    global_levels = {}
    for v in cat_vars:
        overall_vc = df[v].astype(str).value_counts(dropna=True)
        # 문자열 "nan" 제거 (실제 NaN은 이미 drop됨)
        levels = [idx for idx in overall_vc.index if idx != "nan"]
        global_levels[v] = levels

    # (옵션) 전체 평균 (현재 요약문 비워둘 예정이므로 보관만)
    _overall_num_mean = df[num_vars].mean(numeric_only=True)

    out = {"generated_at": datetime.now().astimezone().isoformat(), "segments": []}

    # 세그먼트 루프
    for seg, sub in df.groupby("segment_key", dropna=False):
        seg_size = len(sub)
        if seg_size == 0:
            continue

        # filters dict
        parts = seg.split("-")
        filters = dict(zip(seg_cols, parts))

        # 수치형 요약 (NaN 무시, 표본 1개여도 std=0 되도록 ddof=0 사용)
        numerics = {}
        for v in num_vars:
            s = sub[v].dropna()
            numerics[v] = {
                "mean": round(float(s.mean()), 3) if len(s) else None,
                "std":  round(float(s.std(ddof=0)), 3) if len(s) else None,
                "min":  float(s.min()) if len(s) else None,
                "max":  float(s.max()) if len(s) else None,
            }

        # 범주형 분포: 원데이터 전 범주 포함(없는 건 ratio=0)
        categoricals = {}
        for v in cat_vars:
            vc = sub[v].astype(str).value_counts(normalize=True, dropna=True)
            vc = vc[vc.index != "nan"]  # 문자열 'nan' 제거
            filled = vc.reindex(global_levels[v], fill_value=0.0).sort_values(ascending=False)
            categoricals[v] = [
                {"value": str(idx), "ratio": round(float(r), 4)}
                for idx, r in filled.items()
            ]

        seg_obj = {
            "segment_key": seg,
            "filters": filters,
            "size": int(seg_size),
            "insights": {
                "인사이트1": "",
                "인사이트2": "",
                "인사이트3": "",
                "인사이트4": "",
                "인사이트5": "",
            },  # LLM이 나중에 채울 예정
            "numerics": numerics,
            "categoricals": categoricals,
        }
        out["segments"].append(seg_obj)

    return out


def save_segments_profile(out: dict, path: str | Path) -> None:
    """생성한 세그먼트 프로필 dict를 파일로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ====== main ======
def main():
    # 1. 원본 데이터 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name="String")

    # 2. 세그먼트용 전처리
    filtered_df = generate_segment_df(df)

    # 3. 세그먼트 요약 생성
    out = build_segments_profile(filtered_df)

    # 4. 파일 저장
    save_segments_profile(out, OUT_PATH)
    print(f"[OK] 저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
