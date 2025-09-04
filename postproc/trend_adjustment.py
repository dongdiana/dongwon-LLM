import numpy as np
import pandas as pd
from pathlib import Path


def trend_adjustment(trend, mean, length=12):
    '''
    trend: 검색량 트렌드
    mean: 예측 평균값
    length: 시계열 길이
    '''
    coefficient = length/np.sum(trend)
    adjusted_trend = trend * coefficient
    series = adjusted_trend * mean
    return series


def trend_adjust(df: pd.DataFrame):
    trend_map = {
        "참치캔": np.array([75,100,89,72,72,54,93,63,35,29,43,39]),
        "참치액": np.array([58.51063,86.80851,92.55319,70,84.04255,57.02127,73.82978,
                           71.91489,67.87234,98.29787,100,61.27659]),
        "스팸": np.array([100,100,100]),
        "편의점커피라떼": np.array([3,2,1.5,1,1]),
        "그릭요거트": np.array([69,96,80,89,61])
    }

    for idx, row in df.iterrows():
        key = row['keyword']
        trend = trend_map.get(key)
        if trend is None:
            print(f"⚠️ Warning: '{key}'은 trend_map에 없음. 건너뜀.")
            continue

        mean = df.iloc[idx, -1]

        # 트렌드 적용
        series = trend_adjustment(trend, mean, length=len(trend))
        df.iloc[idx, -len(trend):] = series.astype(int)

    return df


def save_submission(df, path="./postproc/outputs/submission_final_ver1.csv"):
    out = df.copy()

    # 1) 'keyword' 열 제거 (없으면 무시)
    out = out.drop(columns=["keyword"], errors="ignore")

    # 2) product_name 컬럼을 맨 앞으로 (있을 때만)
    if "product_name" in out.columns:
        cols = ["product_name"] + [c for c in out.columns if c != "product_name"]
        out = out[cols]

    # 3) 저장 폴더 자동 생성 + 저장
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved submission to {out_path}")
    return out


if __name__ == "__main__":
    # 1) baseline.csv 읽기
    df = pd.read_csv("./postproc/outputs/baseline.csv", encoding="utf-8-sig")

    # 2) 트렌드 적용
    df_adj = trend_adjust(df)

    # 3) 저장
    save_submission(df_adj, "./postproc/outputs/submission_final.csv")
