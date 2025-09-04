import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def compute_posterior(
    sample1: pd.DataFrame,
    sample2: pd.DataFrame,
    sim_json_path: Path,
) -> Tuple[Dict, Dict]:
    """
    sample1: 2% 인구사항 전처리 데이터 (ex: ./data/population/2020_2%_인구사항_전처리.csv)
    sample2: 표본추출대상집단 데이터 (ex: ./data/population/표본추출대상집단.csv)
    sim_json_path: 시뮬레이션 결과 json 파일 경로 (ex: ./results/시장조사_참치액.json)

    Returns:
        posterior_dict, prior_dict (둘 다 {attr -> {class -> prob}})
    """
    with sim_json_path.open("r", encoding="utf-8") as f:
        sim_result = json.load(f)

    # 시뮬 결과 평탄화 & 숫자 변환
    result_df = pd.json_normalize(sim_result["results"])
    result_df["purchase_decision"] = pd.to_numeric(
        result_df["purchase_decision"], errors="coerce"
    )

    # 시뮬 결과에서 참조할 속성 경로들
    cols = [
        "persona.gender",
        "persona.age",
        "persona.household_size",
        "persona.income",
    ]

    # 사전확률 (prior) 계산
    prior_gender = (sample1["성별"].value_counts(normalize=True)).sort_index()
    prior_age = (sample1["연령대"].value_counts(normalize=True)).sort_index()
    prior_hh = (sample1["가구원수"].value_counts(normalize=True)).sort_index()
    prior_income = (sample2["소득구간"].value_counts(normalize=True)).sort_index()

    priors = [prior_gender, prior_age, prior_hh, prior_income]

    posterior_dict: Dict[str, Dict[str, float]] = {}
    prior_dict: Dict[str, Dict[str, float]] = {}

    for attr, prior in zip(cols, priors):
        # 우도(각 클래스의 구매 확률)
        likelihood = result_df.groupby(attr)["purchase_decision"].mean()

        # 인덱스 정렬/정합 (동일한 class 축을 맞춘다)
        likelihood = likelihood.sort_index()
        likelihood = likelihood.reindex(prior.index)  # prior의 클래스 순서를 기준으로 맞춤

        # 결측/부적합 방어
        # (시뮬 결과에 없는 클래스가 있을 수 있으므로 0으로 채움)
        likelihood = likelihood.fillna(0.0)

        purchase_prob = float((likelihood.values * prior.values).sum())
        if purchase_prob == 0:
            # 모든 확률이 0이면 균등 분포로 fallback
            uniform = pd.Series([1.0 / len(prior)] * len(prior), index=prior.index)
            posterior = uniform
        else:
            posterior = (likelihood * prior) / purchase_prob

        posterior_dict[attr] = {cls: float(val) for cls, val in posterior.items()}
        prior_dict[attr] = {cls: float(val) for cls, val in prior.items()}

    return posterior_dict, prior_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute posteriors from simulation results with priors."
    )
    p.add_argument(
        "--keyword",
        required=True,
        help="시장조사 키워드 (ex: 참치액, 참치캔, 그릭요거트, 편의점커피라떼 등)",
    )
    p.add_argument(
        "--pop-path",
        default="./data/population/2020_2%_인구사항_전처리.csv",
        help="2% 인구사항 전처리 CSV 경로",
    )
    p.add_argument(
        "--sample-path",
        default="./data/population/표본추출대상집단.csv",
        help="표본추출대상집단 CSV 경로",
    )
    p.add_argument(
        "--results-dir",
        default="./results",
        help="시장조사 결과(JSON) 폴더 경로",
    )
    p.add_argument(
        "--out-dir",
        default="./generate_persona/weights",
        help="사후확률 출력 폴더 경로",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV 읽기 인코딩 (기본 utf-8, 필요시 cp949 등으로 변경)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pop_path = Path(args.pop_path)
    sample_path = Path(args.sample_path)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    sim_path = results_dir / f"A_{args.keyword}_시뮬레이션.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    sample1 = pd.read_csv(pop_path, encoding=args.encoding)
    sample2 = pd.read_csv(sample_path, encoding=args.encoding)

    posteriors, priors = compute_posterior(sample1, sample2, sim_path)

    # 저장
    out_post = out_dir / f"{args.keyword}_posterior.json"
    with out_post.open("w", encoding="utf-8") as f:
        json.dump(posteriors, f, ensure_ascii=False, indent=2)

    print(f"[OK] Posterior saved: {out_post}")

if __name__ == "__main__":
    main()
