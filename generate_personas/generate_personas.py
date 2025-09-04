# generate_personas.py
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from itertools import product
import random, string

from dotenv import load_dotenv
from openai import OpenAI

# === 프롬프트 모듈 ===
from prompts import prompt_mc, prompt_sw

# -----------------------------
# 로거 설정
# -----------------------------
logger = logging.getLogger("personas")

def setup_logger(level: str = "INFO", log_file: str | None = None):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(numeric)
    logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(numeric)
        logger.addHandler(fh)

# -----------------------------
# 환경설정 / 클라이언트 초기화
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 환경변수를 확인하세요.")
client = OpenAI(api_key=API_KEY)

# -----------------------------
# 유틸
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_text(resp: Any) -> str:
    """
    OpenAI Responses 응답에서 텍스트를 최대한 안전하게 추출
    """
    try:
        choice0 = getattr(resp, "choices", [None])[0]
        if choice0 and getattr(choice0, "message", None):
            content = choice0.message.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "").strip()
            elif isinstance(content, str):
                return content.strip()
    except Exception:
        pass
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        pass
    return str(resp)

def safe_json_loads(text: str):
    """
    LLM 응답 텍스트에서 JSON만 추출해서 로드
    - ```json ... ``` 코드블럭 제거
    - 실패 시 None
    """
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def preview(text: str, n: int = 400) -> str:
    text = text.replace("\n", "\\n")
    return text[:n] + ("..." if len(text) > n else "")

# -----------------------------
# posterior → 세그먼트 생성
# -----------------------------
def make_posterior_df(keyword: str, weights_dir: Path) -> pd.DataFrame:
    """
    1차 시뮬로 얻은 사후확률(주요 단일 속성들)을 독립 가정으로 곱하여 조인트 seg 확률 생성
    """
    posterior_path = weights_dir / f"{keyword}_posterior.json"
    if not posterior_path.exists():
        raise FileNotFoundError(f"posterior 파일이 없습니다: {posterior_path}")
    logger.info(f"[posterior] load → {posterior_path}")

    with posterior_path.open("r", encoding="utf-8") as f:
        posterior = json.load(f)

    def req(d, *keys):
        cur = d
        for k in keys:
            if k not in cur:
                raise KeyError(f"posterior에 키가 없습니다: {'/'.join(keys)}")
            cur = cur[k]
        return cur

    gender_map = {
        "남성": req(posterior, "persona.gender", "남자"),
        "여성": req(posterior, "persona.gender", "여자"),
    }
    income_map = {
        "고소득(월 700만원 이상)": req(posterior, "persona.income", "고소득(월 700만원 이상)"),
        "중간소득(월 300~700만원 미만)": req(posterior, "persona.income", "중간소득(월 300~700만원 미만)"),
        "저소득(월 300만원 미만)": req(posterior, "persona.income", "저소득(월 300만원 미만)"),
    }
    age_map = {
        "만 19~29세": req(posterior, "persona.age", "20대"),
        "만 30~39세": req(posterior, "persona.age", "30대"),
        "만 40~49세": req(posterior, "persona.age", "40대"),
        "만 50~59세": req(posterior, "persona.age", "50대"),
        "만 60~69세": req(posterior, "persona.age", "60대"),
        "만 70세 이상": req(posterior, "persona.age", "70대 이상"),
    }
    hh_map = {
        "1인가구": req(posterior, "persona.household_size", "1인가구"),
        "2인가구": req(posterior, "persona.household_size", "2인가구"),
        "3인가구 이상": req(posterior, "persona.household_size", "3인가구 이상"),
    }

    combinations = list(product(gender_map.keys(), income_map.keys(), age_map.keys(), hh_map.keys()))
    df = pd.DataFrame(combinations, columns=["gender", "income", "age_band", "hh_size"])

    df["ratio"] = (
        df["gender"].map(gender_map)
        * df["income"].map(income_map)
        * df["age_band"].map(age_map)
        * df["hh_size"].map(hh_map)
    ).astype(float)

    df["ratio"] = df["ratio"].clip(lower=0)
    s = df["ratio"].sum()
    if s <= 0:
        raise ValueError("posterior에서 유효한 확률을 만들 수 없습니다 (합이 0).")
    df["ratio"] = df["ratio"] / s

    df["segment_key"] = df[['income', 'age_band', 'hh_size', 'gender']].astype(str).agg("-".join, axis=1)
    logger.info(f"[posterior] combos={len(df)}, ratio_sum={df['ratio'].sum():.6f}")
    return df

def sample_segments(df: pd.DataFrame, n_samples=100, seed=2025) -> pd.DataFrame:
    np.random.seed(2025)
    idx = np.random.choice(df.index, size=n_samples, replace=True, p=df["ratio"].values)
    out = df.loc[idx].reset_index(drop=True)
    logger.info(f"[sample] n_samples={n_samples}, unique_segments={out['segment_key'].nunique()}")
    return out

def matching_similar_products(keyword: str):
    if keyword == '그릭요거트':
        return {'풀무원 그릭요거트 400g': 44.7, '그릭데이 시그니처 450g': 14.4, '매일바이오 그릭요거트 400g': 40.9}
    elif keyword == '스팸':
        return {'스팸 25% 라이트': 75.0, '리챔 더블라이트': 25.0}
    elif keyword == '편의점커피라떼':
        return {'이디야커피 바닐라라떼 300ml': 3.3, '매일유업 바리스타룰스 바닐라빈라떼 325ml': 96.7}
    else:
        return {}

def assign_brand_random(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    brand_probs = matching_similar_products(keyword)
    if not brand_probs:
        df = df.copy()
        df["brand"] = None
        logger.info("[brand] no brand distribution for keyword; brand=None")
        return df
    brands = list(brand_probs.keys())
    probs = np.array(list(brand_probs.values()), dtype=float)
    probs = probs / probs.sum()
    df = df.copy()
    df["brand"] = np.random.choice(brands, size=len(df), p=probs)
    logger.info(f"[brand] assign from {len(brands)} brands for keyword={keyword}")
    return df

# -----------------------------
# 세그먼트 → 프롬프트 엔트리
# -----------------------------
def _create_segment_data_entry(row: Dict, keyword: str, mode: str) -> Dict:
    meta_profile = {
        "성별": row.get("gender", ""),
        "연령대": row.get("age_band", ""),
        "가구소득": row.get("income", ""),
        "가구원수": row.get("hh_size", "")
    }
    if mode == "swap":
        meta_profile["기존사용제품"] = row.get("brand", "")

    # uuid는 여전히 LLM 템플릿에 미리 포함(현재 설계 유지)
    uuid_str = ''.join(random.choices(string.digits, k=6))

    tpl = {
        "uuid": uuid_str,
        "segment_key_input": row.get('segment_key', ""),
        "reasoning": "",
        "가구소득": row.get("income", ""),
        "연령대": row.get("age_band", ""),
        "가구원수": row.get("hh_size", ""),
        "성별": row.get("gender", "")
    }
    if mode == "swap":
        tpl["기존사용제품"] = row.get("brand", "")

    sub_cols = ['지역', '교육수준', '직업', '건강관심도', '가구요리빈도', '주거형태', '건강투자정도', '운동여부', 'sns사용빈도', '식료품구입빈도', '1회평균식료품구입금액']
    for col in sub_cols:
        tpl[col] = ""

    if keyword in ("그릭요거트", "편의점커피라떼"):
        tpl["우유구입기준"] = ""
    if keyword in ("스팸", "참치캔"):
        tpl["가공식품구입빈도"] = ""
        tpl["가공식품구입기준"] = ""

    return {"meta_profile": meta_profile, "empty_persona_template": tpl}

# -----------------------------
# 프롬프트 생성
# -----------------------------
def load_context_and_product_info(context_dir: Path, keyword: str):
    fp = context_dir / f"{keyword}.json"
    if not fp.exists():
        raise FileNotFoundError(f"컨텍스트 파일이 없습니다: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        jd = json.load(f)
    context = jd.get("market_report", {}).get("content", [])[:5]
    sim = jd.get("유사제품군", {})
    product_info = {name: {
        "product_info": node.get("product_info", []),
        "release": node.get("release"),
        "price": node.get("price"),
    } for name, node in sim.items()}
    if not context:
        raise ValueError(f"'{fp}'에서 market_report.content를 찾지 못했습니다.")
    logger.info(f"[context] {fp.name} loaded (context_items={len(context)}, similar_products={len(product_info)})")
    return context, product_info

def make_batch_persona_prompt(segment_rows: List[Dict], keyword: str, mode: str,
                              context_dir: Path) -> Dict[str, str]:
    """
    기존 반환(system/user)에 더해, 이 배치에 사용된 entries/uuids를 함께 반환(검증·복구용).
    기존 호출부는 system/user만 사용하므로 호환성 유지.
    """
    if not segment_rows:
        raise ValueError("segment_rows는 비어있을 수 없습니다.")
    mode = mode.lower()
    prompt_mod = prompt_mc if mode in ("mc", "multiple_choice") else prompt_sw
    context, product_info = load_context_and_product_info(context_dir, keyword)

    # entries 리스트 생성
    entries: List[Dict[str, Any]] = [
        _create_segment_data_entry(r, keyword, "swap" if mode.startswith("sw") else "mc")
        for r in segment_rows
    ]
    expected_uuids = [
        e.get("empty_persona_template", {}).get("uuid") for e in entries
        if isinstance(e, dict)
    ]
    uuid_to_entry = {
        e["empty_persona_template"]["uuid"]: e
        for e in entries
        if isinstance(e, dict) and "empty_persona_template" in e
    }

    user_prompt = prompt_mod.PERSONA_GEN_PROMPT_BASE.format(
        keyword=keyword,
        num_segments=len(segment_rows),
        context=json.dumps(context, ensure_ascii=False, indent=2),
        product_info=json.dumps(product_info, ensure_ascii=False, indent=2),
        all_segment_data=json.dumps(entries, ensure_ascii=False, indent=2),
        schema=json.dumps(prompt_mod.PERSONA_SCHEMA_JSON, ensure_ascii=False, indent=2),
    )

    system = prompt_mod.SYSTEM_PROMPT.strip()
    user = user_prompt.strip()
    logger.debug(f"[prompt] module={'MC' if prompt_mod is prompt_mc else 'SW'} "
                 f"system_len={len(system)}, user_len={len(user)}")
    # NEW: entries / uuids / mapping 추가 반환
    return {"system": system, "user": user, "entries": entries, "uuids": expected_uuids, "uuid_map": uuid_to_entry}

# NEW: 기존 entries를 그대로 사용해 재시도 프롬프트 구성
def make_prompt_from_entries(entries: List[Dict[str, Any]], keyword: str, mode: str, context_dir: Path) -> Dict[str, str]:
    mode = mode.lower()
    prompt_mod = prompt_mc if mode in ("mc", "multiple_choice") else prompt_sw
    context, product_info = load_context_and_product_info(context_dir, keyword)
    user_prompt = prompt_mod.PERSONA_GEN_PROMPT_BASE.format(
        keyword=keyword,
        num_segments=len(entries),
        context=json.dumps(context, ensure_ascii=False, indent=2),
        product_info=json.dumps(product_info, ensure_ascii=False, indent=2),
        all_segment_data=json.dumps(entries, ensure_ascii=False, indent=2),
        schema=json.dumps(prompt_mod.PERSONA_SCHEMA_JSON, ensure_ascii=False, indent=2),
    )
    return {"system": prompt_mod.SYSTEM_PROMPT.strip(), "user": user_prompt.strip()}

# -----------------------------
# LLM 배치 실행
# -----------------------------
def run_in_batches_and_save(segment_rows: List[Dict], keyword: str, mode: str,
                            out_root: Path, context_dir: Path,
                            batch_size: int = 10, model: str = "gpt-4o-mini",
                            temperature: float = 0.2,
                            log_prompts: bool = False,
                            prompt_preview_chars: int = 400) -> List[Dict]:
    log_root = out_root / "logs" / keyword
    log_dir = log_root / "personas_log"
    dbg_dir = log_root / "debug_prompts"
    ensure_dir(out_root)
    ensure_dir(log_dir)
    if log_prompts:
        ensure_dir(dbg_dir)

    logger.info(f"[run] keyword={keyword} mode={mode} model={model} temp={temperature} "
                f"batch_size={batch_size} segments_total={len(segment_rows)}")

    all_personas: List[Dict] = []
    # NEW: 누락 uuid 엔트리들 모아 두었다가 part_101로 재생성
    recovery_entries: List[Dict[str, Any]] = []

    for i in range(0, len(segment_rows), batch_size):
        batch = segment_rows[i:i+batch_size]
        prompts = make_batch_persona_prompt(batch, keyword, mode, context_dir)

        # 기대 uuid 세트(검증용)
        expected_uuids = set(prompts.get("uuids", []) or [])
        uuid_map = prompts.get("uuid_map", {}) or {}

        if log_prompts:
            bno = i // batch_size + 1
            (dbg_dir / f"batch_{bno:03d}.system.txt").write_text(prompts["system"], encoding="utf-8")
            (dbg_dir / f"batch_{bno:03d}.user.txt").write_text(prompts["user"], encoding="utf-8")
            logger.debug(f"[prompt/save] batch={bno} "
                         f"system_preview='{preview(prompts['system'], prompt_preview_chars)}' "
                         f"user_preview='{preview(prompts['user'], prompt_preview_chars)}'")

        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ],
            temperature=temperature,
        )

        text = extract_text(resp)
        data = safe_json_loads(text)
        bno = i // batch_size + 1
        bidx = f"{i}~{i+len(batch)-1}"

        # 기본 저장
        out_path = log_dir / f"personas_{keyword}_part{bno}.json"
        out_path.write_text(json.dumps(data if data is not None else {"raw_text": text},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(f"[save] {out_path}")

        # 결과 처리/검증
        actual_items: List[Dict[str, Any]] = data if isinstance(data, list) else []
        actual_uuids = {str(it.get("uuid", "")).strip() for it in actual_items if isinstance(it, dict) and it.get("uuid")}
        item_cnt = len(actual_items)
        uuid_cnt = len(actual_uuids)

        logger.info(f"[batch {bidx}] parsed_items={item_cnt}, unique_uuids={uuid_cnt}, expected={len(expected_uuids)}")

        # 기대 개수/uuid 미달 시 누락 목록 수집
        if (item_cnt < len(expected_uuids)) or (uuid_cnt < len(expected_uuids)):
            missing = [u for u in expected_uuids if u not in actual_uuids]
            if missing:
                logger.warning(f"[batch {bidx}] missing_uuids={missing}")
                # 누락된 uuid에 해당하는 원본 entry를 복구 풀에 적재
                for u in missing:
                    ent = uuid_map.get(u)
                    if ent:
                        recovery_entries.append(ent)

        # 합본 누적
        if isinstance(data, list):
            all_personas.extend(data)

    # NEW: 누락분 재생성 → part_101.json, part_102.json, ... (batch_size로 나눔)
    if recovery_entries:
        logger.info(f"[recovery] total missing entries: {len(recovery_entries)} → re-generate in batches of {batch_size}")

        # recovery_entries를 batch_size로 나눠서 101, 102, ... 증분 번호로 저장
        saved_parts = 0
        for offset, start in enumerate(range(0, len(recovery_entries), batch_size), start=101):
            chunk = recovery_entries[start:start+batch_size]

            # 디버그 프롬프트 저장(옵션)
            retry_prompts = make_prompt_from_entries(chunk, keyword, mode, context_dir)
            if log_prompts:
                (dbg_dir / f"batch_{offset:03d}.system.txt").write_text(retry_prompts["system"], encoding="utf-8")
                (dbg_dir / f"batch_{offset:03d}.user.txt").write_text(retry_prompts["user"], encoding="utf-8")

            # LLM 호출
            resp2 = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": retry_prompts["system"]},
                    {"role": "user", "content": retry_prompts["user"]},
                ],
                temperature=temperature,
            )
            text2 = extract_text(resp2)
            data2 = safe_json_loads(text2)

            # 저장 파일명: personas_{keyword}_part_{100+i}.json
            patch_path = log_dir / f"personas_{keyword}_part{offset}.json"
            patch_path.write_text(
                json.dumps(data2 if data2 is not None else {"raw_text": text2}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"[recovery] saved → {patch_path}")

            # 합본 누적
            if isinstance(data2, list):
                all_personas.extend(data2)
            saved_parts += 1

        logger.info(f"[recovery] completed: {saved_parts} patch part(s) written (starting from part_101)")

    # 최종 합본
    final_out = out_root / f"personas_{keyword}_all.json"
    final_out.write_text(json.dumps(all_personas, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[done] 합본 저장 → {final_out} (total_items={len(all_personas)})")
    return all_personas

# -----------------------------
# 메인 (argparse)
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Persona batch generator (MC / SWAP)")
    p.add_argument("--mode", default="mc", choices=["mc", "multiple_choice", "sw", "swap"],
                   help="프롬프트 모드")
    p.add_argument("--keywords", required=True,
                   help="콤마로 구분된 키워드들 예: 그릭요거트,스팸,참치캔")
    p.add_argument("--weights_dir", default="./weights", help="posterior json 디렉터리")
    p.add_argument("--context_dir", default="./data/contexts", help="컨텍스트 json 디렉터리")
    p.add_argument("--out_dir", default="./output", help="출력 루트 디렉터리")
    p.add_argument("--n_samples", type=int, default=1000, help="세그먼트 샘플 개수")
    p.add_argument("--batch_size", type=int, default=10, help="LLM 배치 크기")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI 모델명")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    # ------- logging options -------
    p.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                   help="로깅 레벨")
    p.add_argument("--log_file", default="./logs/generate_personas.log", help="로그 파일 경로")
    p.add_argument("--log_prompts", action="store_true", help="배치별 system/user 프롬프트 파일 저장")
    p.add_argument("--prompt_preview_chars", type=int, default=400, help="프롬프트 프리뷰 로그 길이")

    args = p.parse_args()

    setup_logger(level=args.log_level, log_file=args.log_file)
    logger.info("=== Persona Runner Start ===")
    logger.info(f"args: mode={args.mode}, keywords='{args.keywords}', "
                f"n_samples={args.n_samples}, batch_size={args.batch_size}, "
                f"model={args.model}, temp={args.temperature}, "
                f"weights_dir={args.weights_dir}, context_dir={args.context_dir}, out_dir={args.out_dir}, "
                f"log_prompts={args.log_prompts}, log_level={args.log_level}, log_file={args.log_file}")

    weights_dir = Path(args.weights_dir)
    context_dir = Path(args.context_dir)
    out_root = Path(args.out_dir)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    logger.info(f"parsed_keywords={keywords}")

    for kw in keywords:
        logger.info(f"--- [{kw}] 시작 ({args.mode}) ---")
        df = make_posterior_df(kw, weights_dir=weights_dir)
        sampled = sample_segments(df, n_samples=args.n_samples, seed=2025)
        sampled = assign_brand_random(sampled, kw)
        segment_rows = sampled.to_dict(orient="records")

        run_in_batches_and_save(
            segment_rows=segment_rows,
            keyword=kw,
            mode=args.mode,
            out_root=out_root,
            context_dir=context_dir,
            batch_size=args.batch_size,
            model=args.model,
            temperature=args.temperature,
            log_prompts=args.log_prompts,
            prompt_preview_chars=args.prompt_preview_chars,
        )

    logger.info("=== Persona Runner End ===")

if __name__ == "__main__":
    main()
