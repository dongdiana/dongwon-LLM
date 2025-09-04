"""
segments_insight_generation.py
- segments_profile.json을 읽어 segment_key별로 프롬프트 생성
- LLM이 {"요약문": "...", "근거": ["...", "..."]} 형식으로 답하도록 유도
- 결과를 segments_profile_insights.json으로 저장
"""

from __future__ import annotations
import json, os, math, time, re
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ==== 입출력 경로 ====
ROOT = Path(__file__).resolve().parents[1]
IN_PATH  = ROOT / "data" / "segments" / "segments_profile.json" 
OUT_PATH = ROOT / "data" / "segments" / "segments_profile_insights.json"

# ==== 환경 변수 로드 ====
load_dotenv()  # .env에서 OPENAI_API_KEY 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==== 슬롯 스키마 (중복 선언 방지) ====
SLOTS = ["인사이트1","인사이트2","인사이트3","인사이트4","인사이트5"]

# ==== 인사이트 그룹 (프롬프트 조립용) ====
INSIGHT_GROUPS = {
    "인사이트1": ["지역", "교육수준", "직업"],
    "인사이트2": ["건강관심도", "건강투자정도", "운동여부", "sns사용빈도"],
    "인사이트3": ["주거형태", "가구요리빈도", "식료품구입빈도", "1회평균식료품구입금액"],
    "인사이트4": ["가공식품구입빈도", "가공식품구입기준"],
    "인사이트5": ["우유구입기준"],
}

# ==== 유틸 ====
def _strip_codeblock(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(json)?", "", s, flags=re.IGNORECASE|re.MULTILINE)
        s = s.strip("` \n")
    return s

def _safe_parse_insights(s: str) -> dict:
    s = _strip_codeblock(s)
    data = json.loads(s)  # 실패 시 예외 발생 -> 상위에서 재시도
    # 슬롯 키 보정
    out = {slot: (data.get(slot) or "").strip() for slot in SLOTS}
    return out

def call_llm(prompt: str, *, model="gpt-4o-mini", temperature=0.2, max_retries=2, retry_wait=1.5) -> dict:
    client = OpenAI()
    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[{"role":"user","content": prompt}],
            )
            content = resp.output[0].content[0].text
            insights_dict = _safe_parse_insights(content)  # ✅ 문자열을 dict로
            print(f"[LLM RAW OK] keys={list(insights_dict.keys())}")
            return insights_dict
        except Exception as e:
            print(f"[ERROR] LLM parse failed (attempt {attempt}/{max_retries}): {e}")
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_wait)
    print(f"[FATAL] LLM parse final fail: {last_err}")
    return {slot: "" for slot in SLOTS}


# ===== 프롬프트 생성기 (그룹화 버전) =====
def build_prompt_for_segment(seg: Dict[str, Any]) -> str:
    key     = seg.get("segment_key","")
    filters = seg.get("filters", {})
    size    = seg.get("size", None)
    nums    = seg.get("numerics", {})
    cats    = seg.get("categoricals", {})

    # 상위 K만 요약
    def top_cats(cat_name: str, items: List[Dict[str, Any]], k: int = 5) -> str:
        srt = sorted(items, key=lambda d: d.get("ratio",0), reverse=True)[:k]
        return ", ".join([f"{it['value']}({it['ratio']:.2f})" for it in srt]) if srt else "(없음)"

    # 수치형 포맷터
    def fmt_num(name: str, stats: Dict[str, Any]) -> str:
        def f(x):
            if x is None or (isinstance(x, float) and math.isnan(x)): return "None"
            return f"{x:.3f}" if isinstance(x, (int,float)) else str(x)
        return f"{name}: mean={f(stats.get('mean'))}, std={f(stats.get('std'))}, min={f(stats.get('min'))}, max={f(stats.get('max'))}"

    # 그룹별 블록 생성
    grouped_blocks = []
    for slot, var_list in INSIGHT_GROUPS.items():
        lines = []
        for var_name in var_list:
            # 카테고리형
            if var_name in cats:
                lines.append(f"- {var_name}: {top_cats(var_name, cats[var_name], k=5)}")
            # 수치형
            if var_name in nums:
                lines.append(f"- {fmt_num(var_name, nums[var_name])}")
        block = "\n".join(lines) if lines else "- (관련 데이터 없음)"
        grouped_blocks.append(f"[{slot}]\n{block}")

    prompt = f"""
        다음은 하나의 소비자 세그먼트 프로필입니다. 제공된 정보만 활용해
        '인사이트1'~'인사이트5'까지 각 슬롯별로 **한 문장** 요약을 만드세요.
        반드시 아래의 정확한 JSON 객체만 출력하세요(코드블럭 금지, 추가키 금지):

        {{
        "인사이트1": "<한 문장>",
        "인사이트2": "<한 문장>",
        "인사이트3": "<한 문장>",
        "인사이트4": "<한 문장>",
        "인사이트5": "<한 문장>"
        }}

        요약문 작성 규칙:
        1) 문장 시작은 항상 필터 정보를 나열하며 시작하세요.
        예) "{filters.get('가구소득','')}, {filters.get('가구원수','')}, {filters.get('성별','')}, {filters.get('연령대','')} 세그먼트는 ..."
        2) 각 슬롯은 위의 그룹 블록에서 제공된 변수들만 근거로 삼아 한 문장으로 요약합니다.
        3) 수치를 장황하게 나열하지 말고, 관찰된 특성을 압축하세요.
        4) 데이터가 관찰되지 않는 경우, 설명 문장을 쓰지 말고 반드시 빈 문자열("")을 그대로 넣으세요.
        5) 코드블럭( ``` ) 사용 금지. JSON만 반환.

        [세그먼트]
        - segment_key: {key}
        - size: {size}
        - filters: {json.dumps(filters, ensure_ascii=False)}

        [그룹별 요약 자료]
        {chr(10).join(grouped_blocks)}
        """.strip()

    return prompt


def normalize_insights(ins: dict) -> dict:
    """ 슬롯(인사이트1~5)로 강제 통일"""
    if not isinstance(ins, dict):
        return {slot: "" for slot in SLOTS}

    # 이미 슬롯 스키마면 정렬/결측 보정
    if any(k.startswith("인사이트") for k in ins.keys()):
        return {slot: (ins.get(slot) or "").strip() for slot in SLOTS}

    # 레거시 스키마를 슬롯1로 매핑하고 나머지는 빈 문자열
    if "요약문" in ins:
        first = (ins.get("요약문") or "").strip()
        return {
            "인사이트1": first,
            "인사이트2": "",
            "인사이트3": "",
            "인사이트4": "",
            "인사이트5": "",
        }

    # 그 외 예외: 전부 빈 슬롯
    return {slot: "" for slot in SLOTS}



# ==== 메인 ====
def main():
    # (1) 입력 파일 확인
    if not IN_PATH.exists():
        print(f"[ERROR] 입력 파일이 없습니다: {IN_PATH.resolve()}")
        return
    
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    new_segments = []
    for seg in segments:
        seg_key  = seg.get("segment_key","")
        seg_size = seg.get("size", 0)

        if seg_size <= 5:
            insights_raw = {slot: "" for slot in SLOTS}
        else:
            prompt = build_prompt_for_segment(seg)
            insights_raw = call_llm(prompt)
            if not isinstance(insights_raw, dict) or not any(insights_raw.values()):
                print(f"[WARN] LLM empty for segment={seg_key}, size={seg_size}")

        insights = normalize_insights(insights_raw)
        print(f"[ASSIGN] {seg_key} -> {insights}")

        seg_out = dict(seg)
        seg_out["insights"] = insights
        new_segments.append(seg_out)

    out = {
        "generated_from": str(IN_PATH),
        "generated_at": data.get("generated_at"),
        "model_meta": {"model": "gpt-4o-mini", "temperature": 0.2},
        "segments": new_segments
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved -> {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
