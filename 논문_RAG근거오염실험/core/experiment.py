"""
experiment.py
RAG 단계적 부분오염 실험 v2 — 조건별로 LLM 답변을 대량 수집해 CSV로 저장한다.
(CLAUDE.md/실험설계 v2 참고: 답변 "생성"만 담당하고, 정답/기권/오답 "판정"은 하지 않는다 — judge.py의 몫)

실험 설계:
    답변가능 질문(함정 제외)에 대해 오염 레벨 6단계 × 프롬프트 2종 = 12조건을 돌린다.

    오염 레벨 L0~L5 — 정답 근거를 5개 슬롯 중 1개에 항상 고정하고, 나머지를
        비교용 정상 청크(오염 아님)와 오염 청크(다른 항구, 유사도 상위 순으로 누적)로 채운다.
        L0=오염 0개 ... L4=오염 4개, L5*=정답 근거 없이 5개 전부 오염(v1 방식, 극한 참조).
        (search.build_graduated_context 참고)

    프롬프트 — P0(기본) / P1(기권 유도). (P2는 실패 사례 기반 자동개선 — 별도 스크립트에서
        생성 후 이 파일의 --prompts 인자로 추가 실행)

    같은 질문×레벨 조합의 근거 세트는 "한 번만" 만들어서 P0/P1이 동일하게 공유한다
    (프롬프트마다 다른 근거를 보면 결과 차이가 프롬프트 때문인지 근거가 달라서인지
    구분이 안 되므로 — 실험설계 v2 검토 참고).

    함정 질문(정답 없음)은 12조건 매트릭스에 넣지 않고, L0에서만 P0/P1의 기권율을
    확인하는 별도 보조실험으로 돌린다 (--include-trap 옵션).

사용법:
    # 파일럿 (질문 3개만, 반복 3회)
    python experiment.py --limit-questions 3 --repeats 3 --output results/pilot_v2.csv

    # 본 실험
    python experiment.py --repeats 30 --output results/main_v2.csv --workers 8

    # 함정 질문 보조실험
    python experiment.py --trap-only --repeats 30 --output results/trap_v2.csv
"""

import argparse
import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from llm import get_llm_answer
from search import build_graduated_context, search_normal

DEFAULT_QUESTIONS_FILE = "항만_질문_초안.md"
DEFAULT_GOLD_CHUNKS_FILE = "gold_chunk_ids.csv"
LEVELS = [0, 1, 2, 3, 4, 5]
PROMPTS = ["P0", "P1"]


# ─────────────────────────────────────────────
# 질문/정답근거 로드
# ─────────────────────────────────────────────
def load_questions(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip().startswith("|")]

    rows = [[cell.strip() for cell in ln.strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not re.fullmatch(r"-+", r[0])]
    header, *data = rows

    questions = []
    for i, row in enumerate(data, start=1):
        question, has_answer_label, port, source_doc = (row + ["", "", "", ""])[:4]
        questions.append({
            "question_id": f"q{i}",
            "question": question,
            "port": port.strip(),
            "expected_has_answer": "없음" not in has_answer_label,
            "source_doc": source_doc,
        })
    return questions


def load_gold_chunk_ids(path):
    """question_id -> gold_chunk_id. 함정 질문은 이 파일에 없음(정답 근거가 없으므로)."""
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["question_id"]] = row["gold_chunk_id"].strip()
    return mapping


# ─────────────────────────────────────────────
# 근거(context) 조립 — 질문×레벨 조합마다 한 번만 계산해서 캐시
# ─────────────────────────────────────────────
def build_context_string(chunks):
    if not chunks:
        return None, [], []
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]
    ports = [c["metadata"].get("port", "?") for c in chunks]
    return context, chunk_ids, ports


def build_main_context_cache(questions, gold_chunk_ids):
    """답변가능 질문 × 오염레벨(0~5) 조합마다 근거를 한 번만 만들어 캐시한다."""
    cache = {}
    answerable = [q for q in questions if q["expected_has_answer"]]
    total = len(answerable) * len(LEVELS)
    done = 0
    print(f"근거 세트 준비 중... (질문 {len(answerable)}개 × 레벨 {len(LEVELS)}단계 = {total}개)")
    for q in answerable:
        gold_id = gold_chunk_ids.get(q["question_id"])
        if not gold_id:
            print(f"  [경고] {q['question_id']}에 gold_chunk_id가 없음 — 건너뜀")
            continue
        for level in LEVELS:
            chunks = build_graduated_context(q["question"], question_port=q["port"],
                                              gold_chunk_id=gold_id, level=level)
            cache[(q["question_id"], level)] = build_context_string(chunks)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}]")
    return cache


# ─────────────────────────────────────────────
# 시행 실행
# ─────────────────────────────────────────────
def run_trial(question_record, prompt_code, context, chunk_ids, ports, level, is_trap):
    result = get_llm_answer(
        question_record["question"],
        context=context,
        abstain_instruction=(prompt_code == "P1"),
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_id": question_record["question_id"],
        "question": question_record["question"],
        "port": question_record["port"],
        "is_trap": is_trap,
        "level": level,
        "condition": f"L{level}_{prompt_code}",
        "prompt": prompt_code,
        "num_evidence": len(chunk_ids),
        "evidence_chunk_ids": ";".join(chunk_ids),
        "evidence_ports": ";".join(ports),
        "answer": result["answer"],
        "is_error": result["answer"].startswith("오류:"),
    }


def build_trial_specs(questions, gold_chunk_ids, context_cache, repeats, trap_only):
    """실행할 시행 목록을 미리 다 만들어둔다 (동시실행 시 그대로 제출)."""
    specs = []

    if trap_only:
        trap_qs = [q for q in questions if not q["expected_has_answer"]]
        for q in trap_qs:
            chunks = search_normal(q["question"], top_k=5, port=q["port"])
            context, chunk_ids, ports = build_context_string(chunks)
            for prompt_code in PROMPTS:
                for _ in range(repeats):
                    specs.append((q, prompt_code, context, chunk_ids, ports, 0, True))
        return specs

    answerable = [q for q in questions if q["expected_has_answer"] and q["question_id"] in gold_chunk_ids]
    for q in answerable:
        for level in LEVELS:
            key = (q["question_id"], level)
            if key not in context_cache:
                continue
            context, chunk_ids, ports = context_cache[key]
            for prompt_code in PROMPTS:
                for _ in range(repeats):
                    specs.append((q, prompt_code, context, chunk_ids, ports, level, False))
    return specs


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG 단계적 부분오염 실험 v2 데이터 수집")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--gold-chunks", default=DEFAULT_GOLD_CHUNKS_FILE)
    parser.add_argument("--repeats", type=int, default=1, help="조건×질문마다 반복 횟수")
    parser.add_argument("--limit-questions", type=int, default=None, help="답변가능 질문 수를 앞에서부터 N개로 제한 (파일럿용)")
    parser.add_argument("--trap-only", action="store_true", help="함정 질문 보조실험만 실행 (L0, P0/P1)")
    parser.add_argument("--workers", type=int, default=8, help="동시 실행 개수 (OpenAI 속도제한 걸리면 낮출 것)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    gold_chunk_ids = load_gold_chunk_ids(args.gold_chunks)

    if args.limit_questions and not args.trap_only:
        answerable = [q for q in questions if q["expected_has_answer"]][: args.limit_questions]
        trap = [q for q in questions if not q["expected_has_answer"]]
        questions = answerable + trap

    output_path = args.output or f"results/experiment_v2_{datetime.now():%Y%m%d_%H%M%S}.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.trap_only:
        context_cache = {}
    else:
        context_cache = build_main_context_cache(questions, gold_chunk_ids)

    specs = build_trial_specs(questions, gold_chunk_ids, context_cache, args.repeats, args.trap_only)
    total = len(specs)
    mode = "함정 보조실험(L0만)" if args.trap_only else "본 실험(12조건)"
    print(f"\n{mode} — 총 {total}회 호출 예정 (동시 {args.workers}개)")
    print(f"결과 저장 위치: {output_path}\n")

    fieldnames = [
        "timestamp", "question_id", "question", "port", "is_trap", "level", "condition",
        "prompt", "num_evidence", "evidence_chunk_ids", "evidence_ports", "answer", "is_error",
    ]

    done = 0
    start = time.time()
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_trial, *spec) for spec in specs]
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                f.flush()
                done += 1
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - start
                    print(f"  [{done}/{total}] {elapsed:.1f}초 경과")

    print(f"\n완료: {done}건 저장 → {output_path}")


if __name__ == "__main__":
    main()
