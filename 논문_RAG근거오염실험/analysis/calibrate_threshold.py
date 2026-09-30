"""
calibrate_threshold.py
app.py의 신뢰도 임계값(THRESHOLD_SAFE / THRESHOLD_WARNING)을 실제 데이터로
정하기 위한 일회성 보조 스크립트.

질문 10개(답변 가능 7개 + 함정/답 없음 3개)를 각 3회씩 돌려
"답변↔근거 청크" 코사인 유사도 분포를 기록하고 요약한다.

연구 파이프라인(experiment.py/judge.py/analyze.py/lab.py)과는 무관하며,
app.py의 검색·유사도 계산 로직을 그대로 재사용한다 (app.py 자체는 수정하지 않음).

사용법:
    python calibrate_threshold.py
"""

import csv
from datetime import datetime

from app import compute_similarities, load_collection, load_model, search_with_diversity
from llm import get_llm_answer

TOP_K = 10
REPEATS = 3
OUTPUT_CSV = "results/threshold_calibration.csv"

# (질문, 답변 가능 여부) — 7개는 문서에 실제 답이 있는 질문, 3개는 함정(답 없음) 질문
QUESTIONS = [
    ("정기안전점검은 얼마나 자주 실시해야 하나요?", True),
    ("책임기술자가 되려면 어떤 자격이 필요한가요?", True),
    ("TBM은 몇 단계로 진행되나요?", True),
    ("현문사다리의 설치 각도는 몇 도인가요?", True),
    ("울산항에서 40인치 이상 대형 파이프를 야적할 때 주의사항은 무엇인가요?", True),
    ("SBM 하역작업에서 부이 접근속도는 어떻게 조절하나요?", True),
    ("항만구역에 출입하는 근로자가 착용해야 하는 기본 보호구는 무엇인가요?", True),
    ("선박 접안료 산정 기준은 무엇인가요?", False),
    ("2026년 개정 예정인 접안료 산정 기준은 무엇인가요?", False),
    ("외국인 선원이 접안 작업에 참여하려면 별도로 어떤 자격증이 필요한가요?", False),
]


def run_once(question, model):
    chunks = search_with_diversity(question, top_k=TOP_K)
    if not chunks:
        return None

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    result = get_llm_answer(question, context=context, abstain_instruction=True)
    answer = result["answer"]
    if answer.startswith("오류:"):
        return None

    sims = compute_similarities(answer, chunks, model)
    return {
        "max_similarity": max(sims),
        "min_similarity": min(sims),
        "num_chunks": len(chunks),
        "answer": answer,
    }


def print_stats(label, values):
    if not values:
        print(f"{label}: 데이터 없음")
        return
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2]
    print(f"{label}: n={n}  min={values[0]:.2f}  max={values[-1]:.2f}  mean={mean:.2f}  median={median:.2f}")


def main():
    model = load_model()
    load_collection()

    rows = []
    for question, expected in QUESTIONS:
        for rep in range(1, REPEATS + 1):
            print(f"[{'답변가능' if expected else '함정'}] {question}  (rep {rep}/{REPEATS})")
            r = run_once(question, model)
            if r is None:
                print("  -> 검색 결과 없음/오류, 건너뜀")
                continue
            rows.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "expected_has_answer": expected,
                "rep": rep,
                "max_similarity": round(r["max_similarity"], 4),
                "min_similarity": round(r["min_similarity"], 4),
                "num_chunks": r["num_chunks"],
                "answer": r["answer"],
            })
            print(f"  -> max_similarity={rows[-1]['max_similarity']}")

    if not rows:
        print("기록된 결과가 없습니다.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)}건 저장 완료 -> {OUTPUT_CSV}")

    print("\n=== 유사도(max_similarity) 분포 요약 ===")
    print_stats("답변 가능 질문", [r["max_similarity"] for r in rows if r["expected_has_answer"]])
    print_stats("함정(답 없음) 질문", [r["max_similarity"] for r in rows if not r["expected_has_answer"]])


if __name__ == "__main__":
    main()
