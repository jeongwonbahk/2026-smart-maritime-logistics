"""
analyze.py (v2)
judge.py가 채점한 CSV(들)을 모아 오염 레벨×프롬프트 조건별로 집계한다.
(experiment.py로 답변 생성 → judge.py로 채점 → analyze.py로 집계)

v1과의 차이:
    - 조건이 A/B/B_dirty/C/C_dirty가 아니라 L0~L5(오염레벨) × P0/P1(프롬프트) 12조건.
    - 핵심 결과는 "B_dirty vs C_dirty" 단일 비교가 아니라, 오염 레벨이 올라갈 때
      오답률이 P0/P1에서 각각 어떻게 변하는지 보여주는 "임계점 곡선".
    - 함정 질문(is_trap=True)은 별도 표로 분리 (12조건 매트릭스에 안 섞음).
    - 항구별로 쪼개지 않고 전체 합산으로 본다 (항구당 표본이 작아 개별 비교는 노이즈 큼 —
      실험설계 v2 검토 참고).
    - 반복(30회 등)을 표본 수처럼 취급하지 않기 위해, 질문 단위로 먼저 집계한 결과도
      --by-question 옵션으로 따로 뽑아볼 수 있게 함.
    - [2026-09-11 발견] 질문 유형에 따라 오염과 무관한 기저 오답률 차이가 크다: 단순
      사실조회형(숫자·정의를 묻는 질문)은 L0(오염 0%)에서도 오답률이 0~3%인 반면,
      설명형(이유·점검절차·단계구성을 묻는 질문)은 L0에서도 73~100%로 이미 높다 —
      모델이 원문의 구체적 근거 대신 그럴듯한 일반 상식으로 답을 대체하는 경향 때문.
      question_types.csv로 질문을 분류해 유형별로도 함께 집계한다 (--split-by-type).

사용법:
    python analyze.py --input results/main_v2_judged.csv
    python analyze.py --input results/main_v2_judged.csv results/trap_v2_judged.csv --output results/summary_v2.csv
    python analyze.py --input results/main_v2_judged.csv --by-question results/by_question_v2.csv
"""

import argparse
import csv
import glob
import math
from collections import Counter, defaultdict

LABELS = ["정답", "기권", "오답", "판정불가"]
LEVELS = ["0", "1", "2", "3", "4", "5"]
PROMPTS = ["P0", "P1"]


def wilson_interval(successes, n, z=1.96):
    """비율의 95% Wilson score 신뢰구간. 표본이 적을 때도 정규근사보다 안정적."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_judged_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    return rows


def is_trap_row(row):
    return str(row.get("is_trap", "")).strip().lower() in ("true", "1")


def load_question_types(path="question_types.csv"):
    """question_id -> '사실형'/'설명형'. 없으면 빈 dict(유형별 분석 생략)."""
    mapping = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                mapping[row["question_id"]] = row["question_type"].strip()
    except FileNotFoundError:
        pass
    return mapping


# ─────────────────────────────────────────────
# 조건별(L{level}_{prompt}) 집계 — 메인 12조건 매트릭스
# ─────────────────────────────────────────────
def summarize_by_condition(rows):
    by_condition = defaultdict(Counter)
    for row in rows:
        by_condition[row["condition"]][row.get("judge_label") or "판정불가"] += 1

    summary = []
    for condition, counts in by_condition.items():
        total = sum(counts.values())
        entry = {"condition": condition, "total": total}
        for label in LABELS:
            n = counts.get(label, 0)
            rate = n / total if total else 0.0
            lo, hi = wilson_interval(n, total)
            entry[f"{label}_수"] = n
            entry[f"{label}_비율(%)"] = round(rate * 100, 1)
            entry[f"{label}_95%CI"] = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
        summary.append(entry)

    def sort_key(e):
        level, prompt = e["condition"].split("_")
        return (int(level[1:]), prompt)

    summary.sort(key=sort_key)
    return summary


def print_summary_table(summary):
    print(f"{'조건':<8}{'총':>6}   {'정답':<18}{'기권':<18}{'오답':<18}")
    for e in summary:
        print(
            f"{e['condition']:<8}{e['total']:>6}   "
            f"{e['정답_수']}건({e['정답_비율(%)']}%){'':<6}"
            f"{e['기권_수']}건({e['기권_비율(%)']}%){'':<6}"
            f"{e['오답_수']}건({e['오답_비율(%)']}%)"
        )


# ─────────────────────────────────────────────
# 질문 유형별(사실형/설명형) 오염 레벨 곡선
# ─────────────────────────────────────────────
def summarize_by_type_level(rows, type_map):
    """(질문유형, 레벨, 프롬프트)별로 오답률 집계. type_map에 없는 질문은 제외."""
    by_key = defaultdict(Counter)
    for row in rows:
        qtype = type_map.get(row["question_id"])
        if not qtype:
            continue
        by_key[(qtype, row["level"], row["prompt"])][row.get("judge_label") or "판정불가"] += 1

    out = {}
    for (qtype, level, prompt), counts in by_key.items():
        total = sum(counts.values())
        wrong = counts.get("오답", 0)
        lo, hi = wilson_interval(wrong, total)
        out[(qtype, level, prompt)] = {
            "total": total, "오답_수": wrong,
            "오답_비율(%)": round(wrong / total * 100, 1) if total else 0.0,
            "CI": f"[{lo*100:.1f}, {hi*100:.1f}]",
        }
    return out


def print_type_curves(by_type_level):
    if not by_type_level:
        return
    types = sorted({k[0] for k in by_type_level})
    print("\n질문 유형별 오염 레벨 곡선 (오염과 무관한 기저 오답률 차이 확인용)")
    for qtype in types:
        print(f"\n  [{qtype}]")
        print(f"  {'레벨':<6}{'P0 오답률':<20}{'P1 오답률':<20}")
        for level in LEVELS:
            p0 = by_type_level.get((qtype, level, "P0"))
            p1 = by_type_level.get((qtype, level, "P1"))
            if not p0 or not p1:
                continue
            print(f"  L{level:<5}{p0['오답_비율(%)']}% {p0['CI']:<12}"
                  f"{p1['오답_비율(%)']}% {p1['CI']:<12}")


# ─────────────────────────────────────────────
# 핵심 결과 — 오염 레벨별 오답률 곡선 (P0 vs P1)
# ─────────────────────────────────────────────
def print_threshold_curve(summary):
    by_cond = {e["condition"]: e for e in summary}

    print("\n핵심 결과 — 오염 레벨별 오답률 (P0 vs P1)")
    print(f"{'레벨':<6}{'P0 오답률':<22}{'P1 오답률':<22}{'방어 효과(P0-P1)':<10}")
    for level in LEVELS:
        p0 = by_cond.get(f"L{level}_P0")
        p1 = by_cond.get(f"L{level}_P1")
        if not p0 or not p1:
            continue
        p0_rate, p1_rate = p0["오답_비율(%)"], p1["오답_비율(%)"]
        gap = p0_rate - p1_rate
        label = " (극한·참조)" if level == "5" else ""
        print(f"L{level}{label:<6}{p0_rate}% {p0['오답_95%CI']:<14}"
              f"{p1_rate}% {p1['오답_95%CI']:<14}{gap:+.1f}%p")

    print("\n(P0 오답률이 급증하기 시작하는 레벨이 임계점, 그 지점에서 P0-P1 격차가 클수록 "
          "기권 유도 프롬프트의 방어 효과가 크다는 뜻)")


# ─────────────────────────────────────────────
# 함정 질문 — 별도 보조 결과
# ─────────────────────────────────────────────
def summarize_trap(rows):
    trap_rows = [r for r in rows if is_trap_row(r)]
    if not trap_rows:
        return None
    by_prompt = defaultdict(Counter)
    for row in trap_rows:
        by_prompt[row["prompt"]][row.get("judge_label") or "판정불가"] += 1

    print("\n보조 결과 — 함정 질문(정답 없음) 기권율 (L0만, P0 vs P1)")
    print(f"{'프롬프트':<8}{'총':>6}   {'기권':<18}{'오답':<18}")
    for prompt in PROMPTS:
        counts = by_prompt.get(prompt, Counter())
        total = sum(counts.values())
        if total == 0:
            continue
        abstain_n = counts.get("기권", 0)
        wrong_n = counts.get("오답", 0)
        lo_a, hi_a = wilson_interval(abstain_n, total)
        lo_w, hi_w = wilson_interval(wrong_n, total)
        print(f"{prompt:<8}{total:>6}   "
              f"{abstain_n}건({abstain_n/total*100:.1f}%, CI[{lo_a*100:.1f},{hi_a*100:.1f}]){'':<2}"
              f"{wrong_n}건({wrong_n/total*100:.1f}%, CI[{lo_w*100:.1f},{hi_w*100:.1f}])")
    print("(정답 없는 질문에 대해 오답=근거 없이 확정적으로 답을 지어낸 경우 — 안전 도메인에서 "
          "가장 위험한 실패 유형)")


# ─────────────────────────────────────────────
# 질문 단위 집계 (표본을 반복횟수로 착각하지 않기 위한 보조 결과)
# ─────────────────────────────────────────────
def summarize_by_question(rows):
    """질문×조건마다 오답률을 먼저 계산 — 조건 비교 전에 특정 질문이 결과를 좌우하는지 점검용."""
    by_key = defaultdict(Counter)
    for row in rows:
        if is_trap_row(row):
            continue
        key = (row["question_id"], row["condition"])
        by_key[key][row.get("judge_label") or "판정불가"] += 1

    out = []
    for (qid, condition), counts in by_key.items():
        total = sum(counts.values())
        wrong = counts.get("오답", 0)
        out.append({
            "question_id": qid,
            "condition": condition,
            "total": total,
            "오답_수": wrong,
            "오답_비율(%)": round(wrong / total * 100, 1) if total else 0.0,
        })
    out.sort(key=lambda e: (e["question_id"], e["condition"]))
    return out


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="judge.py 채점 결과(v2)를 오염레벨×프롬프트로 집계")
    parser.add_argument("--input", nargs="+", required=True, help="judge.py 결과 CSV 경로(들), 와일드카드 가능")
    parser.add_argument("--output", default=None, help="조건별 집계 결과를 CSV로 저장할 경로 (선택)")
    parser.add_argument("--by-question", default=None, help="질문×조건별 오답률을 CSV로 저장할 경로 (선택)")
    parser.add_argument("--question-types", default="question_types.csv", help="질문 유형(사실형/설명형) 매핑 CSV")
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    rows = load_judged_rows(paths)
    if not rows:
        print("[중단] 입력 파일에서 데이터를 찾지 못했습니다.")
        return

    rows_with_label = [r for r in rows if r.get("judge_label")]
    if len(rows_with_label) < len(rows):
        print(f"[경고] judge_label이 없는 행 {len(rows) - len(rows_with_label)}건은 제외했습니다.")

    main_rows = [r for r in rows_with_label if not is_trap_row(r)]

    print(f"입력 파일 {len(paths)}개, 총 {len(rows_with_label)}건 (본 매트릭스 {len(main_rows)}건 "
          f"+ 함정 {len(rows_with_label) - len(main_rows)}건)\n")

    summary = summarize_by_condition(main_rows)
    print_summary_table(summary)
    print_threshold_curve(summary)

    type_map = load_question_types(args.question_types)
    if type_map:
        by_type_level = summarize_by_type_level(main_rows, type_map)
        print_type_curves(by_type_level)
    else:
        print(f"\n[참고] '{args.question_types}' 없음 — 질문 유형별 분석 생략")

    summarize_trap(rows_with_label)

    if args.output:
        save_csv(summary, args.output)
        print(f"\n조건별 집계 저장: {args.output}")

    if args.by_question:
        by_q = summarize_by_question(main_rows)
        save_csv(by_q, args.by_question)
        print(f"질문×조건별 집계 저장: {args.by_question}")


if __name__ == "__main__":
    main()
