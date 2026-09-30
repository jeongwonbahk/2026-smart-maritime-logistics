"""
rerun_questions.py
특정 문항만 다시 생성·채점하고, 기존 결과 CSV의 해당 행을 교체한다.

용도 (2026-09-20):
  Q4·Q5 질문에 "컨테이너크레인"이라는 단어가 들어 있었는데, 이 단어는 정답 근거·방해
  근거 어느 청크에도 없다(문서 전체에서 무관한 청크 1곳에만 등장). 근거에 없는 말로
  질문을 한정한 셈이라 제거하고 재실행한다. 청크(distractor_chunk_ids.csv)는 그대로다.

사용:
  python rerun_questions.py --ids Q4,Q5   --target results/designed_strict.csv
  python rerun_questions.py --ids Q4E,Q5E --target results/exp_explanatory_strict.csv
"""
import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from llm import get_llm_answer
from questions_explanatory import EXPLANATORY
from rejudge_strict import judge as judge_strict
from run_designed_contamination import (
    LEVELS, PROMPTS, QUESTIONS, build_context_string, build_designed_context,
    judge_4way, keyword_check, load_distractors,
)

load_dotenv()


def registry():
    reg = {}
    for qid, (q, port, ref, gk, dk) in QUESTIONS.items():
        reg[qid] = (q, port, ref, gk, dk, qid)
    for qid, (q, port, ref, gk, dk, src) in EXPLANATORY.items():
        reg[qid] = (q, port, ref, gk, dk, src)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="쉼표 구분 (예: Q4,Q5)")
    ap.add_argument("--target", required=True, help="교체할 결과 CSV")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ids = [s.strip() for s in args.ids.split(",")]
    reg = registry()
    dmap = load_distractors()

    old = list(csv.DictReader(open(args.target, encoding="utf-8-sig")))
    fields = list(old[0].keys())
    keep = [r for r in old if r["question_id"] not in ids]
    print(f"{args.target}: 전체 {len(old)}행 중 {len(old)-len(keep)}행 교체 예정")
    for qid in ids:
        print(f"  {qid}: {reg[qid][0]}")

    ctx = {}
    for qid in ids:
        q, port, _, _, _, src = reg[qid]
        gold_id, ds = dmap[src]
        for lv in LEVELS:
            ctx[(qid, lv)] = build_context_string(
                build_designed_context(q, port, gold_id, ds, lv))

    specs = [(q, lv, p) for q in ids for lv in LEVELS for p in PROMPTS
             for _ in range(args.repeats)]

    def work(qid, lv, p):
        q, port, ref, gk, dk, _ = reg[qid]
        c = ctx[(qid, lv)]
        ans = get_llm_answer(q, context=c, abstain_instruction=(p == "P1")).get("answer", "")
        j4 = judge_4way(q, ans, ref, c)          # 기존 4분류 판정
        js = judge_strict(q, ans, ref, c)        # 엄격 판정
        hg, hd = keyword_check(ans, gk, dk)
        row = {k: "" for k in fields}
        row.update({"question_id": qid, "question": q, "port": port,
                    "level": str(lv), "prompt": p, "answer": ans,
                    "judge_label_4way": j4["label"], "judge_reason_4way": j4["reason"],
                    "strict_label": js["label"], "strict_reason": js["reason"],
                    "정답값_포함": str(hg), "방해값_포함": str(hd),
                    "kw_verdict": ("오답의심" if (not hg and hd) else
                                   "정답값있음" if hg else "판단보류")})
        for extra in ("question_type", "chunk_src"):
            if extra in fields and not row[extra]:
                row[extra] = "설명형" if qid.endswith("E") else "단답형"
                if extra == "chunk_src":
                    row[extra] = reg[qid][5]
        return row

    new = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, *s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            new.append(f.result())
            if i % 200 == 0 or i == len(specs):
                print(f"  [{i}/{len(specs)}]")

    out = keep + new
    out.sort(key=lambda r: (r["question_id"], int(r["level"]), r["prompt"]))
    with open(args.target, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"저장: {args.target}  (총 {len(out)}행)")

    def corrected(r):
        g = r["정답값_포함"] == "True"; d = r["방해값_포함"] == "True"
        return "오답" if (not g and d) else r["judge_label_4way"]

    print("\n=== 재실행 문항 결과 (L1~L4, P0) ===")
    for qid in ids:
        sub = [r for r in new if r["question_id"] == qid and r["level"] in "1234" and r["prompt"] == "P0"]
        c = Counter(corrected(r) for r in sub)
        print(f"  {qid}: 오답 {c['오답']}/{len(sub)} ({c['오답']/len(sub)*100:.1f}%)   "
              + "  ".join(f"{k} {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
