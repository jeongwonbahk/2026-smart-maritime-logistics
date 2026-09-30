"""
run_2x2_experiment.py
질문유형(단답형/설명형) x 오염레벨(L0~L5) x 프롬프트(P0/P1) 2x2 설계 실험.

단답형 11문항(run_designed_contamination.QUESTIONS)과 설명형 11문항
(questions_explanatory.EXPLANATORY)은 **완전히 같은 정답 근거·방해 근거 청크**를
쓰고 질문 형태만 다르다. 따라서 두 세트의 차이는 순수하게 '질문 유형' 효과다.

22문항 x 6레벨 x 2프롬프트 x 30회 = 7,920건.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from llm import get_llm_answer
from questions_explanatory import EXPLANATORY
from run_designed_contamination import (
    JUDGE_PROMPT_4WAY, LEVELS, PROMPTS, QUESTIONS, VALID_LABELS,
    build_context_string, build_designed_context, keyword_check, load_distractors,
)

load_dotenv()
client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"


def build_all_questions(types=("단답형", "설명형")):
    """(질문, 항구, 정답요약, 정답kw, 방해kw, 유형, 청크출처id)"""
    allq = {}
    if "단답형" in types:
        for qid, (q, port, ref, gk, dk) in QUESTIONS.items():
            allq[qid] = (q, port, ref, gk, dk, "단답형", qid)
    if "설명형" in types:
        for qid, (q, port, ref, gk, dk, src) in EXPLANATORY.items():
            allq[qid] = (q, port, ref, gk, dk, "설명형", src)
    return allq


def judge_4way(question, answer, reference, full_context):
    user = (f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n"
            f"[모델이 실제로 참고한 근거 전체]\n{full_context}\n\n[모델 답변]\n{answer}")
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_PROMPT_4WAY},
                      {"role": "user", "content": user}])
        p = json.loads(r.choices[0].message.content)
        lab = str(p.get("label", "")).strip()
        if lab not in VALID_LABELS:
            return {"label": "판정불가", "reason": f"라벨 오류: {p}"}
        return {"label": lab, "reason": str(p.get("reason", "")).strip()}
    except Exception as e:
        return {"label": "판정불가", "reason": f"오류: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--types", default="설명형",
                    help="쉼표 구분: 단답형, 설명형 (기본: 설명형만)")
    ap.add_argument("--out", default="exp_explanatory",
                    help="results/<out>.csv, results/<out>_judged.csv 로 저장")
    args = ap.parse_args()

    types = tuple(t.strip() for t in args.types.split(","))
    allq = build_all_questions(types)
    dmap = load_distractors()
    total = len(allq) * 6 * 2 * args.repeats
    n_short = sum(1 for v in allq.values() if v[5] == "단답형")
    print(f"2x2 실험: 단답형 {n_short} + 설명형 {len(allq)-n_short} = {len(allq)}문항")
    print(f"  x 6레벨 x 2프롬프트 x {args.repeats}회 = {total}건")

    # 청크 출처가 같은 문항끼리 컨텍스트를 공유 (단답형/설명형이 동일 근거를 보게 함)
    ctx = {}
    for qid, (qtext, port, _, _, _, _, src) in allq.items():
        gold_id, ds = dmap[src]
        for lv in LEVELS:
            chunks = build_designed_context(qtext, port, gold_id, ds, lv)
            ctx[(qid, lv)] = build_context_string(chunks)
    print(f"컨텍스트 구성 완료: {len(ctx)}개")

    specs = [(q, lv, p) for q in allq for lv in LEVELS for p in PROMPTS
             for _ in range(args.repeats)]

    def gen(qid, lv, p):
        qtext, port, _, _, _, qtype, src = allq[qid]
        r = get_llm_answer(qtext, context=ctx[(qid, lv)], abstain_instruction=(p == "P1"))
        return {"question_id": qid, "question_type": qtype, "chunk_src": src,
                "question": qtext, "port": port, "level": lv, "prompt": p,
                "answer": r.get("answer", "")}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen, *s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 400 == 0 or i == len(specs):
                print(f"  [생성 {i}/{len(specs)}]")

    with open("results/%s.csv" % args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("생성 저장: results/%s.csv" % args.out)

    def do_judge(r):
        _, _, ref, gk, dk, _, _ = allq[r["question_id"]]
        res = judge_4way(r["question"], r["answer"], ref, ctx[(r["question_id"], r["level"])])
        hg, hd = keyword_check(r["answer"], gk, dk)
        o = dict(r)
        o["judge_label_4way"] = res["label"]
        o["judge_reason_4way"] = res["reason"]
        o["정답값_포함"] = hg
        o["방해값_포함"] = hd
        return o

    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(do_judge, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            judged.append(f.result())
            if i % 400 == 0 or i == len(rows):
                print(f"  [채점 {i}/{len(rows)}]")

    with open(f"results/{args.out}_judged.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(judged[0].keys()))
        w.writeheader(); w.writerows(judged)
    print(f"채점 저장: results/{args.out}_judged.csv")

    print("\n=== 질문유형 x 레벨 x 프롬프트 — 오답률 ===")
    by = defaultdict(Counter)
    for r in judged:
        by[(r["question_type"], r["level"], r["prompt"])][r["judge_label_4way"]] += 1
    for t in types:
        print(f"\n[{t}]")
        for lv in "012345":
            line = f"  L{lv}: "
            for p in ("P0", "P1"):
                c = by[(t, lv, p)]; n = sum(c.values())
                line += f"{p} 오답 {c['오답']/n*100:5.1f}% ({c['오답']:3d}/{n})   "
            print(line)

    print("\n=== 부분오염(L1~L4) 합산 ===")
    for t in types:
        for p in ("P0", "P1"):
            c = Counter()
            for lv in "1234":
                c += by[(t, lv, p)]
            n = sum(c.values())
            print(f"  {t} {p}: 오답 {c['오답']/n*100:5.1f}% ({c['오답']}/{n})  "
                  f"정답 {c['정답']/n*100:.1f}%  근거초과 {c['근거초과']/n*100:.1f}%  기권 {c['기권']/n*100:.1f}%")


if __name__ == "__main__":
    main()
