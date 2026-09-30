"""
verify_overclaim.py
'근거초과' 판정이 진짜인지 원문 대조로 확인한다.

문제 (2026-09-20 발견):
  judge에 넘긴 '정답 근거'는 정답 청크를 요약한 짧은 문구다. 모델이 정답 청크의
  나머지 내용까지 충실히 답하면, 채점자가 그것을 "근거에 없는 부가설명"으로 보고
  근거초과로 찍는다. 실제로는 제공된 근거 안에 있는 내용이므로 4분류 규칙상 정답이다.
  (Q2 단답형에서 1건 확인 — "시험편 절단·도면 명시"가 c29에 그대로 있었음)

방법:
  근거초과로 찍힌 답변에서 정답 근거 요약본에 없는 문장만 뽑아,
  그 문장이 **모델이 실제로 받은 근거 전체**(정답 청크 + 방해 청크 + 정상 청크)에
  있는지 n-gram 겹침으로 확인한다.
  - 근거 안에 있으면  -> 오판(실제로는 정답)
  - 근거 밖이면      -> 진짜 근거초과
"""
import csv
import re
from collections import Counter, defaultdict

from questions_explanatory import EXPLANATORY
from run_designed_contamination import (
    QUESTIONS, build_context_string, build_designed_context, load_distractors,
)

N = 5  # n-gram 길이. 8은 어미·조사 차이에 너무 민감해 오탐 발생(2026-09-20 확인)


def norm(t):
    t = re.sub(r"[^가-힣A-Za-z0-9㎜%/.]", "", t)
    return t


def grams(t, n=N):
    t = norm(t)
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()


def sentences(t):
    return [s.strip() for s in re.split(r"[.\n·•]|(?<=습니다)|(?<=합니다)", t) if len(s.strip()) > 12]


def build_registry():
    reg = {}
    for qid, (q, port, ref, gk, dk) in QUESTIONS.items():
        reg[qid] = (q, port, ref, qid)
    for qid, (q, port, ref, gk, dk, src) in EXPLANATORY.items():
        reg[qid] = (q, port, ref, src)
    return reg


def main():
    reg = build_registry()
    dmap = load_distractors()

    for path, name in [("results/designed_strict.csv", "단답형"),
                       ("results/exp_explanatory_strict.csv", "설명형")]:
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))

        def corrected(r):
            g = r["정답값_포함"] == "True"
            d = r["방해값_포함"] == "True"
            return "오답" if (not g and d) else r["judge_label_4way"]

        target = [r for r in rows if r["level"] in "12345" and corrected(r) == "근거초과"]
        total = len([r for r in rows if r["level"] in "12345"])
        print(f"\n{'='*70}\n{name} — 근거초과 {len(target)}건 / {total}건 검증\n{'='*70}")
        if not target:
            print("  없음")
            continue

        # 문항×레벨별 근거 전체 캐시
        ctx = {}
        for qid in {r["question_id"] for r in target}:
            q, port, ref, src = reg[qid]
            gold_id, ds = dmap[src]
            for lv in {r["level"] for r in target if r["question_id"] == qid}:
                ctx[(qid, lv)] = grams(build_context_string(
                    build_designed_context(q, port, gold_id, ds, int(lv))))

        verdict = Counter()
        byq = defaultdict(Counter)
        examples = {"오판": [], "진짜": []}
        for r in target:
            qid = r["question_id"]
            ref_g = grams(reg[qid][2])
            ctx_g = ctx[(qid, r["level"])]
            q_g = grams(reg[qid][0])   # 질문 자체 (되풀이 문장 제외용)
            extra = []          # 정답 근거 요약본 밖 문장
            for s in sentences(r["answer"]):
                sg = grams(s)
                if not sg:
                    continue
                if len(sg & q_g) / len(sg) >= 0.5:       # 질문을 되풀이한 도입부
                    continue
                if len(sg & ref_g) / len(sg) < 0.4:      # 요약본과 거의 안 겹침
                    extra.append((s, sg))
            if not extra:
                v = "판단불가"
            else:
                # 요약본 밖 문장들이 근거 전체 안에 있는가
                inside = sum(1 for s, sg in extra if len(sg & ctx_g) / len(sg) >= 0.5)
                v = "오판" if inside == len(extra) else ("진짜" if inside == 0 else "혼합")
            verdict[v] += 1
            byq[qid][v] += 1
            if v in examples and len(examples[v]) < 2 and extra:
                examples[v].append((r, extra[0][0]))

        for k in ("오판", "진짜", "혼합", "판단불가"):
            if verdict[k]:
                lab = {"오판": "근거 안에 있음 → 사실상 정답",
                       "진짜": "근거 밖 내용 → 진짜 근거초과",
                       "혼합": "일부만 근거 안",
                       "판단불가": "요약본 밖 문장 없음"}[k]
                print(f"  {k:5s} {verdict[k]:4d}건 ({verdict[k]/len(target)*100:5.1f}%)  — {lab}")

        real = verdict["진짜"] + verdict["혼합"]
        print(f"\n  → 근거초과 {len(target)/total*100:.1f}% 중 "
              f"진짜는 약 {real/total*100:.1f}%p (오판 {verdict['오판']/total*100:.1f}%p)")

        print("\n  [문항별]")
        for q in sorted(byq, key=lambda x: -sum(byq[x].values())):
            c = byq[q]
            print(f"    {q:6s} 총{sum(c.values()):3d}  오판 {c['오판']:3d}  진짜 {c['진짜']:3d}  혼합 {c['혼합']:3d}")

        for k, lst in examples.items():
            for r, s in lst:
                print(f"\n  [{k} 예시] {r['question_id']} L{r['level']} {r['prompt']}")
                print(f"    문제된 문장: {s[:110]}")


if __name__ == "__main__":
    main()
