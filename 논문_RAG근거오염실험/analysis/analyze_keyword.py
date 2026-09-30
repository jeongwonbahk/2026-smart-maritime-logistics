"""
analyze_keyword.py
문자열 검사 기준으로 오염 채택률을 집계한다.

왜 LLM 판정 대신 이걸 쓰는가 (2026-09-20 검증 결과):
  - 기존(관대) 판정: 설명형에서 정답값이 없고 방해값만 있는 답변도 '정답'으로 인정 →
    실제 오답 13.6%를 0.3%로 과소평가.
  - 엄격 판정: 반대로 과교정 → 48.1%, 그중 34.2%p가 정답값이 답변에 있는데도 오답 처리.
    표본 검수 결과 "15.7원이 아니라 15.7원", 어미 차이('없습니다' vs '없다')를 오답 처리,
    방해 근거 값을 정답으로 착각하는 등 명백한 오류 다수.
  - 단답형에서는 기존 판정과 문자열 검사가 100% 일치(불일치 0건)했다. 즉 LLM 채점자는
    단답형에서는 정확하고 다항목 설명형에서만 무너진다.

따라서 이 실험에서는 **방해 근거를 우리가 직접 지정했다는 점**을 이용해,
답변에 어떤 값이 들어갔는지를 문자열로 직접 판정한다. LLM이 개입하지 않으므로
재현 가능하고 질문 유형에 따른 편향이 없다.

분류:
  정답채택 : 정답값 O, 방해값 X   — 올바른 값만 제시
  병기     : 정답값 O, 방해값 O   — 둘 다 언급 (정답은 살아있음)
  오염채택 : 정답값 X, 방해값 O   — 방해 근거 값만 제시  <= 위험한 실패
  기타     : 둘 다 X             — 기권하거나 값을 언급하지 않음
"""
import argparse
import csv
from collections import Counter, defaultdict

LABELS = ["정답채택", "병기", "오염채택", "기타"]


def classify(row):
    g = row["정답값_포함"] == "True"
    d = row["방해값_포함"] == "True"
    if g and not d:
        return "정답채택"
    if g and d:
        return "병기"
    if not g and d:
        return "오염채택"
    return "기타"


def pct(c, n):
    return {k: round(c[k] / n * 100, 1) for k in LABELS}


def summarize(path, label):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    for r in rows:
        r["kw"] = classify(r)

    print(f"\n{'='*74}\n{label}  ({len(rows)}건)\n{'='*74}")

    print("\n[레벨 x 프롬프트]  정답채택 / 병기 / 오염채택 / 기타")
    by = defaultdict(Counter)
    for r in rows:
        by[(r["level"], r["prompt"])][r["kw"]] += 1
    for lv in "012345":
        line = f"  L{lv}  "
        for p in ("P0", "P1"):
            c = by[(lv, p)]
            n = sum(c.values())
            v = pct(c, n)
            line += (f"{p}: {v['정답채택']:5.1f} /{v['병기']:5.1f} /"
                     f"{v['오염채택']:5.1f} /{v['기타']:5.1f}   ")
        print(line)

    print("\n[부분오염 L1~L4 합산]")
    out = {}
    for p in ("P0", "P1"):
        c = Counter(r["kw"] for r in rows if r["prompt"] == p and r["level"] in "1234")
        n = sum(c.values())
        v = pct(c, n)
        out[p] = v
        print(f"  {p}: 정답채택 {v['정답채택']}%  병기 {v['병기']}%  "
              f"오염채택 {v['오염채택']}%  기타 {v['기타']}%")

    print("\n[문항별 오염채택률 — L1~L4, P0]")
    bq = defaultdict(Counter)
    for r in rows:
        if r["prompt"] == "P0" and r["level"] in "1234":
            bq[r["question_id"]][r["kw"]] += 1
    for q in sorted(bq, key=lambda x: -bq[x]["오염채택"]):
        c = bq[q]
        n = sum(c.values())
        print(f"  {q:6s} 오염채택 {c['오염채택']:3d}/{n} ({c['오염채택']/n*100:5.1f}%)"
              f"   병기 {c['병기']:3d}   정답채택 {c['정답채택']:3d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="results/designed_strict.csv")
    ap.add_argument("--long", default="results/exp_explanatory_strict.csv")
    args = ap.parse_args()

    s = summarize(args.short, "단답형 (설계 오염)")
    e = summarize(args.long, "설명형 (설계 오염)")

    print(f"\n{'='*74}\n최종 비교 — 부분오염(L1~L4) 오염채택률\n{'='*74}")
    print(f"{'':10s}{'단답형':>10s}{'설명형':>10s}")
    for p in ("P0", "P1"):
        print(f"  {p:8s}{s[p]['오염채택']:9.1f}%{e[p]['오염채택']:9.1f}%")
    print("\n  (오염채택 = 정답값 없이 방해 근거 값만 제시 = 위험한 실패)")


if __name__ == "__main__":
    main()
