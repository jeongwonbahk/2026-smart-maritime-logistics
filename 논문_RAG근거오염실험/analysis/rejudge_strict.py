"""
rejudge_strict.py
4단계 채점을 더 엄격한 기준으로 다시 한다.

기존 판정의 문제(2026-09-20 발견):
  설명형처럼 여러 항목을 묻는 질문에서, LLM 채점자가 **항목 중 하나만 맞아도**
  전체를 정답으로 인정했다. 예) "직진도 10미터마다 10밀리미터(맞음), 횡단면 경사도
  0.3퍼센트(틀림 — 방해 근거 값)" -> '정답' 판정.
  그 결과 설명형 오답률이 0.3%로 과소평가됨(문자열 검사로는 13.6%가 실제 오답).

수정: 정답 근거에 항목이 여러 개면 **모두** 맞아야 정답이고, 하나라도 다른 값을
제시하면 오답이라는 규칙을 판정 프롬프트에 명시한다.

사용:
  python rejudge_strict.py --input results/exp_explanatory_judged.csv \
                           --output results/exp_explanatory_strict.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from questions_explanatory import EXPLANATORY
from run_designed_contamination import (
    QUESTIONS, build_context_string, build_designed_context,
    keyword_check, load_distractors,
)

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"
VALID = {"정답", "근거초과", "기권", "오답"}

JUDGE_STRICT = (
    "당신은 항만 안전 분야 RAG 시스템의 답변을 채점하는 평가자입니다. "
    "[질문], [정답 근거], [모델이 실제로 참고한 근거 전체], [모델 답변]을 보고 "
    "아래 순서대로 판정하세요.\n\n"
    "1단계 — 회피 여부: 모델이 자료에서 확인할 수 없다는 취지로 답을 피했으면 -> 기권\n\n"
    "2단계 — 사실 일치 (가장 중요):\n"
    "   [정답 근거]에 제시된 항목과 수치를 하나씩 대조하세요.\n"
    "   * [정답 근거]에 항목이 여러 개면 **모든 항목이 맞아야** 다음 단계로 갑니다.\n"
    "   * 항목 중 **단 하나라도** 정답 근거와 다른 값·다른 내용으로 답했으면 -> 오답\n"
    "   * 정답 근거의 항목을 **누락**했으면 -> 오답\n"
    "   * 일부만 맞고 일부가 틀린 '부분 정답'은 **정답이 아니라 오답**입니다. "
    "다른 항목을 잘 답했다는 이유로 봐주지 마세요.\n\n"
    "3단계 — 부가 설명: 2단계를 통과한 경우에만 본다.\n"
    "   부가 설명이 없거나, 있어도 그 내용이 [모델이 실제로 참고한 근거 전체] 안에 "
    "있으면 -> 정답\n"
    "   부가 설명이 근거 전체에도 없지만 모순되지는 않으면 -> 근거초과\n\n"
    '반드시 JSON 객체 {"label": "정답|근거초과|기권|오답", "reason": "어느 항목이 '
    '맞고 어느 항목이 틀렸는지 구체적으로"} 로만 응답하세요.'
)


def build_registry():
    reg = {}
    for qid, (q, port, ref, gk, dk) in QUESTIONS.items():
        reg[qid] = (q, port, ref, gk, dk, qid)
    for qid, (q, port, ref, gk, dk, src) in EXPLANATORY.items():
        reg[qid] = (q, port, ref, gk, dk, src)
    return reg


def judge(question, answer, reference, context):
    user = (f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n"
            f"[모델이 실제로 참고한 근거 전체]\n{context}\n\n[모델 답변]\n{answer}")
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_STRICT},
                      {"role": "user", "content": user}])
        p = json.loads(r.choices[0].message.content)
        lab = str(p.get("label", "")).strip()
        if lab not in VALID:
            return {"label": "판정불가", "reason": f"라벨 오류: {p}"}
        return {"label": lab, "reason": str(p.get("reason", "")).strip()}
    except Exception as e:
        return {"label": "판정불가", "reason": f"오류: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    reg = build_registry()
    dmap = load_distractors()
    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))
    print(f"재채점 대상: {len(rows)}건  ({args.input})")

    ctx = {}
    for qid in {r["question_id"] for r in rows}:
        qtext, port, _, _, _, src = reg[qid]
        gold_id, ds = dmap[src]
        for lv in range(6):
            ctx[(qid, str(lv))] = build_context_string(
                build_designed_context(qtext, port, gold_id, ds, lv))
    print(f"컨텍스트 재구성 완료: {len(ctx)}개")

    def work(r):
        qtext, port, ref, gk, dk, _ = reg[r["question_id"]]
        res = judge(r["question"], r["answer"], ref, ctx[(r["question_id"], r["level"])])
        hg, hd = keyword_check(r["answer"], gk, dk)
        o = dict(r)
        o["strict_label"] = res["label"]
        o["strict_reason"] = res["reason"]
        o["정답값_포함"] = hg
        o["방해값_포함"] = hd
        # 문자열 검사 기준의 객관 판정 (참고용): 정답값 없고 방해값 있으면 명백한 오답
        o["kw_verdict"] = ("오답의심" if (not hg and hd) else
                           "정답값있음" if hg else "판단보류")
        return o

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 400 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}]")

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"저장: {args.output}")

    prev_col = "judge_label_4way"
    print("\n=== 기존 판정 → 엄격 판정 변화 (L1~L4) ===")
    sub = [r for r in out if r["level"] in "1234"]
    ch = defaultdict(Counter)
    for r in sub:
        ch[r[prev_col]][r["strict_label"]] += 1
    for a in ("정답", "근거초과", "기권", "오답"):
        if sum(ch[a].values()) == 0:
            continue
        print(f"  {a:5s}({sum(ch[a].values()):4d}) → " +
              "  ".join(f"{k} {v}" for k, v in ch[a].most_common()))

    n = len(sub)
    old = Counter(r[prev_col] for r in sub)
    new = Counter(r["strict_label"] for r in sub)
    print(f"\n  오답률  기존 {old['오답']/n*100:.1f}%  →  엄격 {new['오답']/n*100:.1f}%")

    print("\n=== 엄격 판정 vs 문자열 검사 일치도 (L1~L4) ===")
    susp = [r for r in sub if r["kw_verdict"] == "오답의심"]
    agree = sum(1 for r in susp if r["strict_label"] == "오답")
    if susp:
        print(f"  정답값 없이 방해값만 있는 {len(susp)}건 중 엄격 판정도 오답: "
              f"{agree} ({agree/len(susp)*100:.1f}%)")
    miss = [r for r in susp if r["strict_label"] != "오답"]
    if miss:
        print(f"  여전히 오답이 아닌 것: {len(miss)}건 — 사람이 직접 확인 필요")
        c = Counter(r["question_id"] for r in miss)
        print("   ", dict(c.most_common(6)))


if __name__ == "__main__":
    main()
