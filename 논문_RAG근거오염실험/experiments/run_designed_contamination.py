"""
run_designed_contamination.py
"설계된 오염" 실험 — 기존 본실험과 모든 것이 동일하고, 오염 근거를 어디서
가져오는지만 다르다.

  기존(자연): search_contaminated() 가 다른 항구 문서 중 질문과 비슷한 것을 자동 검색
  이번(설계): distractor_chunk_ids.csv 에 미리 지정해둔 "같은 문서의 헷갈리는 조항"

레벨 구조는 build_graduated_context() 와 동일:
  L0~L4 : 정답 근거를 1칸에 고정 + 오염 level개 + 정상 (4-level)개
  L5    : 정답 근거 없이 5칸 전부 오염
오염은 누적(L1 ⊂ L2 ⊂ L3 ⊂ L4 ⊂ L5) — 강한 방해 근거부터 앞에서 잘라 쓴다.

측정은 두 가지를 병행한다.
  1) 4단계 채점(정답/근거초과/기권/오답) — LLM 판정
  2) 충돌 키워드 문자열 검사 — 방해 근거의 고유 숫자가 답변에 들어갔는지.
     방해 근거를 우리가 직접 지정했으므로 가능한 객관 지표이며,
     LLM 판정에 의존하지 않는다.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from llm import get_llm_answer
from search import get_chunk_by_id, search_normal

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
VALID_LABELS = {"정답", "근거초과", "기권", "오답"}
LEVELS = range(6)
PROMPTS = ["P0", "P1"]

JUDGE_PROMPT_4WAY = (
    "당신은 항만 안전 분야 RAG 시스템의 답변을 채점하는 평가자입니다. "
    "[질문], [정답 근거](핵심만 요약된 문구), [모델이 실제로 참고한 근거 전체], "
    "[모델 답변]을 보고 다음 기준으로 하나만 판정하세요.\n"
    "1. 모델 답변이 [정답 근거]의 핵심 내용(필수 항목)을 포함하는가?\n"
    "   포함하지 않거나 정답 근거와 사실적으로 모순되면 -> 오답\n"
    "2. 필수 항목을 포함한다면, 부가 설명이 있는지 본다.\n"
    "   부가 설명이 없으면, 또는 있어도 그 내용이 [모델이 실제로 참고한 근거 전체] 안에 "
    "있으면 -> 정답\n"
    "   부가 설명이 [모델이 실제로 참고한 근거 전체]에도 없지만, 정답 근거나 근거 전체와 "
    "모순되지는 않으면 -> 근거초과\n"
    "3. 모델이 자료에서 확인할 수 없다는 취지로 답을 회피했으면 -> 기권\n"
    '반드시 JSON 객체 {"label": "정답|근거초과|기권|오답", "reason": "..."} 로만 응답하세요.'
)

# question_id -> (질문, 항구, 정답요약, 정답키워드, 1순위 방해근거의 충돌키워드)
QUESTIONS = {
    "Q1": ("항만 하역장비 크레인의 주요 구조물에는 두께 몇 ㎜ 이하의 철판이나 강재 단면을 사용할 수 없나요?",
           "공통", "두께 6㎜ 이하인 철판이나 강재 단면은 사용할 수 없다.", ["6㎜", "6mm", "6 ㎜"], ["25㎜", "25mm"]),
    "Q2": ("항만 하역장비 크레인의 와이어로프 쉬브 그루브는 표면에서 최소 몇 ㎜ 깊이까지 어느 정도의 경도를 가져야 하나요?",
           "공통", "표면에서 최소 3㎜ 이상의 깊이까지 Hs 55～70의 경도를 가져야 한다.", ["3㎜", "3mm"], ["10㎜", "10mm", "46", "52"]),
    "Q3": ("항만 하역장비 크레인 계단의 폭은 몇 ㎜ 이상이어야 하나요?",
           "공통", "계단의 폭은 560㎜ 이상이어야 한다.", ["560"], ["400", "600", "460"]),
    "Q4": ("레일검사기준에서 레일간격(Span)의 편차는 길이 5미터 내지 10미터마다 몇 밀리미터 이내여야 하나요?",
           "공통", "±3밀리미터 이내여야 한다.", ["3밀리", "±3", "3mm", "3 밀리"], ["10밀리", "±10", "10mm"]),
    "Q5": ("레일검사기준에서 레일의 횡단면 경사도는 얼마 이내여야 하나요?",
           "공통", "1/400 이내여야 한다.", ["1/400"], ["0.3퍼센트", "0.3%"]),
    "Q6": ("무역항 등의 항만시설사용료 중 정박료에서 외항선의 기본료(10톤·12시간당)는 얼마인가요?",
           "공통", "187원이다.", ["187"], ["79.1"]),
    "Q7": ("무역항 등의 항만시설사용료 중 계선료에서 외항선의 요율(10톤·12시간당)은 얼마인가요?",
           "공통", "28.5원이다.", ["28.5"], ["13.1"]),
    "Q8": ("일점계류장치(SPM) 정기검사에서 선체 두께계측을 시행해야 하는 것은 사용 몇 년 이상의 SPM인가요?",
           "공통", "사용 15년 이상의 SPM이다.", ["15년"], ["5년"]),
    "Q9": ("일점계류장치(SPM)의 정기검사를 정기검사 지정일로부터 얼마 이상 앞당겨 받은 경우에 차기 정기검사 시기가 재지정되나요?",
           "공통", "3월(3개월) 이상 앞당겨 받은 경우이다.", ["3월", "3개월"], ["15월", "15개월"]),
    "Q10": ("울산항 일반화물 안전매뉴얼의 슬라브 작업에서, 육상으로 넘어온 화물이 지상 몇 미터 높이로 내려올 때 작업자가 접근할 수 있나요?",
            "울산항", "안전높이인 지상 1미터로 내려올 때 접근한다.", ["1미터", "1 미터", "1m", "1M"], ["2미터", "2M", "2m"]),
    "Q12": ("항만하역 통합안전매뉴얼에 따르면 창고 내 화물적재 작업 시 적재높이는 약 몇 미터 정도로 해야 하나요?",
            "공통", "약 2미터 정도로 한다.", ["2미터"], ["3m", "3미터", "최고 적재"]),
}


def load_distractors(path="distractor_chunk_ids.csv"):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["question_id"]] = (r["gold_chunk_id"],
                                 [r[f"d{i}"] for i in range(1, 6)])
    return out


def build_context_string(chunks):
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def build_designed_context(question, port, gold_id, distractors, level):
    """기존 build_graduated_context 와 같은 구조, 오염 출처만 설계된 것으로 교체."""
    if level == 5:
        return [get_chunk_by_id(cid) for cid in distractors[:5]]
    gold = get_chunk_by_id(gold_id)
    contamination = [get_chunk_by_id(cid) for cid in distractors[:level]]
    needed_normal = 4 - level
    comparison = []
    if needed_normal > 0:
        exclude = {gold_id} | set(distractors)
        cands = search_normal(question, top_k=needed_normal + 6, port=port)
        comparison = [c for c in cands if c["chunk_id"] not in exclude][:needed_normal]
    return [gold] + comparison + contamination


def judge_4way(question, answer, reference, full_context):
    user_content = (f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n"
                    f"[모델이 실제로 참고한 근거 전체]\n{full_context}\n\n[모델 답변]\n{answer}")
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_PROMPT_4WAY},
                      {"role": "user", "content": user_content}])
        parsed = json.loads(resp.choices[0].message.content)
        label = str(parsed.get("label", "")).strip()
        if label not in VALID_LABELS:
            return {"label": "판정불가", "reason": f"유효하지 않은 라벨: {parsed}"}
        return {"label": label, "reason": str(parsed.get("reason", "")).strip()}
    except Exception as e:
        return {"label": "판정불가", "reason": f"오류: {e}"}


def _norm(t):
    return re.sub(r"\s+", "", t)


def keyword_check(answer, gold_kws, distractor_kws):
    """정답 값 / 방해 값이 답변에 들어갔는지 (LLM 판정과 독립적인 객관 지표)"""
    a = _norm(answer)
    return (any(_norm(k) in a for k in gold_kws),
            any(_norm(k) in a for k in distractor_kws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dmap = load_distractors()
    qids = [q for q in QUESTIONS if q in dmap]
    total = len(qids) * len(list(LEVELS)) * len(PROMPTS) * args.repeats
    print(f"설계오염 실험: {len(qids)}문항 x 6레벨 x 2프롬프트 x {args.repeats}회 = {total}건")

    ctx = {}
    for qid in qids:
        qtext, port, _, _, _ = QUESTIONS[qid]
        gold_id, ds = dmap[qid]
        for lv in LEVELS:
            chunks = build_designed_context(qtext, port, gold_id, ds, lv)
            ctx[(qid, lv)] = build_context_string(chunks)
    print(f"컨텍스트 구성 완료: {len(ctx)}개 (문항 x 레벨)")

    specs = [(q, lv, p) for q in qids for lv in LEVELS for p in PROMPTS
             for _ in range(args.repeats)]

    def gen(qid, lv, p):
        qtext, port, _, _, _ = QUESTIONS[qid]
        r = get_llm_answer(qtext, context=ctx[(qid, lv)], abstain_instruction=(p == "P1"))
        return {"question_id": qid, "question": qtext, "port": port,
                "level": lv, "prompt": p, "answer": r.get("answer", "")}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen, *s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 200 == 0 or i == len(specs):
                print(f"  [생성 {i}/{len(specs)}]")

    with open("results/designed_contamination.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question", "port", "level", "prompt", "answer"])
        w.writeheader(); w.writerows(rows)
    print("생성 저장: results/designed_contamination.csv")

    def do_judge(r):
        _, _, ref, gold_kw, dist_kw = QUESTIONS[r["question_id"]]
        res = judge_4way(r["question"], r["answer"], ref, ctx[(r["question_id"], r["level"])])
        has_gold, has_dist = keyword_check(r["answer"], gold_kw, dist_kw)
        o = dict(r)
        o["judge_label_4way"] = res["label"]
        o["judge_reason_4way"] = res["reason"]
        o["정답값_포함"] = has_gold
        o["방해값_포함"] = has_dist
        return o

    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(do_judge, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            judged.append(f.result())
            if i % 200 == 0 or i == len(rows):
                print(f"  [채점 {i}/{len(rows)}]")

    with open("results/designed_contamination_judged.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(judged[0].keys()))
        w.writeheader(); w.writerows(judged)
    print("채점 저장: results/designed_contamination_judged.csv")

    print("\n=== 레벨별 (11문항 합산) ===")
    by = defaultdict(Counter)
    kw = defaultdict(lambda: [0, 0])
    for r in judged:
        by[(r["level"], r["prompt"])][r["judge_label_4way"]] += 1
        k = kw[(r["level"], r["prompt"])]
        k[0] += 1
        if r["방해값_포함"]:
            k[1] += 1
    for key in sorted(by, key=lambda x: (int(x[0]), x[1])):
        lv, p = key
        c = by[key]; n = sum(c.values())
        tot, dhit = kw[key]
        print(f"L{lv} {p}: 총{n}  " + "  ".join(f"{k}={v}({v/n*100:.1f}%)" for k, v in c.items())
              + f"   | 방해값 인용 {dhit}/{tot} ({dhit/tot*100:.1f}%)")


if __name__ == "__main__":
    main()
