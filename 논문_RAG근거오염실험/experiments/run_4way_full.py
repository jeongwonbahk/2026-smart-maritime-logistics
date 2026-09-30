"""
run_4way_full.py
test_4way_judge.py에서 검증한 4단계 채점 기준(정답/근거초과/기권/오답)을
q3·q8·q10 세 문항이 아니라 main_v2.csv 전체(11문항 x 6레벨 x 2프롬프트 x 30회
= 3,960건)에 적용한다.

함정 질문(trap_v2.csv)은 애초에 정답 근거 자체가 없어 "근거초과" 개념이 성립하지
않으므로(기권/오답 2분류) 이 스크립트의 대상에서 제외 — trap_v2_judged.csv 그대로 둔다.

기존 답변(main_v2.csv)을 재활용 — 새로 답변 생성 안 함, 재채점만 한다.
(CLAUDE.md 원칙: 답변 생성과 판정을 분리)
"""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from search import build_graduated_context

load_dotenv()
client = OpenAI()

MODEL_NAME = "gpt-4o-mini"
VALID_LABELS = {"정답", "근거초과", "기권", "오답"}

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


def load_questions():
    """main_v2.csv에서 question_id -> (question_text, port) 를 뽑고
    gold_chunk_ids.csv / gold_answers_test.csv를 합쳐 QUESTIONS 딕셔너리를 만든다."""
    rows = list(csv.DictReader(open("results/main_v2.csv", encoding="utf-8-sig")))
    qtext = {}
    for r in rows:
        qtext.setdefault(r["question_id"], (r["question"], r["port"]))

    gold_chunks = {}
    for r in csv.DictReader(open("gold_chunk_ids.csv", encoding="utf-8-sig")):
        gold_chunks[r["question_id"]] = r["gold_chunk_id"]

    references = {}
    for r in csv.DictReader(open("gold_answers_test.csv", encoding="utf-8-sig")):
        references[r["question_id"]] = r["reference_answer"]

    questions = {}
    for qid, (qtxt, port) in qtext.items():
        if qid not in gold_chunks or not references.get(qid):
            continue  # 함정 질문 등 정답 근거가 없는 문항은 제외
        questions[qid] = (qtxt, port, gold_chunks[qid], references[qid])
    return questions


def load_question_types(path="question_types.csv"):
    types = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        types[r["question_id"]] = r["question_type"]
    return types


def build_context_string(chunks):
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def judge_4way(question, answer, reference, full_context):
    user_content = (
        f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n"
        f"[모델이 실제로 참고한 근거 전체]\n{full_context}\n\n[모델 답변]\n{answer}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT_4WAY},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        label = str(parsed.get("label", "")).strip()
        if label not in VALID_LABELS:
            return {"label": "판정불가", "reason": f"유효하지 않은 라벨: {parsed}"}
        return {"label": label, "reason": str(parsed.get("reason", "")).strip()}
    except Exception as e:
        return {"label": "판정불가", "reason": f"오류: {e}"}


def main():
    questions = load_questions()
    qtypes = load_question_types()
    print(f"대상 문항: {len(questions)}개 -> {sorted(questions.keys())}")

    rows = list(csv.DictReader(open("results/main_v2.csv", encoding="utf-8-sig")))
    target = [r for r in rows if r["question_id"] in questions]
    print(f"재채점 대상: {len(target)}건 (전체 main_v2.csv 중 정답 근거 있는 문항만)")

    # 질문×레벨 조합마다 근거 전체를 한 번만 재구성 (캐시)
    context_cache = {}
    for qid, (question, port, gold_id, _) in questions.items():
        for level in range(6):
            chunks = build_graduated_context(question, question_port=port, gold_chunk_id=gold_id, level=level)
            context_cache[(qid, str(level))] = build_context_string(chunks)
    print(f"근거 캐시 구성 완료: {len(context_cache)}개 (문항 x 레벨)")

    def process(row):
        qid = row["question_id"]
        question, port, gold_id, reference = questions[qid]
        full_context = context_cache[(qid, row["level"])]
        result = judge_4way(row["question"], row["answer"], reference, full_context)
        out = dict(row)
        out["judge_label_4way"] = result["label"]
        out["judge_reason_4way"] = result["reason"]
        out["question_type"] = qtypes.get(qid, "")
        return out

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process, r) for r in target]
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i % 200 == 0 or i == len(target):
                print(f"  [{i}/{len(target)}]")

    fieldnames = list(results[0].keys())
    with open("results/main_v2_judged_4way.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # --- 요약 1: 문항 x 프롬프트 (전 레벨 합산) ---
    print("\n=== 문항별 요약 (전 레벨 합산) ===")
    by_qp = defaultdict(Counter)
    for r in results:
        by_qp[(r["question_id"], r["prompt"])][r["judge_label_4way"]] += 1
    for (qid, prompt), counts in sorted(by_qp.items()):
        total = sum(counts.values())
        print(f"{qid} {prompt}: 총{total}  " + "  ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in counts.items()))

    # --- 요약 2: 문항유형 x 레벨 x 프롬프트 ---
    print("\n=== 문항유형별 레벨별 요약 ===")
    by_type_level = defaultdict(Counter)
    for r in results:
        key = (r["question_type"], r["level"], r["prompt"])
        by_type_level[key][r["judge_label_4way"]] += 1
    for (qtype, level, prompt), counts in sorted(by_type_level.items(), key=lambda x: (x[0][0], int(x[0][1]), x[0][2])):
        total = sum(counts.values())
        print(f"{qtype} L{level} {prompt}: 총{total}  " + "  ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in counts.items()))

    # --- 요약 3: 전체 오답률 대조 (기존 3분류 vs 4단계) ---
    print("\n=== 전체 프롬프트별 요약 (전 레벨 합산, 전 문항) ===")
    by_prompt = defaultdict(Counter)
    for r in results:
        by_prompt[r["prompt"]][r["judge_label_4way"]] += 1
    for prompt, counts in sorted(by_prompt.items()):
        total = sum(counts.values())
        print(f"{prompt}: 총{total}  " + "  ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in counts.items()))


if __name__ == "__main__":
    main()
