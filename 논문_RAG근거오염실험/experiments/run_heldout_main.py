"""
run_heldout_main.py
held-out 13문항(run_heldout_experiment.py의 QUESTIONS)에 기존 본실험(main_v2)과
동일한 구조를 적용한다: 6개 오염 레벨(L0~L5) x P0/P1 x 30회 반복 = 4,680건.

13문항 전부 "공통" 항구 문서라, 오염(search_contaminated)의 의미가 원래 설계
("특정 항구 질문 + 다른 항구 문서 주입")와 다르다 — 여기서는 "전국 공통 규정
질문에 특정 항구의 사설 규정이 근거로 끼어드는" 형태가 된다. build_graduated_
context는 question_port="공통"을 넘기면 이 필터링을 그대로 처리한다
(port != "공통" and port != "공통" -> port != "공통" -> 모든 특정 항구 문서가
오염 후보가 됨). 이건 원래 논문 설계와 다른 오염 유형이므로 결과는 기존
main_v2 결과와 나란히 비교하되 별도로 취급해야 한다.

답변 생성과 판정(4단계)을 분리해서 순서대로 수행 (CLAUDE.md 원칙 유지).
"""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from llm import get_llm_answer
from search import build_graduated_context
from run_heldout_experiment import QUESTIONS, build_context_string, JUDGE_PROMPT_4WAY, VALID_LABELS

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
REPEATS = 30
LEVELS = range(6)
PROMPTS = ["P0", "P1"]


def generate(qid, level, prompt_code, context):
    question, port, gold_id, reference = QUESTIONS[qid]
    if prompt_code == "P0":
        result = get_llm_answer(question, context=context)
    else:  # P1
        result = get_llm_answer(question, context=context, abstain_instruction=True)
    return {
        "question_id": qid,
        "question": question,
        "port": port,
        "level": level,
        "prompt": prompt_code,
        "answer": result.get("answer", ""),
    }


def judge_4way(question, answer, reference, full_context):
    user_content = (
        f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n"
        f"[모델이 실제로 참고한 근거 전체]\n{full_context}\n\n[모델 답변]\n{answer}"
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
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
    total = len(QUESTIONS) * len(list(LEVELS)) * len(PROMPTS) * REPEATS
    print(f"held-out 본실험: {len(QUESTIONS)}문항 x {len(list(LEVELS))}레벨 x {len(PROMPTS)}프롬프트 x "
          f"{REPEATS}회 = {total}건 생성 예정")

    # 질문 x 레벨 조합마다 근거를 한 번만 계산해서 P0/P1이 공유 (기존 실험과 동일 원칙)
    context_cache = {}
    for qid, (question, port, gold_id, _) in QUESTIONS.items():
        for level in LEVELS:
            chunks = build_graduated_context(question, question_port=port, gold_chunk_id=gold_id, level=level)
            context_cache[(qid, level)] = build_context_string(chunks)
    print(f"컨텍스트 캐시 구성 완료: {len(context_cache)}개 (문항 x 레벨)")

    specs = []
    for qid in QUESTIONS:
        for level in LEVELS:
            for prompt_code in PROMPTS:
                for _ in range(REPEATS):
                    specs.append((qid, level, prompt_code))

    generated = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(generate, qid, level, p, context_cache[(qid, level)])
            for qid, level, p in specs
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            generated.append(future.result())
            if i % 200 == 0 or i == len(specs):
                print(f"  [생성 {i}/{len(specs)}]")

    with open("results/heldout_main.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["question_id", "question", "port", "level", "prompt", "answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generated)
    print("생성 결과 저장: results/heldout_main.csv")

    def do_judge(row):
        qid = row["question_id"]
        _, _, _, reference = QUESTIONS[qid]
        full_context = context_cache[(qid, row["level"])]
        result = judge_4way(row["question"], row["answer"], reference, full_context)
        out = dict(row)
        out["judge_label_4way"] = result["label"]
        out["judge_reason_4way"] = result["reason"]
        return out

    judged = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(do_judge, r) for r in generated]
        for i, future in enumerate(as_completed(futures), start=1):
            judged.append(future.result())
            if i % 200 == 0 or i == len(generated):
                print(f"  [채점 {i}/{len(generated)}]")

    with open("results/heldout_main_judged.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(judged[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judged)
    print("채점 결과 저장: results/heldout_main_judged.csv")

    print("\n=== 레벨별 요약 (13문항 합산, 390건씩) ===")
    by_level_prompt = defaultdict(Counter)
    for r in judged:
        by_level_prompt[(r["level"], r["prompt"])][r["judge_label_4way"]] += 1
    for (level, prompt), counts in sorted(by_level_prompt.items(), key=lambda x: (x[0][0], x[0][1])):
        total_n = sum(counts.values())
        print(f"L{level} {prompt}: 총{total_n}  " +
              "  ".join(f"{k}={v}({v/total_n*100:.1f}%)" for k, v in counts.items()))

    print("\n=== 전체 요약 (프롬프트별, 6레벨 합산) ===")
    by_prompt = defaultdict(Counter)
    for r in judged:
        by_prompt[r["prompt"]][r["judge_label_4way"]] += 1
    for prompt, counts in sorted(by_prompt.items()):
        total_n = sum(counts.values())
        print(f"{prompt}: 총{total_n}  " +
              "  ".join(f"{k}={v}({v/total_n*100:.1f}%)" for k, v in counts.items()))


if __name__ == "__main__":
    main()
