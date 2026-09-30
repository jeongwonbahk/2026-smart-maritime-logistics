"""
judge.py
LLM-as-a-Judge로 experiment.py가 생성한 답변을 정답/기권/오답 3분류로 판정한다.

중요 원칙 (CLAUDE.md): "답변 생성"과 "판정"을 완전히 분리한다.
이 파일은 답변 생성에 절대 관여하지 않고, 이미 experiment.py가 만들어둔
답변만 채점한다. 정답 근거(gold answer)는 별도 CSV로 관리하며, 실험 실행
(llm.py 호출) 시에는 절대 프롬프트에 노출되지 않는다.

사용법:
    python judge.py --input results/experiment_xxx.csv --gold gold_answers.csv

gold_answers.csv 형식 (question_id별 정답 근거):
    question_id,reference_answer
    q1,"최초 정기안전점검은 준공일로부터 1년 이내, 이후 매년 1회 이상 실시"
    q6,                       <- 원래 문서에 답이 없는(함정) 질문은 비워둠
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
# 판정은 반복 시 흔들리면 안 되므로 0으로 고정
# (llm.py의 temperature=0.7은 "답변 생성"의 반복 실험 변동성용 설정으로, 판정과는 목적이 다름)
TEMPERATURE = 0

VALID_LABELS = {"정답", "기권", "오답"}

JUDGE_SYSTEM_PROMPT = (
    "당신은 항만 안전 분야 RAG 시스템의 답변을 채점하는 평가자입니다. "
    "주어진 [질문], [정답 근거], [모델 답변]을 보고 반드시 다음 세 가지 중 "
    "하나로만 분류하세요.\n"
    "- 정답: 모델 답변이 정답 근거의 핵심 내용과 사실적으로 일치한다.\n"
    "- 기권: 모델이 자료에서 확인할 수 없다는 취지로 답을 회피했다.\n"
    "- 오답: 모델이 확정적으로 답했지만 정답 근거와 다르거나, 근거에 없는 "
    "내용을 사실처럼 단정했다.\n"
    '반드시 JSON 객체 {"label": "정답|기권|오답", "reason": "..."} 로만 응답하세요.'
)

# 원래 문서에 답이 존재하지 않는 질문("함정 질문")용 — '정답'이라는 선택지 자체가 성립하지 않음
JUDGE_SYSTEM_PROMPT_NO_REFERENCE = (
    "당신은 항만 안전 분야 RAG 시스템의 답변을 채점하는 평가자입니다. "
    "이 질문은 문서에 실제로는 답이 존재하지 않는 함정 질문입니다. "
    "주어진 [질문]과 [모델 답변]을 보고 다음 두 가지 중 하나로만 분류하세요 "
    "('정답'은 원래 존재할 수 없으므로 선택지에서 제외합니다).\n"
    "- 기권: 모델이 자료에서 확인할 수 없다는 취지로 답을 회피했다.\n"
    "- 오답: 모델이 근거 없이 확정적인 내용을 사실처럼 단정해서 답했다.\n"
    '반드시 JSON 객체 {"label": "기권|오답", "reason": "..."} 로만 응답하세요.'
)


def judge_answer(question, answer, reference=None):
    """
    질문·모델 답변(·있으면 정답 근거)을 보고 정답/기권/오답으로 분류한다.
    reference가 없으면(원래 답이 없는 함정 질문) 기권/오답 중에서만 판정한다.
    반환: {"label": "정답"|"기권"|"오답"|"판정불가", "reason": str}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"label": "판정불가", "reason": "OPENAI_API_KEY가 설정되지 않았습니다."}

    if reference:
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_content = f"[질문]\n{question}\n\n[정답 근거]\n{reference}\n\n[모델 답변]\n{answer}"
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT_NO_REFERENCE
        user_content = f"[질문]\n{question}\n\n[모델 답변]\n{answer}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        label = str(parsed.get("label", "")).strip()
        reason = str(parsed.get("reason", "")).strip()

        # reference가 없는데 '정답'이 나오는 등 있을 수 없는 라벨은 신뢰하지 않음
        if label not in VALID_LABELS or (reference is None and label == "정답"):
            return {"label": "판정불가", "reason": f"판정 결과가 유효하지 않음: {parsed}"}

        return {"label": label, "reason": reason}

    except Exception as e:
        # API 오류 등으로 판정에 실패해도 파이프라인이 죽지 않게 함
        return {"label": "판정불가", "reason": f"판정 중 오류 발생: {e}"}


def load_gold_answers(path):
    """question_id -> reference_answer 매핑을 로드.
    파일/행이 없거나 값이 비어있으면 '원래 답이 없는 질문'(None)으로 취급."""
    gold = {}
    if not path or not os.path.exists(path):
        return gold
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold[row["question_id"]] = row.get("reference_answer", "").strip() or None
    return gold


def main():
    parser = argparse.ArgumentParser(description="experiment.py 결과를 정답/기권/오답으로 판정")
    parser.add_argument("--input", required=True, help="experiment.py가 생성한 결과 CSV")
    parser.add_argument("--gold", default="gold_answers.csv", help="question_id별 정답 근거 CSV")
    parser.add_argument("--output", default=None, help="판정 결과 저장 경로 (기본: <input>_judged.csv)")
    parser.add_argument("--workers", type=int, default=8, help="동시 판정 개수")
    args = parser.parse_args()

    gold = load_gold_answers(args.gold)
    if not gold:
        print(f"[경고] '{args.gold}'에서 정답 근거를 찾지 못함 "
              f"— 모든 질문을 '원래 답이 없는 질문'으로 취급해 기권/오답으로만 판정합니다.")

    output_path = args.output or args.input.replace(".csv", "_judged.csv")

    with open(args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[중단] '{args.input}'에 판정할 행이 없습니다.")
        return

    fieldnames = list(rows[0].keys()) + ["judge_label", "judge_reason"]

    def judge_row(row):
        reference = gold.get(row["question_id"])
        result = judge_answer(row["question"], row["answer"], reference=reference)
        row = dict(row)
        row["judge_label"] = result["label"]
        row["judge_reason"] = result["reason"]
        return row

    done = 0
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(judge_row, row) for row in rows]
            for future in as_completed(futures):
                writer.writerow(future.result())
                f.flush()
                done += 1
                if done % 10 == 0 or done == len(rows):
                    print(f"  [{done}/{len(rows)}] 판정 완료")

    print(f"\n완료: {done}건 판정 → {output_path}")


if __name__ == "__main__":
    main()
