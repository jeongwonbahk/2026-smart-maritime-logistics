import os

from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 환경 변수(OPENAI_API_KEY)를 읽어온다
load_dotenv()

# 스마트항만 전문가 역할을 부여하는 시스템 프롬프트
SYSTEM_PROMPT = (
    "당신은 스마트항만 작업 절차, 안전 규정, 장비 운용에 대해 답변하는 "
    "전문가입니다. 한국어로 간결하고 정확하게 답변하세요."
)

# 비용 절감을 위해 gpt-4o-mini 모델 사용
MODEL_NAME = "gpt-4o-mini"

# 반복 실험의 재현성을 위해 고정 (변경 금지 — CLAUDE.md 참고)
TEMPERATURE = 0.7

# 기권 유도 지시문 (실험 조건 C, C_dirty에서 시스템 프롬프트에 추가됨)
ABSTAIN_INSTRUCTION = (
    "단, 아래 [참고 자료]에 질문에 대한 근거가 없으면 절대 추측하지 말고 "
    "\"제공된 자료에서 확인할 수 없습니다.\"라고만 답하세요. "
    "자료에 없는 내용을 지어내거나 일반 상식으로 답을 채우지 마세요."
)


def get_llm_answer(question, context=None, abstain_instruction=False, custom_instruction=None):
    """
    스마트항만 관련 질문을 받아 LLM 답변을 생성해 딕셔너리로 반환한다.

    - context: RAG로 검색한 근거 문서 텍스트(선택). 주어지면 프롬프트에 포함된다.
    - abstain_instruction: True면 시스템 프롬프트에 기본 기권 유도 지시문(P1)을 추가한다.
    - custom_instruction: 문자열을 주면 그 문구를 시스템 프롬프트에 대신 추가한다
      (P2 — 실패 사례 기반 자동개선 지시문용. abstain_instruction보다 우선함).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # API 키가 설정되어 있지 않은 경우 에러 메시지를 answer에 담아 반환
    if not api_key:
        return {
            "question": question,
            "answer": "오류: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.",
        }

    system_prompt = SYSTEM_PROMPT
    if custom_instruction:
        system_prompt = f"{system_prompt}\n{custom_instruction}"
    elif abstain_instruction:
        system_prompt = f"{system_prompt}\n{ABSTAIN_INSTRUCTION}"

    if context:
        user_content = f"[참고 자료]\n{context}\n\n[질문]\n{question}"
    else:
        user_content = question

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
        }

    except Exception as e:
        # API 키 오류, 네트워크 오류 등 어떤 예외가 발생해도
        # 프로그램이 죽지 않고 에러 메시지를 answer에 담아 반환한다
        return {
            "question": question,
            "answer": f"오류: LLM 응답 생성 중 문제가 발생했습니다. ({e})",
        }


if __name__ == "__main__":
    test_question = "선박 접안 시 가장 먼저 해야 하는 작업은?"
    result = get_llm_answer(test_question)
    print(result)
