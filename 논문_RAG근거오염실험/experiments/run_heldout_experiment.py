"""
run_heldout_experiment.py
사용자가 새로 만들어온 15문항(전부 "공통" 항구 문서, q3·q8·q10 튜닝에는 전혀
쓰이지 않은 held-out 질문) 중 13개를 검증해 실제 실험에 쓴다.

제외한 2개:
  - "하역장치 시험하중 SWL 구간별 산정" — 표 9.2.2의 숫자 공식(1.25×SWL 등)이
    PDF -> 텍스트 변환 과정에서 소실되어 코퍼스 어디에도 남아있지 않음
    (기존 q2가 겪었던 것과 같은 문제) -> 검증 불가로 제외.
  - "인천항 재난대응 4단계" — 기존 q10과 완전히 동일한 질문/답 -> 중복 제외.

오염 실험 설계 원칙(CLAUDE.md, 항만_질문_초안.md): "공통 문서는 오염 대상이
아니다" -> 이 13문항은 전부 공통 문서라 그라데이션 오염(L0~L5) 실험 대상에는
안 맞음. 대신 L0(오염 없음, gold 근거 고정)만으로 "새 질문에서도 4단계 채점과
프롬프트 수정(부가설명 금지)이 잘 작동하는가"를 확인하는 held-out 검증 실험.

프롬프트 3종:
  P0      : 기본 지시문 없음
  P1      : 기권 유도 지시문 (기존 실험과 동일)
  P_NOELAB: "부가설명 금지" 지시문 (q10에서만 검증됐던 것 — 여기서 새 질문
            13개로 일반화되는지 확인. "출처 확인" 지시문은 이 13문항이 질문
            안에 특정 문서명을 지정하지 않아 적용 대상이 없으므로 제외)

각 조합 10회 반복. 답변 생성 후 4단계 채점(정답/근거초과/기권/오답)까지
한 스크립트에서 수행 — 답변 생성과 판정 로직은 분리해서 호출한다
(CLAUDE.md 원칙 유지: judge 함수는 질문+답변+근거만 보고 독립적으로 판정).
"""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from llm import get_llm_answer
from search import build_graduated_context

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
REPEATS = 10
VALID_LABELS = {"정답", "근거초과", "기권", "오답"}

NOELAB_INSTRUCTION = (
    "핵심 사실만 간결하게 답하세요. 정답에 필요하지 않은 부가 설명이나 배경 "
    "설명은 덧붙이지 마세요."
)

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

# question_id -> (question, port, gold_chunk_id, reference_answer)
QUESTIONS = {
    "h1": (
        "일점계류장치(SPM) 정기검사에서 5년 이상 사용한 화물 호스는 어떻게 검사해야 하나요?",
        "공통", "공통_계류지침_KR_일점계류장치지침2017_정제_c41",
        "SPM의 일부인 화물 호스로서 5년 이상 사용된 호스는 떼어내어 분해하여 검사하고 사용압력으로 "
        "압력시험을 합니다. 호스를 새로 교환했거나 새것으로 교체한 지 5년이 안 된 경우에는 사용 5년이 "
        "될 때까지 검사를 연기할 수 있습니다. 정기검사 때나 사용 5년 이상 된 화물호스에는 진공시험을 "
        "요구할 수 있습니다.",
    ),
    "h2": (
        "일점계류장치(SPM) 위치 도표에 표시하는 스윙서클의 반경은 어떻게 구하나요?",
        "공통", "공통_계류지침_KR_일점계류장치지침2017_정제_c74",
        "호저 하중과 최소조류의 운전조건에서, SPM 중심을 기준으로 SPM의 수평행정, 호저 하중하의 호저 "
        "길이의 수평 투영거리, SPM을 사용하려는 선박 중 가장 긴 선박의 전장(L.O.A)에 허용 안전길이 "
        "30m를 더한 길이입니다.",
    ),
    "h3": (
        "항만 하역장비 크레인의 주요 구조물에는 몇 mm 두께 이하의 철판이나 강재를 사용할 수 없나요?",
        "공통", "공통_안전_해양수산부_항만하역장비제작설치_정제_c10",
        "크레인 주요 구조물에는 두께가 6mm 이하인 철판이나 강재 단면을 사용할 수 없습니다.",
    ),
    "h4": (
        "항만시설물 중 특수구조와 일반구조 시설물의 정밀안전점검 실시 주기는 각각 어떻게 되나요?",
        "공통", "공통_안전규정_해양수산부_항만시설물안전점검지침_정제_c28",
        "특수구조(방파제, 파제제, 안벽·돌핀·소형선부두 등 여객 이용 계류시설)는 준공일 또는 사용승인일"
        "부터 6년 이내에 최초 정밀안전점검을 하고 이후 6년마다 1회 이상 실시합니다. 일반구조(방사제, "
        "방조제, 도류제, 호안, 도로, 교량, 안벽, 물양장, 잔교, 부잔교, 선착장 등)는 준공일 또는 "
        "사용승인일부터 10년 이내에 최초 정밀안전점검을 하고 이후 10년마다 1회 이상 실시합니다.",
    ),
    "h5": (
        "항만시설 사용료를 분할 납부하게 할 때, 어떤 경우에 보증금 예치나 이행보증조치를 해야 하나요?",
        "공통", "공통_정책고시_해양수산부_항만시설사용료 규정 일부개정고시안_정제_c2",
        "연간 사용료가 1천만원 이상인 경우입니다. 이때 항만관리청은 허가받은 자에게 연간 사용료의 "
        "100분의 50 범위에서 보증금을 예치하게 하거나 이행보증조치를 하도록 해야 합니다.",
    ),
    "h6": (
        "휴지상태(태풍)에서 목포 지역 시설장비의 최대순간풍속 기준은 얼마인가요?",
        "공통", "공통_정책_해양수산부_항만시설장비검사기준_정제_c5",
        "지면상에서 20미터 높이를 기준으로 초당 70미터 이상입니다.",
    ),
    "h7": (
        "항만관리청이 항만시설전용사용을 허가할 수 있는 기간은 최대 몇 년인가요?",
        "공통", "공통_정책고시_해양수산부_항만시설 사용 및 사용료에 관한 규정(2017)최종본_정제_c6",
        "별표 1 제1호라목 또는 제1-2호라목에서 정하는 항만시설을 전용사용하려는 자가 있으면 해당 "
        "항만시설의 공공성을 저해하지 않는 범위에서 5년 이내의 기간을 정하여 전용사용을 허가할 수 "
        "있습니다.",
    ),
    "h8": (
        "해양수산부장관이 사용자의 PORT-MIS 이용승인을 취소할 수 있는 경우는 어떤 것들이 있나요?",
        "공통", "공통_정책고시_해양수산부_항만물류통합정보체계 구축·운영 및 이용절차에 관한 규정_정제_c12",
        "신고자 요건 미충족 등에 대해 보완·수정을 통보했으나 기한 내 조치가 없는 경우, 또는 시정요구를 "
        "했으나 조치가 없어 시스템 운영에 지장을 줄 우려가 있는 경우입니다.",
    ),
    "h9": (
        "PORT-MIS 서비스 이용신청을 심사한 결과 신청인이 「항만법」 등 관련 법령에서 정하는 신고자 "
        "요건을 충족하지 못한 경우, 해양수산부장관은 어떻게 해야 하나요?",
        "공통", "공통_정책고시_해양수산부_항만물류통합정보체계 구축·운영 및 이용절차에 관한 규정_정제_c10",
        "해양수산부장관은 신청인, 업체코드, 업체종류, 신청 내용, 해운항만관련 사업 면허 또는 등록 내용, "
        "사업자등록 또는 주민등록 내용을 확인한 결과 요건을 충족하지 않으면 신청인에게 보완·수정할 "
        "것을 통보해야 합니다. 이용신청서의 기록 내용이 사실과 다른 경우에도 동일하게 통보해야 합니다.",
    ),
    "h10": (
        "PORT-MIS에 중대사고가 발생해 재해복구시스템으로도 전자민원업무 처리가 불가능한 경우, 항만운영"
        "에 지장을 주는 상황에서는 어떤 조치를 취할 수 있나요?",
        "공통", "공통_정책고시_해양수산부_항만물류통합정보체계 구축·운영 및 이용절차에 관한 규정_정제_c17",
        "해양수산부장관 및 항만관리청은 중대사고로 PORT-MIS 및 재해복구시스템에서 전자민원업무 처리가 "
        "불가능하여 항만운영에 지장을 준다고 판단되는 경우, 수작업처리 등 대체수단으로 민원처리를 할 "
        "수 있습니다.",
    ),
    "h11": (
        '이 개정고시에서 "해난"이라는 용어가 "해양사고"로 변경되면서, 해양사고의 정의는 어느 법령의 어느'
        " 조항을 따르게 되나요?",
        "공통", "공통_정책고시_해양수산부_항만시설사용료 규정 일부개정고시안_정제_c4",
        "\"관공선, 해난을\"이 \"관공선, 「해양사고의 조사 및 심판에 관한 법률」 제2조제1호에 따른 "
        "해양사고(이하 '해양사고'라 한다)를\"로 개정되어, 해양사고의 정의는 「해양사고의 조사 및 심판에 "
        "관한 법률」 제2조제1호를 따르게 됩니다.",
    ),
    "h12": (
        "크레인 계단의 높이가 10m를 초과할 경우, 몇 m 이내마다 플랫폼을 설치해야 하나요?",
        "공통", "공통_안전_해양수산부_항만하역장비제작설치_정제_c15",
        "계단의 높이가 10m를 초과할 경우에는 7m 이내마다 플랫폼이 설치되어야 합니다. 계단의 경사 각도는"
        " 수평에 대해 50도를 초과하지 않아야 하며, 발판높이는 300mm 이내의 같은 간격으로 하고, 계단의 "
        "폭은 560mm 이상이 되어야 합니다.",
    ),
    "h13": (
        "주권상 장치의 각 캘리퍼 디스크 브레이크는 정격하중을 권상할 때 요구되는 토크의 몇 % 이상의 "
        "동적 용량을 지녀야 하나요?",
        "공통", "공통_안전_해양수산부_항만하역장비제작설치_정제_c25",
        "주권상 장치에는 스러스트로 작동하는 2개의 캘리퍼 디스크 브레이크가 설치되어야 하며, 각 "
        "브레이크는 정격하중을 권상할 때 브레이크가 설치된 축에 요구되는 토크의 최소 100% 이상(2개 "
        "합은 200% 이상)과 같은 동적인 용량을 지녀야 합니다.",
    ),
}

PROMPTS = ["P0", "P1", "P_NOELAB"]


def build_context_string(chunks):
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def generate(qid, prompt_code, context):
    question, port, gold_id, reference = QUESTIONS[qid]
    if prompt_code == "P0":
        result = get_llm_answer(question, context=context)
    elif prompt_code == "P1":
        result = get_llm_answer(question, context=context, abstain_instruction=True)
    else:  # P_NOELAB
        result = get_llm_answer(question, context=context, custom_instruction=NOELAB_INSTRUCTION)
    return {
        "question_id": qid,
        "question": question,
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
    print(f"held-out 질문 {len(QUESTIONS)}개 x 프롬프트 {len(PROMPTS)}종 x {REPEATS}회 "
          f"= {len(QUESTIONS) * len(PROMPTS) * REPEATS}건 생성 예정")

    # 질문당 컨텍스트를 한 번만 계산해서 모든 프롬프트가 공유 (기존 실험과 동일한 원칙)
    context_cache = {}
    for qid, (question, port, gold_id, _) in QUESTIONS.items():
        chunks = build_graduated_context(question, question_port=port, gold_chunk_id=gold_id, level=0)
        context_cache[qid] = build_context_string(chunks)
    print("컨텍스트 캐시 구성 완료")

    # --- 생성 ---
    specs = []
    for qid in QUESTIONS:
        for prompt_code in PROMPTS:
            for _ in range(REPEATS):
                specs.append((qid, prompt_code))

    generated = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate, qid, p, context_cache[qid]) for qid, p in specs]
        for i, future in enumerate(as_completed(futures), start=1):
            generated.append(future.result())
            if i % 50 == 0 or i == len(specs):
                print(f"  [생성 {i}/{len(specs)}]")

    with open("results/heldout_v1.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(generated)

    # --- 채점 (4단계, 답변 생성과 완전히 분리된 별도 단계) ---
    def do_judge(row):
        qid = row["question_id"]
        _, _, _, reference = QUESTIONS[qid]
        result = judge_4way(row["question"], row["answer"], reference, context_cache[qid])
        out = dict(row)
        out["judge_label_4way"] = result["label"]
        out["judge_reason_4way"] = result["reason"]
        return out

    judged = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(do_judge, r) for r in generated]
        for i, future in enumerate(as_completed(futures), start=1):
            judged.append(future.result())
            if i % 50 == 0 or i == len(generated):
                print(f"  [채점 {i}/{len(generated)}]")

    with open("results/heldout_v1_judged.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(judged[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judged)

    # --- 요약 ---
    print("\n=== 문항별 요약 ===")
    by_qp = defaultdict(Counter)
    for r in judged:
        by_qp[(r["question_id"], r["prompt"])][r["judge_label_4way"]] += 1
    for (qid, prompt), counts in sorted(by_qp.items(), key=lambda x: (x[0][0], x[0][1])):
        total = sum(counts.values())
        print(f"{qid} {prompt}: 총{total}  " + "  ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in counts.items()))

    print("\n=== 프롬프트별 전체 요약 (13문항 합산) ===")
    by_prompt = defaultdict(Counter)
    for r in judged:
        by_prompt[r["prompt"]][r["judge_label_4way"]] += 1
    for prompt, counts in sorted(by_prompt.items()):
        total = sum(counts.values())
        print(f"{prompt}: 총{total}  " + "  ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in counts.items()))


if __name__ == "__main__":
    main()
