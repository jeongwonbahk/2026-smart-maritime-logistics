"""
app.py
스마트항만 LLM Hallucination 검증 시스템 — Streamlit 웹앱

질문을 받아 RAG로 근거를 검색하고 LLM 답변을 생성한 뒤, 답변과 근거 문서의
의미론적 유사도로 신뢰도를 산출한다. 신뢰도가 낮으면 답변을 차단하고
왜 그렇게 판단했는지 근거를 함께 보여준다.

APP_SPEC.md 기준으로 작성됨. 연구용 코드(experiment.py/judge.py/analyze.py/lab.py)와는
무관한 별개의 제출·시연용 앱이며, search.py/llm.py의 기존 함수를 그대로 재사용한다.
"""

import time
from collections import Counter

import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from llm import get_llm_answer
from search import search_normal

# top-k 안에서 한 문서가 슬롯을 독점하지 않도록 문서당 최대 청크 수를 제한
MAX_CHUNKS_PER_SOURCE = 3
# 다양성 필터링 전에 미리 훑어볼 후보 개수 배수. 코퍼스가 2700청크를 넘어가면서
# 실제 정답 청크가 top_k*3위 밖으로 밀려나는 사례가 확인되어 여유를 더 둠
SEARCH_OVER_FETCH_MULTIPLIER = 6

EMBED_MODEL = "jhgan/ko-sroberta-multitask"

# 신뢰도 등급 경계값 — calibrate_threshold.py로 질문 10개×3회 실측한 결과로 확정
# (results/threshold_calibration.csv 참고: 함정 질문 유사도 0.08~0.15, 답변 가능 질문은
#  대부분 0.44~0.88. 그 사이 빈 구간에 경계값을 잡음)
THRESHOLD_SAFE = 0.65
THRESHOLD_WARNING = 0.30

# "근거 문서 간 일관성" 판단 기준 — 상위 근거들의 답변 유사도 편차가
# 이 값 이하면 일관적이라고 본다
CONSISTENCY_SPREAD_MAX = 0.20

# "근거 문서 확보"로 인정할 최소 근거 개수
MIN_EVIDENCE_COUNT = 3

# 답변을 한 글자씩 타이핑되듯 보여줄 때 글자당 딜레이(초)
CHAR_STREAM_DELAY = 0.015


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)


def search_with_diversity(question, top_k, port=None, max_per_source=MAX_CHUNKS_PER_SOURCE):
    """search_normal을 그대로 호출하되(시그니처 변경 없음), 문서 하나가 top_k를
    독점하지 않도록 문서(source)당 최대 max_per_source개까지만 남긴다.
    큰 문서 하나가 상위권을 다 차지해 다른 문서의 정답 청크가 밀려나는 것을 방지."""
    over_fetched = search_normal(question, top_k=top_k * SEARCH_OVER_FETCH_MULTIPLIER, port=port)
    result = []
    counts = Counter()
    for c in over_fetched:
        src = c["metadata"].get("source")
        if counts[src] >= max_per_source:
            continue
        result.append(c)
        counts[src] += 1
        if len(result) >= top_k:
            break
    return result


def compute_similarities(answer_text, chunks, model):
    """답변과 각 근거 청크 사이의 코사인 유사도를 구한다.
    (search.py가 주는 distance는 질문↔청크 유사도라 신뢰도 계산에는 쓸 수 없음 —
    답변↔청크 유사도를 별도로 계산해야 한다)"""
    answer_emb = model.encode([answer_text])
    chunk_embs = model.encode([c["text"] for c in chunks])
    sims = cos_sim(answer_emb, chunk_embs)[0].tolist()
    return sims


def grade(max_similarity):
    if max_similarity >= THRESHOLD_SAFE:
        return "Safe", "🟢"
    elif max_similarity >= THRESHOLD_WARNING:
        return "Warning", "🟡"
    return "High Risk", "🔴"


def build_reasoning(max_similarity, chunks, sims, question):
    """숫자만 보여주지 않고, 사람이 읽을 수 있는 판단 근거 목록을 만든다"""
    reasons = []

    ok = max_similarity >= THRESHOLD_SAFE
    reasons.append((
        ok,
        f"답변과 근거 문서의 일치도 — 최고 유사도 {max_similarity:.2f}"
        f" ({'기준 ' + format(THRESHOLD_SAFE, '.2f') + ' 이상' if ok else '기준 ' + format(THRESHOLD_SAFE, '.2f') + ' 미만'})",
    ))

    ok = len(chunks) >= MIN_EVIDENCE_COUNT
    reasons.append((
        ok,
        f"근거 문서 확보 — {len(chunks)}건 검색됨 (최소 {MIN_EVIDENCE_COUNT}건 기준)",
    ))

    if len(sims) >= 2:
        spread = max(sims) - min(sims)
        ok = spread <= CONSISTENCY_SPREAD_MAX
        reasons.append((
            ok,
            f"근거 문서 간 일관성 — 상위 근거 유사도 편차 {spread:.2f}"
            f" ({'편차 적음' if ok else '편차 큼, 근거들이 서로 다른 내용을 말할 수 있음'})",
        ))

    known_ports = {c["metadata"].get("port") for c in chunks if c["metadata"].get("port") != "공통"}
    mentioned_ports = [p for p in known_ports if p and p in question]
    other_port_chunks = [
        c for c in chunks
        if c["metadata"].get("port") not in (None, "공통") and c["metadata"].get("port") not in mentioned_ports
    ]
    if mentioned_ports:
        ok = len(other_port_chunks) == 0
        if ok:
            reasons.append((True, f"추천 항만 일치 — 질문의 항만({', '.join(mentioned_ports)})과 근거 문서의 항만이 일치"))
        else:
            other = ", ".join(sorted({c["metadata"].get("port") for c in other_port_chunks}))
            reasons.append((
                False,
                f"추천 항만 일치 — 질문은 {', '.join(mentioned_ports)} 관련인데"
                f" 근거에 다른 항만 문서({other})가 섞여 있음",
            ))

    return reasons


def stream_chars(text, delay=CHAR_STREAM_DELAY):
    """답변을 한 글자씩 타이핑되는 것처럼 보여주기 위한 제너레이터"""
    for ch in text:
        yield ch
        time.sleep(delay)


def render_verification(turn, model):
    """한 대화 턴의 근거 검증 결과를 계산(최초 1회, 이후 캐시)하고 화면에 표시한다"""
    if turn.get("verify_result") is None:
        with st.spinner("신뢰도 계산 중..."):
            sims = compute_similarities(turn["answer"], turn["chunks"], model)
            max_similarity = max(sims)
            turn["verify_result"] = {
                "sims": sims,
                "max_similarity": max_similarity,
                "trust_score": round(max_similarity * 100),
                "label_emoji": grade(max_similarity),
                "reasoning": build_reasoning(max_similarity, turn["chunks"], sims, turn["question"]),
            }

    vr = turn["verify_result"]
    label, emoji = vr["label_emoji"]

    st.divider()
    st.subheader(f"{emoji} 신뢰도: {label} ({vr['trust_score']}점)")
    st.progress(vr["max_similarity"])

    if label == "High Risk":
        st.error(
            "🔴 **High Risk — 이 답변은 근거로 충분히 뒷받침되지 않습니다**\n\n"
            "잘못된 정보가 실제 안전 지침으로 이어질 수 있으니, 위 답변을 그대로 신뢰하지 마시고 "
            "아래 참조 문서를 직접 확인하시거나 담당자에게 문의하십시오."
        )
    elif label == "Warning":
        st.warning("⚠️ 이 답변은 근거와의 일치도가 낮은 편입니다. 참조 문서를 함께 확인하세요.")

    st.subheader("판단 근거")
    for ok, text in vr["reasoning"]:
        st.markdown(f"{'✅' if ok else '⚠️'} {text}")

    st.subheader("참조 문서")
    for chunk, sim in zip(turn["chunks"], vr["sims"]):
        meta = chunk["metadata"]
        with st.expander(f"{meta.get('source', '알 수 없음')}  (항만: {meta.get('port', '미상')}, 유사도: {sim:.2f})"):
            st.write(chunk["text"])


def render_turn(turn, index, model):
    """대화 한 턴(질문+답변, 필요하면 검증 결과까지)을 채팅 형태로 그린다"""
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        if turn.pop("just_added", False):
            st.write_stream(stream_chars(turn["answer"]))
        else:
            st.write(turn["answer"])

        if turn["chunks"] and not turn["answer"].startswith("오류:"):
            if st.button("답변 검증하기", key=f"verify_btn_{index}"):
                turn["verified"] = True
            if turn.get("verified"):
                render_verification(turn, model)


def main():
    st.set_page_config(page_title="스마트항만 LLM서비스 및 검증", page_icon="⚓")

    st.title("⚓ 스마트항만 LLM서비스 및 검증")
    st.caption("항만 작업 관련 질문에 답변합니다. 답변을 받은 뒤 원하면 '답변 검증하기'로 근거와 신뢰도를 확인할 수 있습니다")

    model = load_model()

    with st.sidebar:
        st.subheader("검색 설정")
        top_k = st.slider("검색할 근거 개수", 1, 10, 10)

    if "history" not in st.session_state:
        st.session_state.history = []

    for i, turn in enumerate(st.session_state.history):
        render_turn(turn, i, model)

    with st.bottom:
        question = st.chat_input("항만 작업 관련 질문을 입력하세요")
        st.caption("AI는 틀린 정보를 말할 수 있습니다. 중요 정보는 꼭 다시 확인하세요.")

    if question:
        question = question.strip()
        try:
            with st.spinner("근거 문서 검색 중..."):
                chunks = search_with_diversity(question, top_k=top_k)

            if not chunks:
                answer = "관련 문서를 찾지 못했습니다. 다른 질문으로 다시 시도해주세요."
            else:
                with st.spinner("답변 생성 중..."):
                    context = "\n\n---\n\n".join(c["text"] for c in chunks)
                    result = get_llm_answer(question, context=context, abstain_instruction=True)
                    answer = result["answer"]

            st.session_state.history.append({
                "question": question,
                "chunks": chunks,
                "answer": answer,
                "verified": False,
                "verify_result": None,
                "just_added": True,
            })
        except Exception as e:
            st.session_state.history.append({
                "question": question,
                "chunks": [],
                "answer": f"오류: 처리 중 문제가 발생했습니다. ({e})",
                "verified": False,
                "verify_result": None,
                "just_added": True,
            })

        st.rerun()


if __name__ == "__main__":
    main()
