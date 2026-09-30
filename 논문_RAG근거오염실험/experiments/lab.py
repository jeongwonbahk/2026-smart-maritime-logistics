"""
lab.py
팀원들이 브라우저에서 실험 변수(top_k, 조건, 항구 등)를 조정하며
답변을 즉시 확인해볼 수 있는 테스트용 도구.

주의: 이건 CLAUDE.md에서 "보류" 상태인 제품용 검증 앱(app.py)이 아니라,
실험 파이프라인(search.py/llm.py/experiment.py/judge.py)을 사람이 눈으로
확인하며 테스트하기 위한 내부용 도구입니다.

실행:
    streamlit run lab.py
"""

import pandas as pd
import streamlit as st

from experiment import ALL_CONDITIONS, run_condition
from judge import judge_answer
from search import search_contaminated, search_normal

st.set_page_config(page_title="RAG 오염 실험 테스트 랩", layout="wide")
st.title("RAG 오염 검색 실험 — 테스트 랩")
st.caption(
    "실험 조건·top_k 등을 조정하며 실제 답변을 바로 확인하는 내부 테스트 도구입니다. "
    "(제품용 검증 앱 app.py와는 별개, 실험 데이터 수집은 experiment.py를 사용하세요)"
)

tab_run, tab_search, tab_judge = st.tabs(["조건별 답변 실행", "검색 결과만 보기", "판정 테스트"])


# ─────────────────────────────────────────────
# 탭 1: 질문 → 조건별 답변 실행
# ─────────────────────────────────────────────
with tab_run:
    col_q, col_opt = st.columns([2, 1])

    with col_q:
        question = st.text_area(
            "질문",
            height=100,
            placeholder="예: 정기안전점검은 얼마나 자주 실시해야 하나요?",
        )

    with col_opt:
        port = st.text_input("질문 항구 (오염 검색 기준 — 예: 부산항)", value="공통")
        top_k = st.slider("top_k (검색할 청크 개수)", 1, 10, 3)
        conditions = st.multiselect("실행할 조건", ALL_CONDITIONS, default=ALL_CONDITIONS)

    run_clicked = st.button("실행", type="primary", disabled=not question.strip())

    if run_clicked:
        question_record = {
            "question_id": "lab",
            "question": question.strip(),
            "port": port.strip() or "공통",
            "expected_has_answer": None,
        }

        results = []
        progress = st.progress(0.0, text="실행 중...")
        for i, condition in enumerate(conditions, start=1):
            row = run_condition(question_record, condition, top_k)
            results.append(row)
            progress.progress(i / len(conditions), text=f"{condition} 완료 ({i}/{len(conditions)})")
        progress.empty()

        for row in results:
            with st.expander(f"[{row['condition']}]  검색된 청크 {row['num_retrieved']}개", expanded=True):
                st.write(row["answer"])
                if row["retrieved_chunk_ids"]:
                    st.caption("근거 청크 ID: " + row["retrieved_chunk_ids"])

        df = pd.DataFrame(results)
        st.subheader("표로 비교")
        st.dataframe(df[["condition", "top_k", "num_retrieved", "answer"]], use_container_width=True)

        st.download_button(
            "이 결과를 CSV로 다운로드",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="lab_result.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
# 탭 2: LLM 호출 없이 검색 결과만 확인 (디버깅용)
# ─────────────────────────────────────────────
with tab_search:
    st.caption("LLM을 호출하지 않고 search.py의 검색 결과(+별표 자동 연결)만 확인합니다.")

    col_q2, col_opt2 = st.columns([2, 1])
    with col_q2:
        search_question = st.text_area("질문", key="search_question", height=80)
    with col_opt2:
        search_top_k = st.slider("top_k", 1, 10, 3, key="search_top_k")
        search_mode = st.radio("검색 방식", ["정상 검색", "오염 검색"], key="search_mode")
        search_port = st.text_input("질문 항구", value="공통", key="search_port")

    if st.button("검색", disabled=not search_question.strip()):
        if search_mode == "정상 검색":
            chunks = search_normal(search_question.strip(), top_k=search_top_k, port=search_port.strip() or None)
        else:
            chunks = search_contaminated(search_question.strip(), question_port=search_port.strip() or "공통", top_k=search_top_k)

        if not chunks:
            st.warning("검색 결과가 없습니다.")
        for c in chunks:
            dist = f"{c['distance']:.4f}" if c["distance"] is not None else "참조로 연결됨"
            label = f"{c['chunk_id']}  (port={c['metadata']['port']}, distance={dist})"
            if c["linked_via"]:
                label += f"  ← {c['linked_via']} 참조"
            with st.expander(label):
                st.write(c["text"])


# ─────────────────────────────────────────────
# 탭 3: judge.py 판정을 즉석에서 테스트
# ─────────────────────────────────────────────
with tab_judge:
    st.caption("질문·모델 답변·(있으면) 정답 근거를 넣으면 judge.py가 정답/기권/오답 중 무엇으로 판정하는지 바로 확인합니다.")

    jq = st.text_area("질문", key="judge_question", height=80)
    ja = st.text_area("모델 답변", key="judge_answer_text", height=100)
    jr = st.text_area("정답 근거 (비워두면 '원래 답이 없는 질문'으로 판정)", key="judge_reference", height=100)

    if st.button("판정", disabled=not (jq.strip() and ja.strip())):
        with st.spinner("판정 중..."):
            result = judge_answer(jq.strip(), ja.strip(), reference=jr.strip() or None)

        label = result["label"]
        color = {"정답": "green", "기권": "orange", "오답": "red"}.get(label, "gray")
        st.markdown(f"### 판정: :{color}[{label}]")
        st.write(result["reason"])
