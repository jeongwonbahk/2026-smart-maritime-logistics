# 앱 — LLM 할루시네이션 검증 데모

2026년 6~8월에 개발한 Streamlit 데모입니다.
질문을 넣으면 LLM 답변을 생성하고, 근거 문서를 검색해 의미 유사도로 대조한 뒤
**Safe / Warning / High Risk**로 표시합니다.

> 프로젝트가 7월에 실험·논문 쪽으로 방향을 바꾸면서, 이 앱은 제출용 데모로 마무리했습니다.
> 이후 연구 작업은 [논문 폴더](../논문_RAG근거오염실험/)에 있습니다.

---

## 흐름

```
질문 입력 → LLM 답변 생성 → 근거 문서 검색(RAG) → 의미 유사도 계산 → Risk Score → 색상 출력
```

## 파일

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 메인 화면 · Risk Score 계산과 시각화 |
| `llm.py` | GPT-4o-mini 답변 생성 |
| `search.py` | ChromaDB 검색 + 유사도 |
| `data_prep.py` | PDF → 정제 → 청킹 → 임베딩 → ChromaDB 적재 |

`llm.py` · `search.py` · `data_prep.py`는 논문 실험과 공용 모듈입니다.
이 폴더만 받아도 돌아가도록 사본을 두었습니다.
연구용 최신본은 [`../논문_RAG근거오염실험/core/`](../논문_RAG근거오염실험/core/)에 있고,
오염 검색(`search_contaminated`)과 단계적 근거 조립(`build_graduated_context`)이 추가돼 있습니다.

## 실행

```bash
pip install -r ../requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

python data_prep.py     # 문서 → ChromaDB (원본 PDF 필요)
streamlit run app.py
```

원본 규정 문서(PDF)와 생성된 ChromaDB는 용량·저작권 문제로 저장소에 포함하지 않았습니다.
