"""
search.py
ChromaDB에 저장된 항만 문서 청크를 검색하는 모듈.

- search_normal: 질문과 관련된 상위 top_k개 청크를 정상적으로 검색
- search_contaminated: RAG 오염 실험용 — 질문과 무관한 다른 항구 문서를
  일부러 근거로 골라서 반환 (공통 문서는 오염으로 치지 않음)

사용법:
    python search.py
"""

from sentence_transformers import SentenceTransformer
import chromadb

# data_prep.py와 동일한 설정을 사용해야 함 (변경 금지 — CLAUDE.md 참고)
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "port_docs"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"

_model = None
_collection = None


def get_model():
    """임베딩 모델을 한 번만 로드해서 재사용"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    """ChromaDB 컬렉션을 한 번만 연결해서 재사용"""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _query_to_results(question, n_results, where=None):
    """질문을 임베딩해 ChromaDB에 질의하고, 청크 리스트로 정리해서 반환"""
    model = get_model()
    query_embedding = model.encode([question]).tolist()

    collection = get_collection()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        chunks.append({
            "chunk_id": id_,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "linked_via": None,   # 의미 검색으로 뽑힌 청크는 참조 연결이 아님
        })
    return chunks


def _expand_with_annex_refs(chunks):
    """
    검색된 청크 중 refs(별표/별지 언급)가 있는데 그 별표 청크가 아직
    결과에 없으면, data_prep.py가 남긴 annex 메타데이터로 통째로 찾아 붙인다.
    (예: 본문이 "별표6 참고"라고만 하고 끝나면 검색 결과만으로는 별표6
    내용을 알 수 없어서 LLM이 못 답하거나 지어낼 수 있음 — 이를 방지)
    """
    collection = get_collection()
    existing_ids = {c["chunk_id"] for c in chunks}
    existing_annex = {c["metadata"].get("annex") for c in chunks if c["metadata"].get("annex")}

    ref_names = set()
    for c in chunks:
        for name in (c["metadata"].get("refs") or "").split(","):
            name = name.strip()
            if name and name not in existing_annex:
                ref_names.add(name)

    expanded = list(chunks)
    for name in ref_names:
        matched = collection.get(where={"annex": name}, include=["documents", "metadatas"])
        for id_, doc, meta in zip(matched["ids"], matched["documents"], matched["metadatas"]):
            if id_ in existing_ids:
                continue
            existing_ids.add(id_)
            expanded.append({
                "chunk_id": id_,
                "text": doc,
                "metadata": meta,
                "distance": None,     # 의미 검색 순위가 아니라 참조로 붙은 청크
                "linked_via": name,   # 어떤 청크의 refs 때문에 딸려왔는지
            })
    return expanded


def search_normal(question, top_k=5, port=None):
    """
    질문과 관련된 청크를 정상적으로 top_k개 검색.
    port를 지정하면 해당 항구(+공통) 문서에서만 검색하고,
    지정하지 않으면 전체 문서에서 검색한다.
    """
    where = None
    if port:
        where = {"port": {"$in": [port, "공통"]}}
    chunks = _query_to_results(question, n_results=top_k, where=where)
    return _expand_with_annex_refs(chunks)


def search_contaminated(question, question_port, top_k=5):
    """
    RAG 오염 검색(실험용): question_port와 무관한 '다른 항구' 문서만 골라서
    근거인 것처럼 반환한다. 공통 문서는 모든 항구에 정당하게 적용되므로
    오염으로 취급하지 않는다 (question_port != 다른 항구 AND != "공통").
    """
    where = {
        "$and": [
            {"port": {"$ne": question_port}},
            {"port": {"$ne": "공통"}},
        ]
    }
    chunks = _query_to_results(question, n_results=top_k, where=where)
    return _expand_with_annex_refs(chunks)


def get_chunk_by_id(chunk_id):
    """
    chunk_id로 청크 하나를 직접 가져온다.
    단계적 오염(v2)에서 정답 근거를 특정 슬롯에 고정할 때 사용 — 검색 순위와
    무관하게 "이 청크가 정답 근거다"라고 미리 확정해둔 chunk_id를 그대로 불러온다.
    """
    collection = get_collection()
    result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {
        "chunk_id": result["ids"][0],
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
        "distance": None,      # 검색 순위로 뽑힌 게 아니라 고정된 청크라 순위 개념 없음
        "linked_via": None,
    }


def build_graduated_context(question, question_port, gold_chunk_id, level):
    """
    실험 설계 v2 — 단계적 부분오염(L0~L5) 근거 조합.

    L0~L4: 정답 근거(gold_chunk_id)를 5개 슬롯 중 1개에 항상 고정하고,
           나머지 4자리를 "비교용 정상 청크"(오염 아님)와 "오염 청크"로 채운다.
           오염 개수 = level (0~4개). 정답 근거가 오염 증가와 함께 같이 사라지는
           문제(CLAUDE.md/실험설계 v2 검토 참고)를 막기 위한 장치 — 오답이 나오면
           "정답 근거가 없어서"가 아니라 순수하게 "오염된 근거에 낚여서"라고
           해석할 수 있게 한다.

           오염 청크는 유사도 상위 순으로 최대 4개를 미리 뽑아두고 레벨만큼
           앞에서 잘라 쓴다 — 레벨을 올릴 때 이전 레벨의 오염 청크를 그대로
           유지하며 1개씩만 누적하는 효과 (L1의 오염 ⊂ L2 ⊂ L3 ⊂ L4).
           이러면 레벨 간 결과 차이가 "오염 개수" 효과만 반영하고, "우연히
           다른 오염 청크가 뽑혀서" 생기는 잡음이 섞이지 않는다.

    L5*: 정답 근거 없이 5개 전부 오염 (v1 방식 그대로, 참조용 극한 조건).

    반환: 청크 리스트(최대 5개), search_normal/search_contaminated와 같은 형식.
    """
    if level == 5:
        return search_contaminated(question, question_port=question_port, top_k=5)

    if level not in (0, 1, 2, 3, 4):
        raise ValueError(f"알 수 없는 오염 레벨: {level} (0~5만 가능)")

    gold_chunk = get_chunk_by_id(gold_chunk_id)
    if gold_chunk is None:
        raise ValueError(f"gold_chunk_id를 찾을 수 없음: {gold_chunk_id}")

    # 오염 후보 4개를 유사도 상위 순으로 미리 뽑아두고 레벨만큼만 사용 (누적 보장)
    contamination_pool = search_contaminated(question, question_port=question_port, top_k=4)
    contamination = contamination_pool[:level]

    # 비교용 정상 청크로 나머지 슬롯을 채움 (정답 근거는 제외)
    needed_normal = 4 - level
    comparison = []
    if needed_normal > 0:
        candidates = search_normal(question, top_k=needed_normal + 3, port=question_port)
        comparison = [c for c in candidates if c["chunk_id"] != gold_chunk_id][:needed_normal]

    return [gold_chunk] + comparison + contamination


def _print_chunk(chunk):
    dist = f"{chunk['distance']:.4f}" if chunk["distance"] is not None else "-"
    linked = f", 참조로 연결됨 via {chunk['linked_via']}" if chunk["linked_via"] else ""
    print(f"  - {chunk['chunk_id']}  (port={chunk['metadata']['port']}, distance={dist}{linked})")
    print(f"    {chunk['text'][:80]}...")


if __name__ == "__main__":
    test_question = "안전점검의 과업내용은 어떻게 되나요?"
    test_port = "부산항"

    print(f"[정상 검색] 질문: {test_question}")
    for chunk in search_normal(test_question, top_k=3):
        _print_chunk(chunk)

    print(f"\n[오염 검색] 질문 항구: {test_port}")
    contaminated = search_contaminated(test_question, question_port=test_port, top_k=3)
    if not contaminated:
        print(f"  결과 없음 — '{test_port}' 이외의 항구 문서가 아직 없는 것으로 보임")
    for chunk in contaminated:
        _print_chunk(chunk)
