"""
find_confusable_pairs.py
"헷갈릴 만한 오염 근거" 후보를 찾는다.

기준: 같은 문서 안에서
  1) 임베딩 유사도가 높고 (실험이 쓰는 것과 같은 모델)
  2) 같은 단위의 숫자를 갖고 있는데 값이 서로 다르고
  3) 공통 어휘가 있는
청크 쌍.

이건 어디까지나 '후보 추리기'다. 예전 find_gold_chunks.py가 자동 매칭만 믿었다가
q1·q2 gold_chunk_id를 틀린 전례가 있으므로, 결과는 반드시 사람이 원문을 읽고
판단한다.
"""
import re
import itertools
from collections import defaultdict

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "port_docs"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"

# 숫자 + 단위 추출 (한글/기호 단위 혼용)
UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(㎜|mm|밀리미터|㎝|cm|m(?![a-z])|미터|%|퍼센트|년|개월|월|일|"
    r"톤|t(?![a-z])|°|도(?!록)|배|회|초|분|시간|원|만원|억)"
)

# 의미 없는 흔한 단어 (공통 어휘 계산에서 제외)
STOPWORDS = {
    "경우", "다음", "사항", "규정", "따른", "따라", "대하여", "대한", "하여야", "한다",
    "있다", "없다", "것은", "것을", "이상", "이하", "이내", "또는", "및", "등의", "등을",
    "제1항", "제2항", "제3항", "해당", "필요", "수", "그", "이", "때", "위하여", "관한",
}


def get_chunks_by_source():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION_NAME)
    res = col.get(include=["documents", "metadatas"])
    by_source = defaultdict(list)
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        by_source[meta["source"]].append((cid, doc))
    return by_source


def extract_units(text):
    """텍스트에서 (단위, 값) 집합을 뽑는다. 단위는 표기 정규화."""
    norm = {"mm": "㎜", "밀리미터": "㎜", "cm": "㎝", "미터": "m", "퍼센트": "%",
            "t": "톤", "도": "°", "월": "개월"}
    out = set()
    for val, unit in UNIT_PATTERN.findall(text):
        unit = norm.get(unit, unit)
        out.add((unit, float(val)))
    return out


def content_words(text):
    """한글 2글자 이상 명사성 토큰 (대충) — 공통 어휘 비교용"""
    words = re.findall(r"[가-힣]{2,}", text)
    return {w for w in words if w not in STOPWORDS}


def numeric_conflict(units_a, units_b):
    """같은 단위인데 값이 다른 조합이 있으면 그 목록을 반환"""
    conflicts = []
    units_a_by_unit = defaultdict(set)
    units_b_by_unit = defaultdict(set)
    for u, v in units_a:
        units_a_by_unit[u].add(v)
    for u, v in units_b:
        units_b_by_unit[u].add(v)
    for unit in set(units_a_by_unit) & set(units_b_by_unit):
        va, vb = units_a_by_unit[unit], units_b_by_unit[unit]
        only_a = va - vb
        only_b = vb - va
        if only_a and only_b:
            conflicts.append((unit, sorted(only_a), sorted(only_b)))
    return conflicts


def main():
    print("청크 로드 중...")
    by_source = get_chunks_by_source()
    model = SentenceTransformer(EMBED_MODEL)

    all_candidates = []

    for source, chunks in sorted(by_source.items()):
        # 숫자가 있는 청크만 대상으로 (정답/오답이 명확히 갈려야 하므로)
        numeric_chunks = [(cid, txt, extract_units(txt)) for cid, txt in chunks]
        numeric_chunks = [(cid, txt, u) for cid, txt, u in numeric_chunks if u]
        if len(numeric_chunks) < 2:
            continue

        texts = [txt for _, txt, _ in numeric_chunks]
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        sim = embs @ embs.T

        doc_candidates = []
        for i, j in itertools.combinations(range(len(numeric_chunks)), 2):
            score = float(sim[i, j])
            if score < 0.60:
                continue
            cid_a, txt_a, units_a = numeric_chunks[i]
            cid_b, txt_b, units_b = numeric_chunks[j]
            conflicts = numeric_conflict(units_a, units_b)
            if not conflicts:
                continue
            shared = content_words(txt_a) & content_words(txt_b)
            if len(shared) < 3:
                continue
            doc_candidates.append({
                "source": source, "score": score,
                "cid_a": cid_a, "txt_a": txt_a,
                "cid_b": cid_b, "txt_b": txt_b,
                "conflicts": conflicts,
                "shared": sorted(shared)[:12],
            })

        doc_candidates.sort(key=lambda c: -c["score"])
        all_candidates.extend(doc_candidates[:8])   # 문서당 상위 8쌍만
        if doc_candidates:
            print(f"  {source}: {len(doc_candidates)}쌍 (상위 {min(8,len(doc_candidates))}개 채택)")

    all_candidates.sort(key=lambda c: (c["source"], -c["score"]))

    with open("confusable_pairs_report.txt", "w", encoding="utf-8") as f:
        f.write(f"총 후보 {len(all_candidates)}쌍\n")
        f.write("=" * 100 + "\n\n")
        for n, c in enumerate(all_candidates, 1):
            f.write(f"[{n}] 유사도 {c['score']:.3f} | {c['source']}\n")
            f.write(f"  숫자충돌: ")
            for unit, only_a, only_b in c["conflicts"]:
                f.write(f"{unit}({only_a} vs {only_b}) ")
            f.write(f"\n  공통어휘: {', '.join(c['shared'])}\n\n")
            f.write(f"  A) {c['cid_a']}\n")
            f.write(f"     {c['txt_a'][:400]}\n\n")
            f.write(f"  B) {c['cid_b']}\n")
            f.write(f"     {c['txt_b'][:400]}\n")
            f.write("-" * 100 + "\n\n")

    print(f"\n총 {len(all_candidates)}쌍 -> confusable_pairs_report.txt")


if __name__ == "__main__":
    main()
