"""
find_cross_port_pairs.py
공통 문서 ↔ 항만별(지역) 문서 사이에서 "짝"을 찾는다.

짝의 조건 (팀원 제안, 2026-09-22):
  - 구조가 비슷하고 (임베딩 유사도)
  - 같은 단위의 숫자가 서로 다르다

이 방식이 좋은 이유:
  기존에 부산항을 제외한 사유는 "전용 문서가 8청크뿐이라 비율이 안 맞는다"였다.
  그런데 비율로 섞지 않고 **짝이 있는 청크만 골라 쓰면** 그 제약이 사라진다.
  또한 "공통 질문 ← 특정 항만 값 주입"은 원래 연구 정의(다른 항구 문서를 근거로
  주입)와 정확히 일치한다.

설계:
  짝 있는 세트  -> 어려운 문제 (오염이 그럴듯함)
  짝 없는 공통  -> 쉬운 문제  (대조군)
"""
import re
import itertools
from collections import defaultdict

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION = "port_docs"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
SIM_MIN = 0.62
SHARED_MIN = 3

UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(㎜|mm|밀리미터|㎝|cm|m(?![a-z])|미터|%|퍼센트|년|개월|월|일|"
    r"톤|t(?![a-z])|°|도(?!록)|배|회|초|분|시간|원|만원)"
)
NORM = {"mm": "㎜", "밀리미터": "㎜", "cm": "㎝", "미터": "m", "퍼센트": "%",
        "t": "톤", "도": "°", "월": "개월"}
STOP = {"경우", "다음", "사항", "규정", "따른", "따라", "대하여", "대한", "하여야", "한다",
        "있다", "없다", "것은", "것을", "이상", "이하", "이내", "또는", "및", "등의", "등을",
        "해당", "필요", "위하여", "관한", "작업", "안전", "위험", "확인", "실시"}


def units(t):
    out = set()
    for v, u in UNIT.findall(t):
        out.add((NORM.get(u, u), float(v)))
    return out


def words(t):
    return {w for w in re.findall(r"[가-힣]{2,}", t) if w not in STOP}


def conflicts(ua, ub):
    ca, cb = defaultdict(set), defaultdict(set)
    for u, v in ua:
        ca[u].add(v)
    for u, v in ub:
        cb[u].add(v)
    out = []
    for u in set(ca) & set(cb):
        only_a, only_b = ca[u] - cb[u], cb[u] - ca[u]
        if only_a and only_b:
            out.append((u, sorted(only_a)[:3], sorted(only_b)[:3]))
    return out


def main():
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    res = col.get(include=["documents", "metadatas"])

    common, local = [], []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        u = units(doc)
        if not u:
            continue
        item = (cid, doc, u, meta["port"], meta["source"])
        (common if meta["port"] == "공통" else local).append(item)

    print(f"숫자 포함 청크 — 공통 {len(common)} / 지역 {len(local)}")
    by_port = defaultdict(int)
    for _, _, _, p, _ in local:
        by_port[p] += 1
    print("지역 내역:", dict(by_port))

    model = SentenceTransformer(EMBED_MODEL)
    print("임베딩 계산 중...")
    ec = model.encode([d for _, d, _, _, _ in common], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=False)
    el = model.encode([d for _, d, _, _, _ in local], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=False)
    sim = ec @ el.T
    print(f"유사도 행렬 {sim.shape}")

    cands = []
    idx = np.argwhere(sim >= SIM_MIN)
    print(f"유사도 {SIM_MIN} 이상 쌍: {len(idx)}개 — 숫자충돌 필터링 중")
    for i, j in idx:
        cid_c, doc_c, u_c, _, src_c = common[i]
        cid_l, doc_l, u_l, port_l, src_l = local[j]
        cf = conflicts(u_c, u_l)
        if not cf:
            continue
        shared = words(doc_c) & words(doc_l)
        if len(shared) < SHARED_MIN:
            continue
        cands.append({
            "score": float(sim[i, j]), "port": port_l,
            "cid_c": cid_c, "doc_c": doc_c, "src_c": src_c,
            "cid_l": cid_l, "doc_l": doc_l, "src_l": src_l,
            "cf": cf, "shared": sorted(shared)[:10],
        })

    cands.sort(key=lambda c: -c["score"])
    print(f"\n조건 통과 쌍: {len(cands)}개")
    cnt = defaultdict(int)
    for c in cands:
        cnt[c["port"]] += 1
    print("항만별:", dict(cnt))

    with open("cross_port_pairs.txt", "w", encoding="utf-8") as f:
        f.write(f"공통 ↔ 지역 짝 후보 {len(cands)}개\n" + "=" * 100 + "\n\n")
        for n, c in enumerate(cands[:80], 1):
            f.write(f"[{n}] 유사도 {c['score']:.3f} | 지역={c['port']}\n")
            f.write("  숫자충돌: ")
            for u, a, b in c["cf"]:
                f.write(f"{u}(공통{a} vs {c['port']}{b}) ")
            f.write(f"\n  공통어휘: {', '.join(c['shared'])}\n\n")
            f.write(f"  [공통] {c['cid_c']}\n     {c['doc_c'][:330]}\n\n")
            f.write(f"  [{c['port']}] {c['cid_l']}\n     {c['doc_l'][:330]}\n")
            f.write("-" * 100 + "\n\n")
    print("→ cross_port_pairs.txt")

    print("\n=== 상위 5개 미리보기 ===")
    for c in cands[:5]:
        print(f"\n유사도 {c['score']:.3f} | {c['port']}")
        for u, a, b in c["cf"][:2]:
            print(f"  충돌 {u}: 공통{a} vs {c['port']}{b}")
        print(f"  공통: {c['doc_c'][:90]}".replace("\n", " "))
        print(f"  지역: {c['doc_l'][:90]}".replace("\n", " "))


if __name__ == "__main__":
    main()
