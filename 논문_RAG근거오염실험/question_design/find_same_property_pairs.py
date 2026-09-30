"""
find_same_property_pairs.py
공통 ↔ 지역 문서에서 "같은 항목명에 다른 값"인 쌍을 찾는다.

왜 이 방식인가 (2026-09-22):
  앞서 임베딩 유사도로 찾은 628쌍은 주제만 비슷하고 충돌 숫자가 서로 다른 속성이었다
  (공통 60°=총걸림각 vs 울산 30°=붐 각도). 실험이 성립하려면 Q5처럼
  **같은 항목을 가리키는 표현에 값만 다른** 쌍이어야 한다.

방법:
  각 청크에서 (숫자 앞 문구, 숫자, 단위)를 뽑고, 숫자 앞 문구를 정규화해 key로 삼는다.
  같은 key가 공통 청크와 지역 청크 양쪽에 나타나면서 값이 다르면 후보.
"""
import re
from collections import defaultdict

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION = "port_docs"

UNIT = re.compile(
    r"([가-힣A-Za-z()·\s]{4,30}?)\s*"          # 숫자 앞 문구
    r"(\d+(?:\.\d+)?)\s*"
    r"(㎜|mm|밀리미터|㎝|cm|m(?![a-z])|미터|%|퍼센트|년|개월|톤|°|도(?!록)|배|회|시간|원)"
)
NORM = {"mm": "㎜", "밀리미터": "㎜", "cm": "㎝", "미터": "m", "퍼센트": "%", "도": "°"}
# 의미 없는 연결어 — key에서 제거
TRIM = re.compile(r"(이내|이상|이하|약|최소|최대|정도|는|은|이|가|을|를|의|에|로|으로|및|와|과|:|：)+$")
STOPKEY = {"", "높이", "길이", "폭", "수", "것", "때", "경우", "기준", "이상", "이하"}


def norm_key(s):
    s = re.sub(r"\s+", "", s)
    s = TRIM.sub("", s)
    return s[-14:]            # 뒤쪽 14자가 항목명에 가깝다


def extract(text):
    out = []
    for pre, val, unit in UNIT.findall(text):
        k = norm_key(pre)
        if len(k) < 4 or k in STOPKEY:
            continue
        out.append((k, NORM.get(unit, unit), float(val)))
    return out


def main():
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    res = col.get(include=["documents", "metadatas"])

    # key -> port -> [(chunk_id, value, source, 원문조각)]
    table = defaultdict(lambda: defaultdict(list))
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        port = meta["port"]
        for k, unit, v in extract(doc):
            i = doc.replace(" ", "").find(k)
            snippet = re.sub(r"\s+", " ", doc)[:0]  # 나중에 채움
            table[(k, unit)][port].append((cid, v, meta["source"], doc))

    hits = []
    for (k, unit), byport in table.items():
        if "공통" not in byport:
            continue
        cvals = {v for _, v, _, _ in byport["공통"]}
        for port, items in byport.items():
            if port == "공통":
                continue
            lvals = {v for _, v, _, _ in items}
            if not (lvals - cvals):        # 값이 달라야 함
                continue
            c = byport["공통"][0]
            l = next(x for x in items if x[1] not in cvals)
            hits.append({
                "key": k, "unit": unit, "port": port,
                "c_val": c[1], "c_cid": c[0], "c_src": c[2], "c_doc": c[3],
                "l_val": l[1], "l_cid": l[0], "l_src": l[2], "l_doc": l[3],
            })

    hits.sort(key=lambda h: (-len(h["key"]), h["port"]))
    print(f"같은 항목명 + 다른 값 쌍: {len(hits)}개")
    from collections import Counter
    print("항만별:", dict(Counter(h["port"] for h in hits)))

    def around(doc, key, val, unit):
        flat = re.sub(r"\s+", "", doc)
        i = flat.find(key)
        return flat[max(0, i - 20):i + 40] if i >= 0 else flat[:60]

    with open("same_property_pairs.txt", "w", encoding="utf-8") as f:
        f.write(f"같은 항목명 + 다른 값 쌍 {len(hits)}개\n" + "=" * 96 + "\n\n")
        for n, h in enumerate(hits, 1):
            f.write(f"[{n}] 항목 '{h['key']}' ({h['unit']})  "
                    f"공통 {h['c_val']} vs {h['port']} {h['l_val']}\n")
            f.write(f"  [공통] {h['c_cid']}\n")
            f.write(f"     …{around(h['c_doc'], h['key'], h['c_val'], h['unit'])}…\n")
            f.write(f"  [{h['port']}] {h['l_cid']}\n")
            f.write(f"     …{around(h['l_doc'], h['key'], h['l_val'], h['unit'])}…\n")
            f.write("-" * 96 + "\n\n")
    print("→ same_property_pairs.txt")

    print("\n=== 상위 12개 ===")
    for h in hits[:12]:
        print(f"\n'{h['key']}' ({h['unit']})  공통 {h['c_val']} vs {h['port']} {h['l_val']}")
        print(f"   공통: …{around(h['c_doc'], h['key'], h['c_val'], h['unit'])}…")
        print(f"   {h['port']}: …{around(h['l_doc'], h['key'], h['l_val'], h['unit'])}…")


if __name__ == "__main__":
    main()
