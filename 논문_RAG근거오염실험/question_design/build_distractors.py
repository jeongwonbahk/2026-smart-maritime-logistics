"""
build_distractors.py
질문세트_오염설계.md 의 11문항에 대해 방해 근거(설계 오염) 5개씩을 확정해
distractor_chunk_ids.csv 로 저장한다.

순서 = 헷갈리는 강도 순(강 → 약). 실험에서 레벨만큼 앞에서 잘라 쓴다
(기존 build_graduated_context 가 검색 결과를 [:level] 자르던 것과 같은 구조).

1~2순위는 손으로 고른 것(같은 속성 + 숫자 충돌), 3~5순위는 같은 문서에서
같은 단위의 다른 숫자를 가진 청크 중 질문과 유사한 순.
"""
import csv

D = "공통_안전_해양수산부_항만하역장비제작설치_정제_c{}"
G = "공통_정책_해양수산부_항만시설장비검사기준_정제_c{}"
F = "공통_정책고시_해양수산부_항만시설 사용 및 사용료에 관한 규정(2017)최종본_정제_c{}"
M = "공통_계류지침_KR_일점계류장치지침2017_정제_c{}"
U = "울산항_안전규정_울산항만공사_일반화물안전매뉴얼_정제_c{}"
H = "공통_안전규정_한국항만연수원_항만하역통합안전매뉴얼_정제_c{}"

ROWS = [
    ("Q1",  D.format(10), [D.format(13), D.format(31), D.format(29), D.format(16), D.format(15)]),
    ("Q2",  D.format(29), [D.format(31), D.format(13), D.format(10), D.format(16), D.format(15)]),
    ("Q3",  D.format(15), [D.format(16), D.format(17), D.format(13), D.format(10), D.format(31)]),
    ("Q4",  G.format(47), [G.format(58), G.format(53), G.format(79), G.format(77), G.format(40)]),
    ("Q5",  G.format(47), [G.format(58), G.format(84), G.format(80), G.format(73), G.format(53)]),
    ("Q6",  F.format(27), [F.format(33), F.format(32), F.format(26), F.format(29), F.format(30)]),
    ("Q7",  F.format(27), [F.format(33), F.format(32), F.format(26), F.format(29), F.format(35)]),
    ("Q8",  M.format(35), [M.format(41), M.format(40), M.format(30), M.format(44), M.format(78)]),
    ("Q9",  M.format(30), [M.format(31), M.format(46), M.format(35), M.format(41), M.format(40)]),
    ("Q10", U.format(79), [U.format(125), U.format(111), U.format(80), U.format(101), U.format(50)]),
    ("Q12", H.format(156),[H.format(160), H.format(14), H.format(99), H.format(31), H.format(52)]),
]

if __name__ == "__main__":
    import chromadb
    col = chromadb.PersistentClient(path="./chroma_db").get_collection("port_docs")
    need = {c for _, g, ds in ROWS for c in [g] + ds}
    found = set(col.get(ids=sorted(need))["ids"])
    missing = need - found
    if missing:
        raise SystemExit(f"존재하지 않는 chunk_id: {missing}")

    with open("distractor_chunk_ids.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "gold_chunk_id", "d1", "d2", "d3", "d4", "d5"])
        for qid, gold, ds in ROWS:
            w.writerow([qid, gold] + ds)
    print(f"{len(ROWS)}문항 x 방해근거 5개 -> distractor_chunk_ids.csv (전부 존재 확인)")
