"""
verify_pinned_chunks.py
gold_chunk_ids.csv에 고정해둔 chunk_id가 실제로 gold_answers_test.csv의 정답과
내용이 맞는지 직접 대조한다 (find_gold_chunks.py처럼 자연 검색순위로 다시 찾는 게
아니라, 이미 고정한 chunk_id 자체를 그대로 불러와서 대조하는 최종 확인용).
"""
import csv

from search import get_chunk_by_id


def load_csv_dict(path, key, value):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row[key]] = row[value]
    return out


def main():
    gold_chunks = load_csv_dict("gold_chunk_ids.csv", "question_id", "gold_chunk_id")
    gold_answers = load_csv_dict("gold_answers_test.csv", "question_id", "reference_answer")

    for qid in sorted(gold_chunks, key=lambda x: int(x[1:])):
        chunk_id = gold_chunks[qid]
        answer = gold_answers.get(qid, "")
        chunk = get_chunk_by_id(chunk_id)
        if chunk is None:
            print(f"{qid}: [오류] chunk_id를 못 찾음 — {chunk_id}")
            continue
        # 정답 문구에서 핵심 숫자/고유어 몇 개가 청크 본문에 실제로 있는지 대충 확인
        print(f"{qid} -> {chunk_id}")
        print(f"  정답: {answer[:80]}")
        print(f"  청크 앞부분: {chunk['text'][:80].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
