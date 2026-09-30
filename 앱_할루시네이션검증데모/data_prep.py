"""
data_prep.py
PDF/마크다운 → 텍스트 추출 → 정제 → 청킹 → 임베딩 → ChromaDB 저장

사용법:
    pip install pypdf langchain-text-splitters sentence-transformers chromadb
    python data_prep.py

문서 파일명 규칙 (.pdf, .md 둘 다 지원):
    [항구]_[카테고리]_[출처]_[제목]_정제.pdf(.md)
    예) 부산항_접안_BPA_접안절차서_정제.pdf
    → 항구가 없는 일반 자료는 port="공통" 으로 자동 처리

    .md는 이미 정제된 텍스트이므로 PDF 추출 단계(레이아웃 깨짐, 단어 중간
    줄바꿈 등) 없이 바로 청킹된다. 원본이 스캔본이거나 PDF 정제 품질이
    떨어질 때는 직접 마크다운으로 옮겨 적는 것도 좋은 대안.
"""

import os
import re
import glob
import unicodedata
from collections import Counter

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DOCUMENTS_DIR = "./documents"          # 원본 PDF 폴더
CHROMA_PATH = "./chroma_db"            # ChromaDB 저장 경로
COLLECTION_NAME = "port_docs"

CHUNK_SIZE = 500                       # 청크 크기(자)
CHUNK_OVERLAP = 50                     # 청크 겹침(자)
EMBED_MODEL = "jhgan/ko-sroberta-multitask"   # 한국어 임베딩 모델

RESET_COLLECTION = True                # True면 실행 시 기존 컬렉션 삭제 후 새로 적재


# ─────────────────────────────────────────────
# 1. PDF/마크다운 → 텍스트
# ─────────────────────────────────────────────
def extract_pages(pdf_path):
    """PDF에서 페이지별 텍스트 리스트를 반환"""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        pages.append(text)
    return pages


def extract_pages_md(md_path):
    """마크다운(.md)은 이미 정제된 텍스트이므로 그대로 한 페이지로 반환"""
    with open(md_path, encoding="utf-8") as f:
        return [f.read()]


# ─────────────────────────────────────────────
# 2. 정제
# ─────────────────────────────────────────────
def _normalize_for_repeat_check(line):
    """
    "Ministry of Oceans and Fisheries _ 15"처럼 페이지 번호만 바뀌고
    나머지는 똑같은 머리말/꼬리말을 잡아내기 위해, 숫자를 전부 '#'로
    통일해서 비교한다 (그래야 페이지마다 값이 달라도 같은 패턴으로 인식됨).
    layout 추출 모드는 페이지 번호 자릿수에 따라 공백 개수도 미세하게
    달라지므로("#_ 해양수산부" vs "#   _ 해양수산부") 공백도 함께 정규화한다.
    """
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", line))


def detect_repeated_lines(pages, threshold=0.3):
    """
    페이지마다 반복 등장하는 머리말/꼬리말 자동 탐지.
    전체 페이지의 threshold 비율 이상에서 나타나는 짧은 줄을 반복 라인으로 간주.
    숫자·공백은 정규화해서 비교하므로 페이지 번호가 섞인 꼬리말도 잡아낸다.

    threshold 기본값을 0.5가 아닌 0.3으로 낮춘 이유: 문서가 본문/부록처럼
    서로 다른 머리말·꼬리말 관례를 쓰는 두 구간으로 나뉘어 있으면, 각 관례가
    전체 페이지의 절반에 못 미쳐 0.5 기준으로는 하나도 안 잡힐 수 있다.
    """
    if len(pages) < 3:
        return set()

    counter = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        # 각 페이지의 처음 2줄, 마지막 2줄만 후보로 검사
        candidates = lines[:2] + lines[-2:]
        for ln in set(candidates):
            if len(ln) <= 60:          # 너무 긴 줄은 본문일 가능성이 높음
                counter[_normalize_for_repeat_check(ln)] += 1

    # 짧은 문서(예: 3페이지)는 len(pages)*threshold가 1 미만으로 내려가 버려서
    # 단 한 번만 등장한 줄까지 "반복 머리말"로 오판정해 본문을 삭제하는 버그가 있었음
    # (예: 3페이지짜리 문서에서 실제 조문 제목이 통째로 사라짐). 최소 2회 이상
    # 등장해야 반복으로 인정하도록 하한을 둠.
    limit = max(2, len(pages) * threshold)
    return {pattern for pattern, cnt in counter.items() if cnt >= limit}


def clean_text(pages):
    """페이지 리스트를 정제해 하나의 텍스트로 합침"""
    repeated = detect_repeated_lines(pages)
    cleaned_pages = []

    for text in pages:
        lines = []   # 각 원소: [정제된 줄 텍스트, 줄 끝에 실제 공백이 있었는지]
        for raw in text.split("\n"):
            had_trailing_space = raw != raw.rstrip()   # strip 전에 원본 공백 여부 확인
            line = raw.strip()

            if not line:
                continue
            if _normalize_for_repeat_check(line) in repeated:   # 반복 머리말/꼬리말
                continue
            if re.fullmatch(r"-?\s*\d+\s*-?", line):   # "12", "- 12 -" 형태 페이지 번호
                continue
            if re.fullmatch(r"[^\w가-힣]+", line):      # 특수문자만 있는 줄
                continue

            # 이전 줄이 공백 없이 끝났고 앞뒤 모두 한글이면
            # 페이지 폭 제한으로 단어 중간이 줄바꿈된 것 → 공백 없이 이어붙임
            if (
                lines
                and not lines[-1][1]
                and re.search(r"[가-힣]$", lines[-1][0])
                and re.match(r"^[가-힣]", line)
            ):
                lines[-1][0] += line
                lines[-1][1] = had_trailing_space
            else:
                lines.append([line, had_trailing_space])

        cleaned_pages.append("\n".join(l for l, _ in lines))

    full = "\n".join(cleaned_pages)
    full = re.sub(r"\n{3,}", "\n\n", full)   # 과도한 빈 줄 축소
    full = re.sub(r"[ \t]+", " ", full)      # 연속 공백을 한 칸으로
    full = re.sub(r"(다\.|음\.|함\.)\s+", r"\1\n", full)  # 문장 종결 뒤 줄바꿈 삽입
    return full.strip()


# ─────────────────────────────────────────────
# 3. 메타데이터 (항구 추출)
# ─────────────────────────────────────────────
def extract_port(filename):
    """
    파일명 첫 토큰이 '~항' 형태면 항구명으로, 아니면 '공통'.
    예) 부산항_접안_BPA_... → '부산항'
        안전_해수부_...     → '공통'

    macOS는 한글 파일명을 NFD(자모 분해)로 반환하는데, 소스 코드에 적은
    문자열은 보통 NFC라서 정규화 없이 비교하면 "인천항".endswith("항")이
    False가 되는 등 항구가 전부 "공통"으로 잘못 분류되는 버그가 생긴다.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    first = unicodedata.normalize("NFC", base.split("_")[0])
    if first.endswith("항") or first.endswith("항만"):
        return first
    return "공통"


# ─────────────────────────────────────────────
# 4. 별표/별지 섹션 분리 및 참조 연결
# ─────────────────────────────────────────────
# 문서에서 별표류 섹션을 가리키는 표현들. 새로운 문서에서 다른 표현이
# 발견되면 이 리스트에 추가하면 됨 (예: "붙임", "서식" 등).
ANNEX_KEYWORDS = ["별표", "별지", "부록", "첨부"]
_ANNEX_PATTERN = "|".join(ANNEX_KEYWORDS)

# 줄 시작에 오는 "[별표 3]", "별지1", "부록2" 같은 섹션 헤더
ANNEX_HEADER = re.compile(rf"^\s*[\[\(〔]?\s*({_ANNEX_PATTERN})\s*(\d+)")
# 본문 속 "별표3 참고" 같은 언급
ANNEX_REF = re.compile(rf"({_ANNEX_PATTERN})\s*(\d+)")


def split_annex_sections(cleaned_text, doc_key):
    """
    정제된 텍스트를 [(섹션이름, 섹션텍스트), ...]로 분리.
    섹션이름이 None이면 본문, 아니면 "{doc_key}::별표3"처럼 채워지며 해당
    별표/별지/부록 섹션. "별표N" 같은 번호는 문서마다 겹칠 수 있으므로
    doc_key(파일명 기반)를 붙여 문서 간 충돌을 막는다.
    """
    sections = []
    current_name = None
    current_lines = []

    for ln in cleaned_text.split("\n"):
        m = ANNEX_HEADER.match(ln)
        if m:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
            current_name = f"{doc_key}::{m.group(1)}{m.group(2)}"
            current_lines = [ln]
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))
    return sections


def find_refs(chunk, own_annex, doc_key):
    """청크 안에서 '별표3 참고' 같은 언급을 찾아 '{doc_key}::별표3,...' 형태로 반환.
    "별표N" 언급은 항상 그 문서 자신의 별표를 가리키므로 doc_key를 붙여
    다른 문서의 동일 번호 별표와 섞이지 않게 한다.
    자기 자신이 속한 섹션(own_annex)은 제외."""
    found = []
    for m in ANNEX_REF.finditer(chunk):
        name = f"{doc_key}::{m.group(1)}{m.group(2)}"
        if name != own_annex and name not in found:
            found.append(name)
    return ",".join(found)


# ─────────────────────────────────────────────
# 5. 청킹
# ─────────────────────────────────────────────
def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 문단 → 줄 → 문장 → 어절 순으로 자연스러운 경계를 우선 탐색
        separators=["\n\n", "\n", "다. ", "음. ", ". ", " ", ""],
        length_function=len,
    )


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    pdf_paths = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    md_paths = glob.glob(os.path.join(DOCUMENTS_DIR, "*.md"))
    doc_paths = sorted(pdf_paths + md_paths)
    if not doc_paths:
        print(f"[중단] {DOCUMENTS_DIR} 에 PDF/마크다운 문서가 없습니다.")
        return

    print(f"문서 {len(doc_paths)}개 발견 (PDF {len(pdf_paths)}개, 마크다운 {len(md_paths)}개)\n")

    splitter = build_splitter()

    all_texts, all_ids, all_metas = [], [], []

    for path in doc_paths:
        # NFC로 정규화해서 chunk_id/source 메타데이터가 일관되게 저장되도록 함
        # (그래야 나중에 search.py 등에서 소스 코드에 적은 문자열과 비교가 어긋나지 않음)
        filename = unicodedata.normalize("NFC", os.path.basename(path))
        base = os.path.splitext(filename)[0]
        port = extract_port(filename)
        is_md = filename.lower().endswith(".md")

        try:
            pages = extract_pages_md(path) if is_md else extract_pages(path)
        except Exception as e:
            print(f"  [건너뜀] {filename} — 읽기 실패: {e}")
            continue

        text = clean_text(pages)
        if not text:
            print(f"  [건너뜀] {filename} — 추출된 텍스트 없음(스캔본 가능성)")
            continue

        # 별표/별지/부록 섹션을 본문과 분리해서 따로 청킹
        # (별표 내용이 앞뒤 본문과 섞여서 청크 경계가 애매해지는 것을 방지)
        sections = split_annex_sections(text, doc_key=base)

        chunk_i = 0
        n_annex = 0
        n_ref = 0
        for annex_name, sec_text in sections:
            for chunk in splitter.split_text(sec_text):
                chunk_i += 1
                chunk_id = f"{base}_c{chunk_i}"
                refs = find_refs(chunk, annex_name, doc_key=base)
                if annex_name:
                    n_annex += 1
                if refs:
                    n_ref += 1

                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metas.append({
                    "chunk_id": chunk_id,
                    "source": filename,
                    "port": port,
                    "annex": annex_name or "",   # 이 청크가 속한 별표/별지 (본문이면 빈 문자열)
                    "refs": refs,                # 이 청크가 언급하는 다른 별표/별지
                })

        print(f"  {filename}  |  port={port}  |  {len(pages)}p → {chunk_i}청크"
              f"  (별표 청크 {n_annex}개, 별표 언급 청크 {n_ref}개)")

    if not all_texts:
        print("\n[중단] 생성된 청크가 없습니다.")
        return

    # ── 임베딩 ──
    print(f"\n임베딩 생성 중... (모델: {EMBED_MODEL})")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=32)
    embeddings = [e.tolist() for e in embeddings]

    # ── ChromaDB 저장 ──
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 코사인 유사도 사용
    )

    # 대량 입력 시 배치로 나눠 저장
    B = 500
    for s in range(0, len(all_ids), B):
        collection.add(
            ids=all_ids[s:s+B],
            documents=all_texts[s:s+B],
            embeddings=embeddings[s:s+B],
            metadatas=all_metas[s:s+B],
        )

    # ── 요약 ──
    port_counts = Counter(m["port"] for m in all_metas)
    print("\n" + "=" * 46)
    print(f"청크 {len(all_ids)}개 저장 완료  →  {CHROMA_PATH}")
    print("-" * 46)
    print("항구별 청크 수")
    for port, cnt in port_counts.most_common():
        print(f"  {port:<8} {cnt}")
    print("=" * 46)

    if len(port_counts) < 2:
        print("\n[경고] 항구가 1종류뿐입니다.")
        print("       오염 검색(다른 항구 문서 주입) 실험을 하려면")
        print("       최소 2개 항구의 문서가 필요합니다.")


if __name__ == "__main__":
    main()
