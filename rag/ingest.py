"""PDF -> clean section-aware chunks -> embeddings -> Chroma.

Design notes:
  * Section metadata comes from the PDF outline at index time (no page-range guessing
    at query time), and chunks never span two sections.
  * Running headers/footers ("1.2 • History of Psychology 9", "Access for free at
    openstax.org") are dropped instead of being embedded into every chunk.
  * Unicode is normalised rather than deleted, so names like "Piaget" and symbols survive.
  * The back matter (references, index) is excluded using the outline, not a fixed page.
"""
import bisect
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, asdict

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

SECTION_RE = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")
HEADER_RE = re.compile(r"^(\d+\s+)?\d+(\.\d+)?\s+•\s+.+?(\s+\d+)?$")   # running headers
FOOTER_RE = re.compile(r"^Access for free at openstax\.org", re.I)


@dataclass
class Segment:
    section: str          # "6.3", or "6.0" (chapter intro) / "6.review" (end-of-chapter)
    section_title: str
    start_page: int       # printed page numbers
    end_page: int         # inclusive


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    section: str
    section_title: str


def printed(pdf_index0: int) -> int:
    return pdf_index0 + 1 - config.PAGE_OFFSET


def build_segments(doc) -> list[Segment]:
    """Turn the PDF outline into contiguous page ranges labelled with section numbers."""
    toc = doc.get_toc(simple=True)                       # [level, title, 1-based page]
    numbered = [i for i, (_, t, _) in enumerate(toc) if SECTION_RE.match(t.strip())]
    first, last = numbered[0], numbered[-1]
    first_ch = SECTION_RE.match(toc[first][1].strip()).group(1)

    # Start at the chapter-1 title entry (the nearest higher-level entry before section 1.1).
    start = first
    while start > 0 and toc[start][0] >= toc[first][0]:
        start -= 1
    # Stop at the first entry after the last section that is at chapter level or above,
    # e.g. "References" / "Index".
    stop = last + 1
    while stop < len(toc) and toc[stop][0] >= toc[last][0]:
        stop += 1
    end_page = toc[stop][2] - 1 if stop < len(toc) else doc.page_count

    segs: list[Segment] = []
    chapter = first_ch
    for i in range(start, stop):
        level, title, page = toc[i]
        title = title.strip()
        m = SECTION_RE.match(title)
        if m:
            chapter = m.group(1)
            sec, name = f"{m.group(1)}.{m.group(2)}", m.group(3)
        elif level < toc[first][0]:
            # Chapter title: its chapter is the next numbered section's chapter.
            nxt = next(j for j in numbered if j > i)
            chapter = SECTION_RE.match(toc[nxt][1].strip()).group(1)
            sec, name = f"{chapter}.0", title
        elif title.lower() == "introduction":
            sec, name = f"{chapter}.0", title
        else:
            sec, name = f"{chapter}.review", title          # Key Terms, Summary, ...
        nxt_page = toc[i + 1][2] if i + 1 < stop else end_page + 1
        segs.append(Segment(sec, name, printed(page - 1), printed(nxt_page - 1) - 1))
    return segs


def clean_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.isdigit() or FOOTER_RE.match(s) or (HEADER_RE.match(s) and len(s) < 90):
            continue
        lines.append(s)
    return " ".join(lines)


def page_texts(doc) -> dict[int, str]:
    return {printed(i): clean_page(doc[i].get_text("text")) for i in range(doc.page_count)}


def segment_texts(pages: dict[int, str], segs: list[Segment]) -> list[tuple[Segment, str, list[int], list[int]]]:
    """Join the book into one text stream and cut it at each segment's heading.

    Numbered sections usually start mid-page, so a boundary is placed at the heading text
    ("6.3 Operant Conditioning") when it is found on the start page, else at the page start.
    Returns (segment, text, page_numbers, page_start_offsets_within_text) per segment.
    """
    first, last = segs[0].start_page, segs[-1].end_page
    stream, page_at = "", []                  # page_at: (offset, page) for every page
    for p in range(first, last + 1):
        page_at.append((len(stream), p))
        stream += pages.get(p, "") + " "
    page_off = dict((p, o) for o, p in page_at)

    bounds = []
    for seg in segs:
        o = page_off[seg.start_page]
        nxt = page_off.get(seg.start_page + 1, len(stream))
        idx = stream.find(f"{seg.section} {seg.section_title}", o, nxt) \
            if SECTION_RE.match(f"{seg.section} {seg.section_title}") else -1
        bounds.append(idx if idx >= 0 else o)
    bounds.append(len(stream))

    offs = [o for o, _ in page_at]
    out = []
    for k, seg in enumerate(segs):
        a, b = bounds[k], max(bounds[k], bounds[k + 1])
        i0 = bisect.bisect_right(offs, a) - 1
        i1 = bisect.bisect_right(offs, max(a, b - 1)) - 1
        pages_in = [page_at[i][1] for i in range(i0, i1 + 1)]
        starts = [max(0, page_at[i][0] - a) for i in range(i0, i1 + 1)]
        out.append((seg, stream[a:b], pages_in, starts))
    return out


def chunk_segments(seg_texts) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    for seg, text, pages_in, starts in seg_texts:
        if len(text.strip()) < 50:
            continue
        for n, doc in enumerate(splitter.create_documents([text])):
            page = pages_in[bisect.bisect_right(starts, doc.metadata["start_index"]) - 1]
            chunks.append(Chunk(
                id=f"{seg.section}:{seg.start_page}:{n}", text=doc.page_content.strip(),
                page=page, section=seg.section, section_title=seg.section_title,
            ))
    return chunks


def build_index():
    from .retrieve import embed_passages

    doc = pymupdf.open(config.BOOK_PDF)
    segs = build_segments(doc)
    chunks = chunk_segments(segment_texts(page_texts(doc), segs))
    n_sections = len({c.section for c in chunks if SECTION_RE.match(f"{c.section} x")
                      and not c.section.endswith(".0")})
    print(f"{len(segs)} outline segments, {n_sections} numbered sections, {len(chunks)} chunks")

    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True)
    (config.INDEX_DIR / "chunks.jsonl").write_text(
        "\n".join(json.dumps(asdict(c)) for c in chunks) + "\n")

    vectors = embed_passages([c.text for c in chunks])
    client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 1000):
        batch = chunks[i:i + 1000]
        col.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i + 1000].tolist(),
            metadatas=[{"page": c.page, "section": c.section, "section_title": c.section_title}
                       for c in batch],
        )
    print(f"Indexed {col.count()} chunks into {config.INDEX_DIR / 'chroma'}")
