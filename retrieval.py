"""
=============================================================
  Lab 2 – Knowledge Engineering & Domain Grounding
  File: retrieval_test.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

Run 3 test queries against the ChromaDB vector store.
Results are printed to console AND saved to retrieval_test.md

RUN (after ingest_data.py has been run):
  python retrieval_test.py

NO API KEY REQUIRED – uses local sentence-transformers model.
=============================================================
"""

import chromadb
import datetime
from sentence_transformers import SentenceTransformer

# ── Config (must match ingest_data.py) ─────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "resume_assistant_kb"

print(f"[Model] Loading local embedding model: {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
print("[Model] Ready!\n")


def embed(text: str) -> list:
    return embedder.encode([text], normalize_embeddings=True)[0].tolist()


def query_db(query: str, n: int = 3, where: dict = None) -> list:
    """Query ChromaDB and return top-n results."""
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results    = collection.query(
        query_embeddings=[embed(query)],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": t, "metadata": m, "distance": round(d, 4)}
        for t, m, d in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def display_and_record(test_num: int, query: str, results: list,
                       filter_used: str = None) -> str:
    """Print results to console and return as Markdown string."""
    divider = "-" * 60
    print(f"\n{'='*60}")
    print(f"  TEST {test_num}")
    print(f"{'='*60}")
    print(f"  Query  : {query}")
    if filter_used:
        print(f"  Filter : {filter_used}")
    print(divider)

    md = [f"### Test {test_num}",
          f"**Query:** `{query}`"]
    if filter_used:
        md.append(f"**Metadata Filter:** `{filter_used}`")
    md.append("")

    for i, r in enumerate(results, 1):
        m       = r["metadata"]
        preview = r["text"][:280].replace("\n", " ").strip()

        print(f"\n  Result #{i}")
        print(f"    Title          : {m.get('title','?')}")
        print(f"    doc_type       : {m.get('doc_type','?')}")
        print(f"    department     : {m.get('department','?')}")
        print(f"    priority_level : {m.get('priority_level','?')}")
        print(f"    Chunk          : {m.get('chunk_index',0)+1} / {m.get('total_chunks',1)}")
        print(f"    Distance       : {r['distance']}  (lower = more similar)")
        print(f"    Preview        : {preview}...")

        md += [
            f"**Result #{i}**",
            f"- **Title:** {m.get('title','?')}",
            f"- **doc_type:** `{m.get('doc_type','?')}`",
            f"- **department:** `{m.get('department','?')}`",
            f"- **priority_level:** `{m.get('priority_level','?')}`",
            f"- **Chunk:** {m.get('chunk_index',0)+1} of {m.get('total_chunks',1)}",
            f"- **Distance:** {r['distance']}",
            f"- **Preview:** _{preview}..._",
            "",
        ]

    print(divider)
    return "\n".join(md)


def main():
    print("=" * 60)
    print("  Lab 2 – Retrieval Tests (FREE, no API key)")
    print("  Project: Agentic AI Resume & Job Application Assistant")
    print("=" * 60)

    report = [
        "# retrieval_test.md",
        "## Lab 2 – Retrieval Test Results",
        f"**Project:** Agentic AI Resume & Job Application Assistant  ",
        f"**Date:** {datetime.date.today()}  ",
        f"**Embedding Model:** {EMBED_MODEL} (local, free)  ",
        f"**Collection:** `{COLLECTION_NAME}`  ",
        "", "---", "",
    ]

    # ── TEST 1: Pure Semantic Retrieval ────────────────────
    q1   = "What formatting rules should I follow to make my resume ATS compatible?"
    res1 = query_db(q1, n=3)
    md1  = display_and_record(1, q1, res1)
    report += [md1, "", "---", ""]

    # ── TEST 2: Agent-Focused Semantic Retrieval ───────────
    q2   = "How does the Resume Improver Agent rewrite bullet points for better impact?"
    res2 = query_db(q2, n=3)
    md2  = display_and_record(2, q2, res2)
    report += [md2, "", "---", ""]

    # ── TEST 3: METADATA FILTERING (required by lab spec) ──
    # Filter: only search within department = "domain_knowledge"
    # This mirrors: "Find the price, but only from the Enterprise document"
    q3     = "What is the ATS score formula and what threshold triggers another improvement cycle?"
    filt3  = {"department": {"$eq": "domain_knowledge"}}
    res3   = query_db(q3, n=3, where=filt3)
    fdesc  = 'department == "domain_knowledge"'
    md3    = display_and_record(3, q3, res3, filter_used=fdesc)
    report += [md3, "", "---", ""]

    # ── Summary Table ──────────────────────────────────────
    report += [
        "## Summary Table",
        "",
        "| Test | Filter | Top Result | Distance |",
        "|------|--------|-----------|----------|",
        f"| 1 | None (semantic) | {res1[0]['metadata'].get('title','?')} | {res1[0]['distance']} |",
        f"| 2 | None (semantic) | {res2[0]['metadata'].get('title','?')} | {res2[0]['distance']} |",
        f"| 3 | department == domain_knowledge | {res3[0]['metadata'].get('title','?')} | {res3[0]['distance']} |",
        "",
        "**Key Finding:** Test 3 shows metadata filtering restricts results to only "
        "`domain_knowledge` documents, preventing agent description docs from appearing "
        "even if they are semantically close. This significantly improves retrieval precision.",
    ]

    # Save to file
    out_path = "retrieval_test_output.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n✅ Results saved to: {out_path}")
    print("   Submit: ingest_data.py, retrieval_test.md, grounding_justification.txt")


if __name__ == "__main__":
    main()