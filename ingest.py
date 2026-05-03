"""
=============================================================
  Lab 2 – Knowledge Engineering & Domain Grounding
  File: ingest_data.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

NO API KEY REQUIRED — Uses HuggingFace local embedding model.
Model: sentence-transformers/all-MiniLM-L6-v2 (~90MB, free)
Downloads automatically on first run, then cached locally.

INSTALL (one time only):
  pip install -r requirements.txt

RUN:
  python ingest_data.py
"""

import os
import re
import hashlib

# ── Third-party ────────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer

# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

EMBED_MODEL     = "all-MiniLM-L6-v2"   # free, local, ~90MB, no API key
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "resume_assistant_kb"
CHUNK_SIZE      = 400   # target characters per chunk
CHUNK_OVERLAP   = 80    # overlap characters between chunks

# Load the local embedding model once at startup
print(f"[Model] Loading local embedding model: {EMBED_MODEL}")
print("        (First run downloads ~90MB — one-time only, then cached)")
embedder = SentenceTransformer(EMBED_MODEL)
print("[Model] Model loaded successfully!\n")


# ════════════════════════════════════════════════════════════
#  RAW KNOWLEDGE DOCUMENTS
#  These are YOUR project-specific documents that the LLM
#  does NOT know about from its training data.
# ════════════════════════════════════════════════════════════

RAW_DOCUMENTS = [

    # ── Agent 1 ────────────────────────────────────────────
    {
        "title": "Resume Analyzer Agent",
        "doc_type": "agent_description",
        "department": "core_agents",
        "priority_level": "high",
        "last_updated": "2025-01-15",
        "content": """
        Resume Analyzer Agent – Role and Responsibilities

        The Resume Analyzer Agent is the entry-point agent in the multi-agent pipeline.
        Its primary responsibility is to parse an uploaded resume and produce a structured
        analysis that all downstream agents can act upon.

        Core Responsibilities:
        1. Extract key resume sections: Contact Info, Summary, Skills, Work Experience,
           Education, Certifications, Projects, and Awards.
        2. Identify weak or missing sections such as a resume with no Summary or no
           quantifiable achievements in the Experience section.
        3. Detect skill entities including programming languages, frameworks, tools,
           and soft skills using pattern matching and LLM analysis.
        4. Return a structured output with normalized skill tokens for comparison
           by the Job Matching Agent.

        Inputs: Raw resume text string or PDF file converted to text using PyPDF.
        Outputs: Structured dictionary with sections, extracted skills list,
                 weak sections list, word count, and readability score.

        Implementation Notes:
        - Uses LangChain LLMChain with a carefully crafted system prompt.
        - Falls back to regex-based heuristics if confidence is low.
        - Stores results in Shared Memory under key: resume_analysis colon session_id.
        - Typical processing time: 3 to 8 seconds depending on resume length.
        """
    },

    # ── Agent 2 ────────────────────────────────────────────
    {
        "title": "Job Matching Agent",
        "doc_type": "agent_description",
        "department": "core_agents",
        "priority_level": "high",
        "last_updated": "2025-01-15",
        "content": """
        Job Matching Agent – Role and Responsibilities

        The Job Matching Agent receives the structured resume analysis from the
        Resume Analyzer Agent and a raw job description provided by the user.
        It performs semantic and keyword-based matching to quantify alignment.

        Core Responsibilities:
        1. Parse the job description to extract required skills, preferred skills,
           experience level, education requirements, and domain keywords.
        2. Compute a skill match score from 0 to 100 percent by comparing resume
           skills against job required skills using cosine similarity on embeddings.
        3. Produce a skill gap list containing skills present in the job description
           but absent from the resume.
        4. Classify the match into tiers: Excellent at 80 percent or above,
           Good between 60 and 79 percent, Fair between 40 and 59 percent,
           and Poor below 40 percent.

        Inputs: Resume analysis dictionary and raw job description text.
        Outputs: match score float, skill gaps list, matched skills list,
                 tier string, and job description keywords list.

        Implementation Notes:
        - Embedding similarity computed using sentence-transformers locally.
        - Keyword matching uses term frequency weighting for domain specific terms.
        - Results cached in shared memory to avoid redundant computation.
        """
    },

    # ── Agent 3 ────────────────────────────────────────────
    {
        "title": "Resume Improver Agent",
        "doc_type": "agent_description",
        "department": "core_agents",
        "priority_level": "high",
        "last_updated": "2025-01-15",
        "content": """
        Resume Improver Agent – Role and Responsibilities

        The Resume Improver Agent takes the original resume content and the skill gap
        analysis to generate an enhanced version of the resume tailored to the target
        job description.

        Core Responsibilities:
        1. Rewrite bullet points using the STAR method: Situation, Task, Action, Result.
           This emphasizes measurable impact and outcomes rather than just listing duties.
        2. Inject missing job-relevant keywords naturally into the experience and
           skills sections without keyword stuffing.
        3. Strengthen the Professional Summary to mirror the tone and terminology
           of the target job description.
        4. Remove irrelevant or outdated experience that reduces signal to noise ratio.
        5. Ensure consistent tense, formatting, and parallel structure across all
           bullet points throughout the document.

        Inputs: Original resume text, skill gaps list, job description keywords list,
                and matched skills list.
        Outputs: Improved resume text in Markdown format and change log list.

        Example Improvement:
        Before: Worked on backend systems.
        After: Engineered RESTful microservices in Python and FastAPI reducing
               API latency by 35 percent and supporting 50 thousand daily active users.

        Implementation Notes:
        - Uses Reflection pattern: draft, self-critique, finalize in multi-turn chain.
        - Temperature set to 0.3 for controlled but creative rewriting.
        """
    },

    # ── Agent 4 ────────────────────────────────────────────
    {
        "title": "ATS Reviewer Agent",
        "doc_type": "agent_description",
        "department": "core_agents",
        "priority_level": "high",
        "last_updated": "2025-01-15",
        "content": """
        ATS Reviewer Agent – Role and Responsibilities

        The ATS Reviewer Agent performs the final review of the improved resume
        to ensure it passes automated screening systems used by most employers.
        ATS stands for Applicant Tracking System.

        Core Responsibilities:
        1. Check keyword density: target keywords should appear 2 to 4 times.
        2. Flag formatting issues that break ATS parsing: tables, text boxes,
           headers and footers with contact info, non-standard section headings,
           unusual fonts, and embedded images.
        3. Verify section heading compatibility with common ATS systems:
           Taleo, Greenhouse, Workday, and Lever.
        4. Score keyword coverage: keyword matches divided by total job description
           keywords multiplied by 100 gives the coverage percentage.
        5. Generate an ATS Compatibility Report with clear actionable recommendations.

        Inputs: Improved resume text and job description keywords list.
        Outputs: ats score integer from 0 to 100, flagged issues list,
                 recommendations list, and keyword density report dictionary.

        Feedback Loop Trigger:
        If ats score is less than 70, the Agent Orchestrator automatically sends
        the resume back to the Resume Improver Agent for another improvement cycle.
        Maximum number of feedback loop iterations is 3 to prevent infinite loops.
        """
    },

    # ── Architecture ────────────────────────────────────────
    {
        "title": "System Architecture and Agent Workflow",
        "doc_type": "architecture",
        "department": "system_design",
        "priority_level": "high",
        "last_updated": "2025-01-10",
        "content": """
        System Architecture – Agentic AI Resume and Job Application Assistant

        Five High-Level Components:
        1. User Interface built with Streamlit where users upload resumes and
           paste job descriptions and then view the results.
        2. Agent Orchestrator built with LangChain that sequences agent invocations
           and manages the feedback loop between agents.
        3. Specialized AI Agents: Resume Analyzer, Job Matcher, Resume Improver,
           and ATS Reviewer, each implemented as a separate LangChain chain.
        4. Shared Memory built with ChromaDB storing session context and results.
        5. LLM Backend using OpenAI GPT or any compatible open source LLM.

        Agent Execution Order:
        Step 1: User uploads resume and pastes job description in the UI.
        Step 2: Resume Analyzer Agent processes the resume and extracts structure.
        Step 3: Job Matching Agent evaluates alignment with the job description.
        Step 4: Resume Improver Agent refines and rewrites the resume content.
        Step 5: ATS Reviewer Agent performs final compatibility evaluation.
        Step 6: If ATS score is below 70 the feedback loop sends resume back to step 4.
        Step 7: Final improved resume and report displayed to user.

        Memory Architecture:
        Short term memory uses in-session Python dictionary for agent outputs.
        Long term memory uses ChromaDB collection named resume_assistant_kb for
        domain knowledge including agent rules, ATS guidelines, and skill keywords.
        Each session isolated by unique session ID to prevent data leakage.

        The feedback loop is the core agentic behavior in this system.
        It demonstrates autonomous decision making: the Orchestrator evaluates
        the ATS score and decides whether to iterate or finalize.
        Maximum of 3 iterations and early exit when ats score reaches 70.
        """
    },

    # ── ATS Guidelines ──────────────────────────────────────
    {
        "title": "ATS Optimization Guidelines",
        "doc_type": "ats_guidelines",
        "department": "domain_knowledge",
        "priority_level": "high",
        "last_updated": "2025-01-12",
        "content": """
        ATS Optimization Guidelines – Domain Knowledge Base

        An Applicant Tracking System is software used by 98 percent of Fortune 500
        companies to automatically screen resumes before a human recruiter sees them.
        Resumes that fail ATS parsing are automatically rejected and never seen.

        Critical ATS Formatting Rules:
        Use a single column layout because multi-column layouts confuse most ATS software.
        Avoid tables, text boxes, headers, footers, and sidebars completely.
        Use standard fonts such as Arial, Calibri, or Times New Roman in 10 to 12 point.
        Do not use graphics, logos, charts, or images anywhere in the resume.
        Save the file as docx format which is most compatible with all ATS systems.
        Use only standard bullet points and avoid custom symbols or special characters.

        Required Section Headings for ATS Compatibility:
        Work Experience or Professional Experience for the job history section.
        Education for the academic background section.
        Skills or Technical Skills or Core Competencies for the skills section.
        Certifications or Licenses for credentials section.
        Summary or Professional Summary or Profile for the opening statement.
        Avoid creative headings like My Story, Career Journey, or What I Do.

        Keyword Optimization Strategy:
        Mirror the exact keywords from the job description in your resume.
        Include both spelled out and abbreviated forms such as Artificial Intelligence and AI.
        Place high priority keywords in the Skills section AND in experience bullets.
        Target keyword density is 2 to 4 occurrences per important keyword.
        Do not use more than 5 occurrences as this looks like keyword stuffing.

        ATS Score Formula Used in This Project:
        ats score equals keyword matches divided by total job description keywords times 60
        plus formatting score times 30
        plus section completeness times 10.
        The passing threshold is ats score greater than or equal to 70.
        When ats score is below 70 the feedback loop triggers another improvement cycle.
        When ats score reaches 70 or above the system exits the feedback loop completely.

        Common ATS Platforms:
        Taleo by Oracle is strict on formatting and prefers docx over pdf format.
        Greenhouse has good PDF support and handles moderate document complexity.
        Workday prefers docx format and auto-parses sections very well.
        Lever is the most modern parser and most forgiving of formatting choices.
        iCIMS is conservative and works best with plain simple formatting only.
        """
    },

    # ── Skill Keywords ──────────────────────────────────────
    {
        "title": "Industry Skill Keywords Technology Domain",
        "doc_type": "skill_keywords",
        "department": "domain_knowledge",
        "priority_level": "medium",
        "last_updated": "2025-01-12",
        "content": """
        Industry Skill Keywords for Resume Matching and Gap Analysis

        Programming Languages:
        Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust, Ruby, PHP,
        Swift, Kotlin, Scala, R, MATLAB, Bash, PowerShell, SQL, NoSQL.

        Artificial Intelligence and Machine Learning:
        Machine Learning, Deep Learning, Neural Networks, Natural Language Processing,
        Computer Vision, Reinforcement Learning, Transfer Learning, Fine-tuning,
        Retrieval Augmented Generation, RAG, Agentic AI, Multi-Agent Systems,
        LangChain, LlamaIndex, OpenAI API, Hugging Face, TensorFlow, PyTorch,
        Scikit-learn, XGBoost, FAISS, ChromaDB, Pinecone, Sentence Transformers,
        Prompt Engineering, Vector Databases, Embeddings, Large Language Models.

        Web Development:
        React, Next.js, Angular, Vue.js, Django, FastAPI, Flask, Node.js,
        Express.js, REST API, GraphQL, WebSockets, HTML5, CSS3, Tailwind CSS.

        Databases:
        PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Cassandra,
        DynamoDB, Firebase, SQLite, Snowflake, BigQuery.

        Cloud and DevOps:
        AWS, Azure, GCP, Docker, Kubernetes, CI/CD, GitHub Actions, Jenkins,
        Terraform, Ansible, Linux, Nginx, Microservices, Load Balancing.

        Soft Skills Important for ATS:
        Problem Solving, Team Collaboration, Communication, Agile, Scrum,
        Leadership, Project Management, Time Management, Critical Thinking,
        Stakeholder Management, Technical Writing, Code Review, Mentoring.

        Data Science Tools:
        Data Analysis, Data Visualization, Pandas, NumPy, Matplotlib, Seaborn,
        Power BI, Tableau, A/B Testing, Statistical Analysis, Data Cleaning,
        Feature Engineering, Model Deployment, MLOps, Jupyter Notebook.
        """
    },

    # ── Evaluation ──────────────────────────────────────────
    {
        "title": "Evaluation Metrics and Success Criteria",
        "doc_type": "evaluation",
        "department": "quality_assurance",
        "priority_level": "medium",
        "last_updated": "2025-01-14",
        "content": """
        Evaluation Metrics – Agentic AI Resume and Job Application Assistant

        Metric 1: Resume Job Match Score (Primary Metric)
        Computed as cosine similarity between resume embedding and job description
        embedding multiplied by 100 to get a percentage score.
        Target improvement is at least 15 percentage points after the system runs.
        Baseline is measured from the unmodified original resume.

        Metric 2: ATS Keyword Coverage Score
        Formula divides keywords present in resume by total keywords in job description
        and multiplies by 100 to get a percentage.
        Target is 70 percent or higher coverage after ATS Reviewer optimization.
        Measured both before and after the Resume Improver runs.

        Metric 3: Improvement Quality Before and After Comparison
        Quantitative: word count change, bullet point count, keyword density increase,
        and STAR method compliance percentage.
        STAR method compliance is the percentage of bullets containing measurable outcomes.
        Qualitative: human review of readability, relevance, and impact.
        Clarity score uses Flesch-Kincaid readability index with target between 60 and 70.

        Metric 4: User Satisfaction
        Collected via Streamlit feedback widget with 1 to 5 star rating.
        Session completion rate measures whether user downloaded the improved resume.

        Benchmark Comparison:
        Single agent baseline uses a single LLM prompt: improve my resume.
        Multi-agent system uses the full four-agent pipeline with feedback loop.
        The multi-agent approach provides structured, explainable, iterative improvements
        compared to the one-shot monolithic single-agent approach.

        Iteration Tracking:
        Each feedback loop iteration logs: iteration number, ats score,
        match score, and list of changes made.
        Maximum iterations is 3. Early exit when ats score reaches 70 or above.
        """
    },

    # ── Problem Statement ───────────────────────────────────
    {
        "title": "Problem Statement and Project Motivation",
        "doc_type": "project_overview",
        "department": "management",
        "priority_level": "low",
        "last_updated": "2025-01-05",
        "content": """
        Problem Statement – Why This Project Was Built

        Challenge 1: ATS Rejection is the Primary Pain Point.
        75 percent of resumes are rejected by ATS before a human ever sees them.
        Most candidates do not understand why their resume was rejected.
        The ATS Reviewer Agent solves this by providing transparent actionable feedback.

        Challenge 2: Skill Gap Blindness.
        Applicants often do not know which specific skills they are missing for a role.
        Generic resume templates do not adapt to job-specific requirements.
        The Job Matching Agent identifies exact skill gaps with detailed evidence.

        Challenge 3: Weak Resume Writing.
        Most candidates write weak duty-focused bullets instead of impact-focused ones.
        Weak example: Responsible for managing a team.
        Strong example: Led a team of 8 engineers to deliver a 2 million dollar
        project 3 weeks ahead of schedule saving the company 120 thousand dollars.
        The Resume Improver Agent rewrites bullets using STAR methodology.

        Challenge 4: Time and Financial Cost.
        Professional resume writing services cost between 150 and 500 dollars.
        Average job seeker applies to 50 or more positions each requiring tailoring.
        This system automates per-job-description tailoring in under 2 minutes for free.

        Why Use Agentic AI Instead of a Single Prompt?
        Single-prompt LLM approaches produce generic and unreliable improvements.
        Agentic AI enables specialization where each agent masters exactly one task.
        Feedback loops allow iterative refinement until a quality threshold is met.
        Each agent produces auditable structured output improving explainability.
        Modularity allows each agent to be updated or replaced independently.
        """
    },
]


# ════════════════════════════════════════════════════════════
#  TASK 1 – TEXT CLEANING
# ════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Strip domain-specific noise: HTML, page numbers, URLs,
    control characters, excessive whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)                          # HTML tags
    text = re.sub(r"page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"confidential|proprietary", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)                     # URLs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)        # control chars
    text = re.sub(r"[ \t]+", " ", text)                           # multi-spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                        # multi-newlines
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


# ════════════════════════════════════════════════════════════
#  TASK 2a – SEMANTIC CHUNKING
# ════════════════════════════════════════════════════════════

def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split text respecting paragraph and sentence boundaries.
    Keeps full bullet points and rules together in one chunk.
    Adds overlap between chunks to preserve context.
    """
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks, current, prev_tail = [], "", ""

    for para in paragraphs:
        if len(para) > chunk_size:                    # large paragraph → split by sentence
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sent) <= chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        chunks.append(prev_tail + current)
                        prev_tail = current[-overlap:] + " " if len(current) > overlap else ""
                    current = sent
        else:
            if len(current) + len(para) <= chunk_size:
                current += ("\n" if current else "") + para
            else:
                if current:
                    chunks.append(prev_tail + current)
                    prev_tail = current[-overlap:] + " " if len(current) > overlap else ""
                current = para

    if current.strip():
        chunks.append(prev_tail + current)

    return chunks


# ════════════════════════════════════════════════════════════
#  TASK 2b – LOCAL EMBEDDING (FREE, NO API KEY)
# ════════════════════════════════════════════════════════════

def get_embeddings(texts: list) -> list:
    """Embed using local sentence-transformers model. Returns list of vectors."""
    vectors = embedder.encode(texts, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════
#  METADATA BUILDER – Minimum 3 searchable tags (lab requirement)
# ════════════════════════════════════════════════════════════

def build_metadata(doc: dict, chunk_index: int, total_chunks: int,
                   chunk_text: str) -> dict:
    return {
        "doc_type":       doc["doc_type"],        # Tag 1 (required)
        "department":     doc["department"],       # Tag 2 (required)
        "priority_level": doc["priority_level"],   # Tag 3 (required)
        "title":          doc["title"],            # bonus
        "last_updated":   doc["last_updated"],     # bonus
        "chunk_index":    chunk_index,             # bonus
        "total_chunks":   total_chunks,            # bonus
        "char_count":     len(chunk_text),         # bonus
        "chunk_hash":     hashlib.md5(chunk_text.encode()).hexdigest()[:12],  # bonus
    }


# ════════════════════════════════════════════════════════════
#  TASK 3 – CHROMADB VECTOR INDEXING
# ════════════════════════════════════════════════════════════

def init_chroma():
    print(f"[ChromaDB] Initializing at: {CHROMA_PATH}")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[ChromaDB] Collection '{COLLECTION_NAME}' ready.")
    print(f"           Existing chunks: {collection.count()}\n")
    return collection


def upsert_chunks(collection, ids, embeddings, documents, metadatas):
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids        = ids[i:i+BATCH],
            embeddings = embeddings[i:i+BATCH],
            documents  = documents[i:i+BATCH],
            metadatas  = metadatas[i:i+BATCH],
        )
    print(f"  ↳ Upserted {len(ids)} chunks into ChromaDB.")


# ════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 60)
    print("  Lab 2 – Data Ingestion Pipeline")
    print("  Agentic AI Resume & Job Application Assistant")
    print("  Using: Local HuggingFace embeddings (NO API KEY)")
    print("=" * 60 + "\n")

    collection = init_chroma()
    all_ids, all_embs, all_docs, all_metas = [], [], [], []

    for idx, doc in enumerate(RAW_DOCUMENTS):
        print(f"[{idx+1}/{len(RAW_DOCUMENTS)}] '{doc['title']}'")

        cleaned    = clean_text(doc["content"])
        chunks     = semantic_chunk(cleaned)
        embeddings = get_embeddings(chunks)

        print(f"  Cleaned: {len(cleaned)} chars | Chunks: {len(chunks)} | Embedded: {len(embeddings)}")

        for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            all_ids.append(f"{doc['doc_type']}__{idx:02d}__{ci:03d}")
            all_embs.append(emb)
            all_docs.append(chunk)
            all_metas.append(build_metadata(doc, ci, len(chunks), chunk))

    print(f"\n[ChromaDB] Loading {len(all_ids)} total chunks...")
    upsert_chunks(collection, all_ids, all_embs, all_docs, all_metas)

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents  : {len(RAW_DOCUMENTS)}")
    print(f"  Chunks     : {len(all_ids)}")
    print(f"  DB total   : {collection.count()}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Path       : {os.path.abspath(CHROMA_PATH)}")
    print("=" * 60)

    # Quick sanity test
    print("\n[Sanity Test] Verifying retrieval works...")
    q   = "How does the ATS Reviewer Agent check keyword density?"
    res = collection.query(
        query_embeddings=[get_embeddings([q])[0]],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )
    print(f'Query: "{q}"\n')
    for i, (text, meta, dist) in enumerate(zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0])):
        print(f"  #{i+1} | {meta['title']} | dist={dist:.4f}")
        print(f"       {text[:120].strip()}...\n")

    print("✅ Done! Now run:  python retrieval_test.py")


# ════════════════════════════════════════════════════════════
#  RETRIEVAL HELPER (imported by retrieval_test.py)
# ════════════════════════════════════════════════════════════

def retrieve(query: str, n: int = 3, filter_meta: dict = None) -> list:
    """
    Retrieve top-n relevant chunks for a query.
    Optionally filter by metadata field.
    Returns list of {text, metadata, distance}.
    """
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    q_emb      = get_embeddings([query])[0]
    results    = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        where=filter_meta,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"text": t, "metadata": m, "distance": round(d, 4)}
        for t, m, d in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


if __name__ == "__main__":
    run_pipeline()