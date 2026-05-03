"""
=============================================================
  Lab 3 – The Reasoning Loop (LangGraph)
  File: tools.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

TASK 1: Tool Engineering with Pydantic

All tools use:
- @tool decorator from langchain_core.tools
- Pydantic models for strict input validation
- Descriptive docstrings (LLM uses these to decide when to call)

NO API KEY REQUIRED - Uses local models from Lab 2.
=============================================================
"""

import re
import chromadb
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# ── Configuration (must match Lab 2) ────────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "resume_assistant_kb"

# Load embedding model once at module import
print("[Tools] Loading sentence-transformers model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("[Tools] Model loaded.\n")


# ════════════════════════════════════════════════════════════
#  TOOL 1: Grounding Tool (Vector DB Retrieval)
# ════════════════════════════════════════════════════════════

class KnowledgeQueryInput(BaseModel):
    """Input schema for querying the knowledge base."""
    query: str = Field(
        description="The question or search query to find information in the knowledge base. "
                    "Examples: 'What are ATS formatting rules?', 'How does the Resume Improver work?'"
    )
    n_results: int = Field(
        default=3,
        description="Number of relevant chunks to retrieve (default 3, max 5)",
        ge=1,
        le=5
    )


@tool(args_schema=KnowledgeQueryInput)
def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Query the Resume Assistant knowledge base built in Lab 2.
    
    Use this tool when you need information about:
    - ATS (Applicant Tracking System) rules and formatting guidelines
    - How the Resume Analyzer, Job Matcher, Resume Improver, or ATS Reviewer agents work
    - Industry skill keywords for technology domain
    - System architecture and agent workflow
    - Evaluation metrics and success criteria
    
    This tool retrieves relevant chunks from the vector database using semantic search.
    
    Args:
        query: The search question or topic
        n_results: Number of results to return (1-5)
    
    Returns:
        String containing the top relevant knowledge chunks with their metadata.
    """
    try:
        # Embed the query
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
        
        # Connect to ChromaDB
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        
        # Retrieve
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        if not results["documents"][0]:
            return "No relevant information found in the knowledge base."
        
        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            output.append(
                f"[Result {i}]\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Type: {meta.get('doc_type', 'Unknown')}\n"
                f"Relevance Score: {1 - dist:.2f}\n"
                f"Content:\n{doc}\n"
            )
        
        return "\n---\n".join(output)
    
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


# ════════════════════════════════════════════════════════════
#  TOOL 2: Extract Skills from Resume Text
# ════════════════════════════════════════════════════════════

class ExtractSkillsInput(BaseModel):
    """Input schema for extracting skills from resume text."""
    resume_text: str = Field(
        description="The raw text content of the resume from which to extract skills"
    )


@tool(args_schema=ExtractSkillsInput)
def extract_skills_from_resume(resume_text: str) -> str:
    """
    Extract technical and soft skills from resume text.
    
    Use this tool when you have a resume and need to identify what skills
    the candidate has listed. This is the first step in the resume analysis pipeline.
    
    The tool looks for:
    - Programming languages (Python, Java, JavaScript, etc.)
    - Frameworks and libraries (React, Django, TensorFlow, etc.)
    - Tools and platforms (Docker, AWS, Git, etc.)
    - Soft skills (Leadership, Communication, etc.)
    - Certifications and domain knowledge
    
    Args:
        resume_text: The full text content of the resume
    
    Returns:
        JSON string containing categorized skills found in the resume.
    """
    # Skill pattern database (simplified - in production this would be more comprehensive)
    skill_patterns = {
        "Programming Languages": [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
            "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "SQL"
        ],
        "AI/ML": [
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow",
            "PyTorch", "Scikit-learn", "Keras", "OpenCV", "Hugging Face", "LangChain",
            "RAG", "Vector Database", "Embeddings", "Fine-tuning", "LLM"
        ],
        "Web Development": [
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "HTML", "CSS", "REST API", "GraphQL", "Next.js"
        ],
        "Cloud & DevOps": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins",
            "Terraform", "Ansible", "Linux"
        ],
        "Databases": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "ChromaDB",
            "Pinecone", "FAISS"
        ],
        "Soft Skills": [
            "Leadership", "Communication", "Problem Solving", "Team Collaboration",
            "Project Management", "Agile", "Scrum"
        ]
    }
    
    found_skills = {category: [] for category in skill_patterns.keys()}
    resume_lower = resume_text.lower()
    
    # Extract skills
    for category, skills in skill_patterns.items():
        for skill in skills:
            # Case-insensitive search
            if skill.lower() in resume_lower:
                found_skills[category].append(skill)
    
    # Count total
    total_skills = sum(len(skills) for skills in found_skills.values())
    
    # Format output
    output = [f"Extracted {total_skills} skills from resume:\n"]
    for category, skills in found_skills.items():
        if skills:
            output.append(f"{category}: {', '.join(skills)}")
    
    if total_skills == 0:
        return "No recognizable skills found in the resume. The resume may need better formatting or skill keywords."
    
    return "\n".join(output)


# ════════════════════════════════════════════════════════════
#  TOOL 3: Calculate Skill Match Score
# ════════════════════════════════════════════════════════════

class CalculateMatchInput(BaseModel):
    """Input schema for calculating resume-job match score."""
    resume_skills: str = Field(
        description="Comma-separated list of skills from the resume. "
                    "Example: 'Python, Machine Learning, Docker, AWS'"
    )
    job_skills: str = Field(
        description="Comma-separated list of required skills from job description. "
                    "Example: 'Python, TensorFlow, Kubernetes, PostgreSQL'"
    )


@tool(args_schema=CalculateMatchInput)
def calculate_skill_match_score(resume_skills: str, job_skills: str) -> str:
    """
    Calculate how well resume skills match job requirements.
    
    Use this tool after extracting skills from both the resume and job description.
    It computes:
    - Match percentage (how many job skills are present in resume)
    - Missing skills (what the candidate lacks)
    - Extra skills (what the candidate has beyond requirements)
    
    The match score is critical for deciding whether to proceed with resume improvement
    or to inform the candidate of their current standing.
    
    Args:
        resume_skills: Comma-separated skills from resume
        job_skills: Comma-separated required skills from job
    
    Returns:
        Detailed match analysis with score, matched skills, missing skills, and tier.
    """
    # Parse skills
    resume_set = {s.strip().lower() for s in resume_skills.split(",") if s.strip()}
    job_set    = {s.strip().lower() for s in job_skills.split(",") if s.strip()}
    
    # Calculate matches
    matched_skills = resume_set.intersection(job_set)
    missing_skills = job_set - resume_set
    extra_skills   = resume_set - job_set
    
    # Match score
    if len(job_set) == 0:
        match_score = 0.0
    else:
        match_score = (len(matched_skills) / len(job_set)) * 100
    
    # Tier classification
    if match_score >= 80:
        tier = "Excellent"
    elif match_score >= 60:
        tier = "Good"
    elif match_score >= 40:
        tier = "Fair"
    else:
        tier = "Poor"
    
    # Format output
    output = [
        f"Skill Match Analysis:",
        f"",
        f"Match Score: {match_score:.1f}% ({tier})",
        f"",
        f"Matched Skills ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}",
        f"",
        f"Missing Skills ({len(missing_skills)}): {', '.join(missing_skills) if missing_skills else 'None'}",
        f"",
        f"Extra Skills ({len(extra_skills)}): {', '.join(list(extra_skills)[:10]) if extra_skills else 'None'}",
    ]
    
    return "\n".join(output)


# ════════════════════════════════════════════════════════════
#  TOOL 4: Calculate ATS Compatibility Score
# ════════════════════════════════════════════════════════════

class ATSScoreInput(BaseModel):
    """Input schema for ATS scoring."""
    resume_text: str = Field(
        description="The full text of the resume to score for ATS compatibility"
    )
    job_keywords: str = Field(
        description="Comma-separated list of important keywords from the job description. "
                    "Example: 'Python, AWS, Machine Learning, Agile'"
    )


@tool(args_schema=ATSScoreInput)
def calculate_ats_score(resume_text: str, job_keywords: str) -> str:
    """
    Calculate ATS (Applicant Tracking System) compatibility score for a resume.
    
    Use this tool to evaluate whether a resume will pass automated screening systems
    used by 98% of Fortune 500 companies. This is CRITICAL because resumes with low
    ATS scores are automatically rejected before any human sees them.
    
    The score is based on:
    - Keyword coverage (60%): How many job keywords appear in the resume
    - Formatting quality (30%): Checks for ATS-friendly formatting
    - Section completeness (10%): Presence of required sections
    
    Passing threshold: 70 or above
    Below 70: Resume needs improvement
    
    Args:
        resume_text: Full resume content
        job_keywords: Important keywords from job description
    
    Returns:
        ATS score (0-100), keyword coverage details, and formatting issues.
    """
    resume_lower = resume_text.lower()
    keywords = [k.strip().lower() for k in job_keywords.split(",") if k.strip()]
    
    # ── Component 1: Keyword Coverage (60 points max) ──────
    matched_keywords = []
    keyword_density = {}
    
    for keyword in keywords:
        count = resume_lower.count(keyword)
        if count > 0:
            matched_keywords.append(keyword)
            keyword_density[keyword] = count
    
    if len(keywords) == 0:
        keyword_score = 0
    else:
        keyword_score = (len(matched_keywords) / len(keywords)) * 60
    
    # ── Component 2: Formatting Check (30 points max) ──────
    formatting_issues = []
    formatting_score = 30
    
    # Check for problematic patterns
    if re.search(r'\│|\║|\|', resume_text):  # Tables/columns
        formatting_issues.append("Contains table characters (may break ATS)")
        formatting_score -= 10
    
    if len(resume_text.split('\n')) < 10:
        formatting_issues.append("Resume too short or improperly formatted")
        formatting_score -= 10
    
    # Check for standard section headings
    required_sections = ["experience", "education", "skills"]
    for section in required_sections:
        if section not in resume_lower:
            formatting_issues.append(f"Missing '{section}' section heading")
            formatting_score -= 5
    
    formatting_score = max(0, formatting_score)
    
    # ── Component 3: Section Completeness (10 points max) ──
    completeness_score = 10  # Assume complete for now
    
    # ── Total ATS Score ─────────────────────────────────────
    total_score = keyword_score + formatting_score + completeness_score
    
    # ── Status ──────────────────────────────────────────────
    if total_score >= 70:
        status = "PASS - Resume is ATS-friendly"
    else:
        status = "NEEDS IMPROVEMENT - Resume may be rejected by ATS"
    
    # ── Format Output ───────────────────────────────────────
    output = [
        f"ATS Compatibility Score: {total_score:.0f}/100",
        f"Status: {status}",
        f"",
        f"Breakdown:",
        f"  Keyword Coverage: {keyword_score:.1f}/60 ({len(matched_keywords)}/{len(keywords)} keywords found)",
        f"  Formatting: {formatting_score:.1f}/30",
        f"  Completeness: {completeness_score:.1f}/10",
        f"",
        f"Matched Keywords: {', '.join(matched_keywords) if matched_keywords else 'None'}",
    ]
    
    if formatting_issues:
        output.append(f"\nFormatting Issues:")
        for issue in formatting_issues:
            output.append(f"  - {issue}")
    
    if total_score < 70:
        output.append(f"\nRecommendation: Use Resume Improver to increase keyword coverage and fix formatting.")
    
    return "\n".join(output)


# ════════════════════════════════════════════════════════════
#  TOOL 5: Generate Resume Improvement Suggestions
# ════════════════════════════════════════════════════════════

class ImprovementInput(BaseModel):
    """Input schema for generating improvement suggestions."""
    missing_skills: str = Field(
        description="Comma-separated list of skills the resume is missing. "
                    "Example: 'Docker, Kubernetes, CI/CD'"
    )
    ats_score: float = Field(
        description="The current ATS score (0-100)",
        ge=0,
        le=100
    )


@tool(args_schema=ImprovementInput)
def generate_improvement_suggestions(missing_skills: str, ats_score: float) -> str:
    """
    Generate actionable suggestions to improve a resume based on skill gaps and ATS score.
    
    Use this tool after calculating skill match and ATS scores. It provides:
    - Specific actions to address missing skills
    - Keyword injection strategies
    - Formatting improvements
    - STAR method bullet point templates
    
    These suggestions are what the Resume Improver Agent uses to rewrite the resume.
    
    Args:
        missing_skills: Comma-separated skills to add
        ats_score: Current ATS compatibility score
    
    Returns:
        Detailed, actionable improvement plan.
    """
    suggestions = []
    skills = [s.strip() for s in missing_skills.split(",") if s.strip()]
    
    # ── Skill Gap Suggestions ───────────────────────────────
    if skills:
        suggestions.append("MISSING SKILLS - Add these to your resume:")
        for skill in skills[:5]:  # Top 5 most important
            suggestions.append(
                f"  • {skill}: Review your past work. Did you use {skill} even minimally? "
                f"If yes, add it to your Skills section AND mention it in a relevant bullet point."
            )
        suggestions.append("")
    
    # ── ATS Score-Specific Suggestions ──────────────────────
    if ats_score < 70:
        suggestions.append("ATS IMPROVEMENTS (Score < 70):")
        suggestions.append("  • Increase keyword density: Mention top 5 job keywords 2-3 times each")
        suggestions.append("  • Use standard section headings: 'Professional Experience', 'Education', 'Skills'")
        suggestions.append("  • Remove tables, text boxes, and graphics")
        suggestions.append("  • Add a 'Professional Summary' with 3-4 job-specific keywords")
        suggestions.append("")
    
    # ── Bullet Point Improvement ────────────────────────────
    suggestions.append("BULLET POINT REWRITING - Use STAR Method:")
    suggestions.append("  Template: [Action Verb] + [What You Did] + [How] + [Measurable Result]")
    suggestions.append("  Example: 'Engineered RESTful APIs using Python/FastAPI, reducing response time by 40% and supporting 100K daily requests'")
    suggestions.append("")
    
    # ── Keyword Injection Strategy ──────────────────────────
    suggestions.append("KEYWORD INJECTION STRATEGY:")
    suggestions.append("  1. Skills Section: List missing skills naturally")
    suggestions.append("  2. Experience Bullets: Weave 1-2 keywords into each accomplishment")
    suggestions.append("  3. Professional Summary: Use 3-4 high-priority keywords in opening paragraph")
    suggestions.append("")
    
    return "\n".join(suggestions)


# ════════════════════════════════════════════════════════════
#  EXPORT: List of All Tools
# ════════════════════════════════════════════════════════════

# This list is imported by graph.py
ALL_TOOLS = [
    query_knowledge_base,
    extract_skills_from_resume,
    calculate_skill_match_score,
    calculate_ats_score,
    generate_improvement_suggestions,
]

if __name__ == "__main__":
    print("=" * 60)
    print("  Lab 3 Tools - Loaded Successfully")
    print("=" * 60)
    print(f"Total Tools: {len(ALL_TOOLS)}")
    for i, tool in enumerate(ALL_TOOLS, 1):
        print(f"  {i}. {tool.name}")
    print("=" * 60)