"""
=============================================================
  AI407L Mid-Exam - Part A (FINAL VERSION)
  File: tools.py
  
  Updated with ACTUAL email sending via Gmail SMTP
=============================================================
"""

import re
import chromadb
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "resume_assistant_kb"

# Gmail SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Load embedding model
print("[Tools] Loading sentence-transformers model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("[Tools] Model loaded.\n")


# ════════════════════════════════════════════════════════════
#  LAB 2: THE "GROUNDING" TOOL
# ════════════════════════════════════════════════════════════

class KnowledgeQueryInput(BaseModel):
    """Input schema for querying the knowledge base."""
    query: str = Field(description="The question to search for in the knowledge base")
    n_results: int = Field(default=3, ge=1, le=5, description="Number of results (1-5)")


@tool(args_schema=KnowledgeQueryInput)
def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Query the Resume Assistant knowledge base using RAG.
    
    LAB 2 REQUIREMENT: The "Grounding" Tool
    
    Use when you need domain knowledge about:
    - ATS formatting rules and guidelines
    - Resume improvement best practices
    - Industry skill keywords
    - Email writing best practices
    
    Args:
        query: Search query
        n_results: Number of results (1-5)
    
    Returns:
        Relevant knowledge chunks with metadata
    """
    try:
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
        
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            return "No relevant information found in knowledge base."
        
        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            output.append(
                f"[Knowledge Source {i}]\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Relevance: {(1 - dist) * 100:.1f}%\n"
                f"Content:\n{doc}\n"
            )
        
        return "\n---\n".join(output)
    
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


# ════════════════════════════════════════════════════════════
#  LAB 3: "ACTION" TOOLS
# ════════════════════════════════════════════════════════════

class ExtractSkillsInput(BaseModel):
    """Pydantic schema for skill extraction input."""
    resume_text: str = Field(description="The full text content of the resume to analyze")


@tool(args_schema=ExtractSkillsInput)
def extract_skills_from_resume(resume_text: str) -> str:
    """
    Extract technical and soft skills from a resume.
    
    LAB 3 REQUIREMENT: Action tool with Pydantic validation.
    LAB 4 REQUIREMENT: Assigned to Researcher agent only.
    
    Args:
        resume_text: Complete resume content
    
    Returns:
        Structured list of extracted skills by category
    """
    skill_patterns = {
        "Programming Languages": [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
            "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "SQL"
        ],
        "AI/ML": [
            "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch",
            "Scikit-learn", "LangChain", "RAG", "Vector Database", "LLM"
        ],
        "Web Development": [
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "HTML", "CSS", "REST API", "GraphQL"
        ],
        "Cloud & DevOps": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins"
        ],
        "Databases": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "ChromaDB", "Elasticsearch"
        ],
        "Soft Skills": [
            "Leadership", "Communication", "Problem Solving", "Team Collaboration"
        ]
    }
    
    found_skills = {category: [] for category in skill_patterns.keys()}
    resume_lower = resume_text.lower()
    
    for category, skills in skill_patterns.items():
        for skill in skills:
            if skill.lower() in resume_lower:
                found_skills[category].append(skill)
    
    total = sum(len(skills) for skills in found_skills.values())
    
    if total == 0:
        return "No recognizable skills found in resume."
    
    output = [f"Extracted {total} skills from resume:\n"]
    for category, skills in found_skills.items():
        if skills:
            output.append(f"{category}: {', '.join(skills)}")
    
    return "\n".join(output)


class SkillMatchInput(BaseModel):
    """Pydantic schema for skill matching."""
    resume_skills: str = Field(description="Comma-separated list of skills from resume")
    job_skills: str = Field(description="Comma-separated list of required skills from job")


@tool(args_schema=SkillMatchInput)
def calculate_skill_match_score(resume_skills: str, job_skills: str) -> str:
    """
    Calculate match percentage between resume and job requirements.
    
    LAB 3 REQUIREMENT: Action tool with Pydantic validation.
    LAB 4 REQUIREMENT: Assigned to Analyst agent only.
    
    Args:
        resume_skills: Skills from resume (comma-separated)
        job_skills: Required skills from job (comma-separated)
    
    Returns:
        Detailed match analysis with score and recommendations
    """
    resume_set = {s.strip().lower() for s in resume_skills.split(",") if s.strip()}
    job_set    = {s.strip().lower() for s in job_skills.split(",") if s.strip()}
    
    if not job_set:
        return "Error: No job skills provided for comparison."
    
    matched_skills = resume_set.intersection(job_set)
    missing_skills = job_set - resume_set
    extra_skills   = resume_set - job_set
    
    match_score = (len(matched_skills) / len(job_set)) * 100
    
    tier = "Excellent" if match_score >= 80 else "Good" if match_score >= 60 else "Fair" if match_score >= 40 else "Poor"
    
    return (
        f"Skill Match Analysis:\n"
        f"\n"
        f"Match Score: {match_score:.1f}% ({tier})\n"
        f"\n"
        f"Matched Skills ({len(matched_skills)}): {', '.join(sorted(matched_skills)) if matched_skills else 'None'}\n"
        f"\n"
        f"Missing Skills ({len(missing_skills)}): {', '.join(sorted(missing_skills)) if missing_skills else 'None'}\n"
        f"\n"
        f"Additional Skills ({len(extra_skills)}): {', '.join(sorted(list(extra_skills)[:10])) if extra_skills else 'None'}"
    )


class ATSScoreInput(BaseModel):
    """Pydantic schema for ATS scoring."""
    resume_text: str = Field(description="Full resume text to evaluate")
    job_keywords: str = Field(description="Comma-separated keywords from job description")


@tool(args_schema=ATSScoreInput)
def calculate_ats_score(resume_text: str, job_keywords: str) -> str:
    """
    Calculate ATS (Applicant Tracking System) compatibility score.
    
    LAB 3 REQUIREMENT: Action tool performing calculations.
    LAB 4 REQUIREMENT: Assigned to Analyst agent only.
    
    Formula: keyword_coverage×60 + formatting×30 + completeness×10
    Passing threshold: 70/100
    
    Args:
        resume_text: Complete resume content
        job_keywords: Important keywords from job posting
    
    Returns:
        ATS score, breakdown, and recommendations
    """
    resume_lower = resume_text.lower()
    keywords = [k.strip().lower() for k in job_keywords.split(",") if k.strip()]
    
    matched_keywords = []
    for keyword in keywords:
        if keyword in resume_lower:
            matched_keywords.append(keyword)
    
    keyword_score = (len(matched_keywords) / len(keywords) * 60) if keywords else 0
    
    formatting_score = 30
    formatting_issues = []
    
    if re.search(r'\│|\║|\|', resume_text):
        formatting_issues.append("Contains table characters")
        formatting_score -= 10
    
    if "experience" not in resume_lower:
        formatting_issues.append("Missing 'Experience' section")
        formatting_score -= 5
    if "skills" not in resume_lower:
        formatting_issues.append("Missing 'Skills' section")
        formatting_score -= 5
    if "education" not in resume_lower:
        formatting_issues.append("Missing 'Education' section")
        formatting_score -= 5
    
    formatting_score = max(0, formatting_score)
    completeness_score = 10
    total_score = keyword_score + formatting_score + completeness_score
    
    status = "PASS - Resume is ATS-friendly" if total_score >= 70 else "NEEDS IMPROVEMENT"
    
    output = [
        f"ATS Compatibility Score: {total_score:.0f}/100",
        f"Status: {status}",
        f"",
        f"Breakdown:",
        f"  Keyword Coverage: {keyword_score:.1f}/60 ({len(matched_keywords)}/{len(keywords)} keywords found)",
        f"  Formatting Quality: {formatting_score}/30",
        f"  Section Completeness: {completeness_score}/10",
        f"",
        f"Matched Keywords: {', '.join(matched_keywords) if matched_keywords else 'None'}"
    ]
    
    if formatting_issues:
        output.append(f"\nFormatting Issues:")
        for issue in formatting_issues:
            output.append(f"  - {issue}")
    
    if total_score < 70:
        output.append(f"\nRecommendation: Resume needs improvement to pass ATS screening.")
    
    return "\n".join(output)


class ImprovementInput(BaseModel):
    """Pydantic schema for improvement suggestions."""
    missing_skills: str = Field(description="Comma-separated list of missing skills")
    ats_score: float = Field(ge=0, le=100, description="Current ATS score")


@tool(args_schema=ImprovementInput)
def generate_improvement_suggestions(missing_skills: str, ats_score: float) -> str:
    """
    Generate actionable resume improvement recommendations.
    
    LAB 3 REQUIREMENT: Action tool with Pydantic validation.
    LAB 4 REQUIREMENT: Assigned to Writer agent only.
    
    Args:
        missing_skills: Skills to add (comma-separated)
        ats_score: Current ATS score (0-100)
    
    Returns:
        Detailed improvement plan with specific actions
    """
    suggestions = []
    skills = [s.strip() for s in missing_skills.split(",") if s.strip()]
    
    if skills:
        suggestions.append("MISSING SKILLS - Priority Actions:")
        for skill in skills[:5]:
            suggestions.append(
                f"  • {skill}: Review past projects. If you've used {skill}, add it to "
                f"Skills section AND mention in a relevant bullet point."
            )
        suggestions.append("")
    
    if ats_score < 70:
        suggestions.append("ATS OPTIMIZATION (Score < 70):")
        suggestions.append("  • Increase keyword density: Top 5 job keywords should appear 2-3 times each")
        suggestions.append("  • Use standard section headings: 'Professional Experience', 'Education', 'Skills'")
        suggestions.append("  • Remove tables, columns, and graphics")
        suggestions.append("  • Add 'Professional Summary' section with 3-4 key job keywords")
        suggestions.append("")
    
    suggestions.append("BULLET POINT REWRITING - Use STAR Method:")
    suggestions.append("  Template: [Action Verb] + [What You Did] + [How/Tools] + [Measurable Result]")
    suggestions.append("  ")
    suggestions.append("  Before: 'Worked on backend development'")
    suggestions.append("  After:  'Engineered RESTful APIs using Python/FastAPI, reducing response")
    suggestions.append("           time by 40% and supporting 100K daily requests'")
    suggestions.append("")
    
    suggestions.append("KEYWORD INJECTION STRATEGY:")
    suggestions.append("  1. Skills Section: List all missing skills you actually possess")
    suggestions.append("  2. Experience Bullets: Weave 1-2 keywords into each accomplishment")
    suggestions.append("  3. Professional Summary: Use 3-4 high-priority keywords in opening")
    
    return "\n".join(suggestions)


# ════════════════════════════════════════════════════════════
#  LAB 4 EXAMPLE: WRITE PROFESSIONAL EMAIL
#  "Agent A researches, Agent B writes professional email"
# ════════════════════════════════════════════════════════════

class WriteEmailInput(BaseModel):
    """Pydantic schema for writing professional email."""
    recipient_name: str = Field(description="Name of the recipient")
    company_name: str = Field(description="Company name")
    job_title: str = Field(description="Job title applying for")
    key_qualifications: str = Field(description="Key qualifications to highlight")


@tool(args_schema=WriteEmailInput)
def write_professional_email(
    recipient_name: str,
    company_name: str,
    job_title: str,
    key_qualifications: str
) -> str:
    """
    Write a professional cover email for job application.
    
    LAB 4 REQUIREMENT: Agent collaboration example.
    This demonstrates "Agent A researches qualifications, Agent B writes email"
    
    Args:
        recipient_name: Hiring manager's name
        company_name: Company applying to
        job_title: Position title
        key_qualifications: Qualifications to highlight
    
    Returns:
        Professional email text
    """
    email_body = f"""Subject: Application for {job_title} Position

Dear {recipient_name},

I am writing to express my strong interest in the {job_title} position at {company_name}. With my background and skills, I am confident I would be a valuable addition to your team.

Key Qualifications:
{key_qualifications}

I am particularly excited about the opportunity to contribute to {company_name}'s mission and would welcome the chance to discuss how my experience aligns with your needs.

I have attached my resume for your review and would be happy to provide any additional information you may need.

Thank you for considering my application. I look forward to the opportunity to speak with you.

Best regards,
[Candidate Name]
[Contact Information]"""

    return email_body


# ════════════════════════════════════════════════════════════
#  LAB 5: HIGH-RISK TOOL WITH ACTUAL EMAIL SENDING
# ════════════════════════════════════════════════════════════

class SendResumeInput(BaseModel):
    """Pydantic schema for resume submission (HIGH-RISK)."""
    employer_email: str = Field(description="Employer's email address")
    email_subject: str = Field(description="Email subject line")
    email_body: str = Field(description="Email body/cover letter text")
    sender_email: str = Field(description="Your Gmail address")
    sender_password: str = Field(description="Your Gmail app password")


@tool(args_schema=SendResumeInput)
def send_resume_to_employer(
    employer_email: str,
    email_subject: str,
    email_body: str,
    sender_email: str,
    sender_password: str
) -> str:
    """
    HIGH-RISK TOOL: Actually send resume email via Gmail SMTP.
    
    LAB 5 REQUIREMENT: High-risk action requiring human approval.
    
    This tool ACTUALLY sends an email using Gmail's SMTP server.
    Once sent, it CANNOT be undone.
    
    SETUP REQUIRED:
    1. Enable 2-Factor Authentication on Gmail
    2. Generate App Password: https://myaccount.google.com/apppasswords
    3. Use App Password (not your regular Gmail password)
    
    Args:
        employer_email: Recipient's email
        email_subject: Subject line
        email_body: Email content
        sender_email: Your Gmail address
        sender_password: Your Gmail app password (16-character)
    
    Returns:
        Confirmation of actual email send
    """
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = employer_email
        msg['Subject'] = email_subject
        
        # Attach email body
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Connect to Gmail SMTP server
        print(f"\n[Email] Connecting to Gmail SMTP server...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure connection
        
        # Login
        print(f"[Email] Authenticating as {sender_email}...")
        server.login(sender_email, sender_password)
        
        # Send email
        print(f"[Email] Sending email to {employer_email}...")
        text = msg.as_string()
        server.sendmail(sender_email, employer_email, text)
        
        # Close connection
        server.quit()
        
        return (
            f"✅ EMAIL SUCCESSFULLY SENT!\n"
            f"\n"
            f"Delivery Confirmation:\n"
            f"  From: {sender_email}\n"
            f"  To: {employer_email}\n"
            f"  Subject: {email_subject}\n"
            f"  Server: {SMTP_SERVER}\n"
            f"  Status: DELIVERED ✓\n"
            f"\n"
            f"The email has been actually sent via Gmail SMTP.\n"
            f"The recipient will receive it in their inbox shortly.\n"
            f"\n"
            f"Email Preview:\n"
            f"{'='*60}\n"
            f"{email_body[:300]}...\n"
            f"{'='*60}"
        )
        
    except smtplib.SMTPAuthenticationError:
        return (
            f"❌ EMAIL SEND FAILED: Authentication Error\n"
            f"\n"
            f"Gmail rejected the login credentials.\n"
            f"\n"
            f"Setup Instructions:\n"
            f"1. Go to: https://myaccount.google.com/apppasswords\n"
            f"2. Enable 2-Factor Authentication if not already enabled\n"
            f"3. Generate an 'App Password' for 'Mail'\n"
            f"4. Use the 16-character app password (not your regular password)\n"
            f"\n"
            f"Error: Invalid email or app password"
        )
    
    except smtplib.SMTPException as e:
        return (
            f"❌ EMAIL SEND FAILED: SMTP Error\n"
            f"\n"
            f"Error details: {str(e)}\n"
            f"\n"
            f"Common issues:\n"
            f"1. Check internet connection\n"
            f"2. Verify Gmail SMTP is not blocked by firewall\n"
            f"3. Ensure app password is correct (16 characters, no spaces)"
        )
    
    except Exception as e:
        return (
            f"❌ EMAIL SEND FAILED: Unexpected Error\n"
            f"\n"
            f"Error: {str(e)}\n"
            f"\n"
            f"Please verify:\n"
            f"1. Sender email is a valid Gmail address\n"
            f"2. Recipient email is properly formatted\n"
            f"3. Network connection is active"
        )


# ════════════════════════════════════════════════════════════
#  LAB 4: ROLE-BASED TOOL RESTRICTION
# ════════════════════════════════════════════════════════════

RESEARCHER_TOOLS = [query_knowledge_base, extract_skills_from_resume]
ANALYST_TOOLS = [query_knowledge_base, calculate_skill_match_score, calculate_ats_score]
WRITER_TOOLS = [query_knowledge_base, generate_improvement_suggestions, write_professional_email]
EXECUTOR_TOOLS = [send_resume_to_employer]

ALL_TOOLS = [
    query_knowledge_base,
    extract_skills_from_resume,
    calculate_skill_match_score,
    calculate_ats_score,
    generate_improvement_suggestions,
    write_professional_email,
    send_resume_to_employer
]


if __name__ == "__main__":
    print("=" * 60)
    print("  Mid-Exam Part A - Tools Module")
    print("  WITH ACTUAL EMAIL SENDING")
    print("=" * 60)
    print(f"\nTotal Tools: {len(ALL_TOOLS)}")
    for i, tool in enumerate(ALL_TOOLS, 1):
        risk = " ⚠️ HIGH-RISK (ACTUALLY SENDS EMAIL)" if tool.name == "send_resume_to_employer" else ""
        print(f"  {i}. {tool.name}{risk}")
    print("\n" + "=" * 60)
    print("IMPORTANT: send_resume_to_employer now ACTUALLY sends emails!")
    print("Setup required:")
    print("  1. Gmail account with 2FA enabled")
    print("  2. App password generated from Google Account settings")
    print("  3. Use app password (16-char) not regular password")
    print("=" * 60)