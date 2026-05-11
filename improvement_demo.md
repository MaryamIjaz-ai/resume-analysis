# Improvement Demonstration
AI407L Final Exam – Part A

## 1. Common Issue Identified

### Problem
The Resume Assistant was generating generic responses that were not fully aligned with the user’s resume and job description.

Example issues:
- Missing ATS analysis
- Missing missing-skill detection
- Generic email generation
- Weak job-specific recommendations

The system initially relied only on a simple prompt without retrieval grounding from ChromaDB.

---

# BEFORE IMPROVEMENT

## User Query

```text
Analyze this resume:
Python, SQL, Machine Learning

Job requires:
Python, AWS, Docker

System Output (Before)
Your resume looks good.
You have technical skills and experience.
Consider improving formatting.

Problems in Old Output
No ATS score
No missing skills
No job match analysis
Generic feedback
No retrieval grounding
No professional email generation

Improvement Applied
Fix Implemented

The system was improved using:

A. Better Prompt Engineering

The agent prompt was modified to force:

Resume analysis
ATS scoring
Missing skill detection
Professional email drafting
Structured outputs
B. Retrieval-Augmented Generation (RAG)

ChromaDB retrieval was integrated using:

query_knowledge_base
Resume improvement knowledge
ATS formatting rules
Industry skill keyword matching

## AFTER IMPROVEMENT
User Query
Analyze this resume:
Python, SQL, Machine Learning

Job requires:
Python, AWS, Docker
System Output (After)
Resume Analysis:
The candidate demonstrates strong skills in Python, SQL, and Machine Learning.

Skill Match Score: 67%

Missing Skills:
- AWS
- Docker

ATS Score: 74/100

Recommendations:
- Add cloud deployment projects
- Include Docker usage in projects
- Mention AWS services if previously used
- Improve keyword density

Professional Email Draft:

Dear Hiring Manager,

I am excited to apply for the role. My experience in Python, SQL, and Machine Learning aligns strongly with your technical requirements. I am eager to contribute my analytical and problem-solving skills to your organization.

Best Regards,
Candidate