# retrieval_test.md
## Lab 2 – Retrieval Test Results
**Project:** Agentic AI Resume & Job Application Assistant  
**Date:** 2026-02-15  
**Embedding Model:** all-MiniLM-L6-v2 (local, free)  
**Collection:** `resume_assistant_kb`  

---

### Test 1
**Query:** `What formatting rules should I follow to make my resume ATS compatible?`

**Result #1**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 3 of 7
- **Distance:** 0.2961
- **Preview:** _Use only standard bullet points and avoid custom symbols or special characters. Required Section Headings for ATS Compatibility: Work Experience or Professional Experience for the job history section. Education for the academic background section. Skills or Technical Skills or C..._

**Result #2**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 2 of 7
- **Distance:** 0.3412
- **Preview:** _e a single column layout because multi-column layouts confuse most ATS software. Avoid tables, text boxes, headers, footers, and sidebars completely. Use standard fonts such as Arial, Calibri, or Times New Roman in 10 to 12 point. Do not use graphics, logos, charts, or images any..._

**Result #3**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 1 of 7
- **Distance:** 0.4096
- **Preview:** _ATS Optimization Guidelines Domain Knowledge Base An Applicant Tracking System is software used by 98 percent of Fortune 500 companies to automatically screen resumes before a human recruiter sees them. Resumes that fail ATS parsing are automatically rejected and never seen. Crit..._


---

### Test 2
**Query:** `How does the Resume Improver Agent rewrite bullet points for better impact?`

**Result #1**
- **Title:** Resume Improver Agent
- **doc_type:** `agent_description`
- **department:** `core_agents`
- **priority_level:** `high`
- **Chunk:** 1 of 4
- **Distance:** 0.2997
- **Preview:** _Resume Improver Agent Role and Responsibilities The Resume Improver Agent takes the original resume content and the skill gap analysis to generate an enhanced version of the resume tailored to the target job description. Core Responsibilities: 1. Rewrite bullet points using the S..._

**Result #2**
- **Title:** Resume Improver Agent
- **doc_type:** `agent_description`
- **department:** `core_agents`
- **priority_level:** `high`
- **Chunk:** 2 of 4
- **Distance:** 0.4177
- **Preview:** _1. Rewrite bullet points using the STAR method: Situation, Task, Action, Result. This emphasizes measurable impact and outcomes rather than just listing duties. 2. Inject missing job-relevant keywords naturally into the experience and skills sections without keyword stuffing. 3...._

**Result #3**
- **Title:** Problem Statement and Project Motivation
- **doc_type:** `project_overview`
- **department:** `management`
- **priority_level:** `low`
- **Chunk:** 3 of 5
- **Distance:** 0.4222
- **Preview:** _s instead of impact-focused ones. Weak example: Responsible for managing a team. Strong example: Led a team of 8 engineers to deliver a 2 million dollar project 3 weeks ahead of schedule saving the company 120 thousand dollars. The Resume Improver Agent rewrites bullets using STA..._


---

### Test 3
**Query:** `What is the ATS score formula and what threshold triggers another improvement cycle?`
**Metadata Filter:** `department == "domain_knowledge"`

**Result #1**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 6 of 7
- **Distance:** 0.4711
- **Preview:** _teness times 10. The passing threshold is ats score greater than or equal to 70. When ats score is below 70 the feedback loop triggers another improvement cycle. When ats score reaches 70 or above the system exits the feedback loop completely. Common ATS Platforms: Taleo by Oracl..._

**Result #2**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 1 of 7
- **Distance:** 0.5947
- **Preview:** _ATS Optimization Guidelines Domain Knowledge Base An Applicant Tracking System is software used by 98 percent of Fortune 500 companies to automatically screen resumes before a human recruiter sees them. Resumes that fail ATS parsing are automatically rejected and never seen. Crit..._

**Result #3**
- **Title:** ATS Optimization Guidelines
- **doc_type:** `ats_guidelines`
- **department:** `domain_knowledge`
- **priority_level:** `high`
- **Chunk:** 5 of 7
- **Distance:** 0.6027
- **Preview:** _nce bullets. Target keyword density is 2 to 4 occurrences per important keyword. Do not use more than 5 occurrences as this looks like keyword stuffing. ATS Score Formula Used in This Project: ats score equals keyword matches divided by total job description keywords times 60 plu..._


---

## Summary Table

| Test | Filter | Top Result | Distance |
|------|--------|-----------|----------|
| 1 | None (semantic) | ATS Optimization Guidelines | 0.2961 |
| 2 | None (semantic) | Resume Improver Agent | 0.2997 |
| 3 | department == domain_knowledge | ATS Optimization Guidelines | 0.4711 |

**Key Finding:** Test 3 shows metadata filtering restricts results to only `domain_knowledge` documents, preventing agent description docs from appearing even if they are semantically close. This significantly improves retrieval precision.