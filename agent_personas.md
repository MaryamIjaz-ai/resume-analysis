# Agent Personas - Lab 4 Requirement

**Lab Manual Requirement:** "A description of each agent's role, goal, and restricted toolset"

---

## 1. Researcher Agent

**Role:** Resume Analysis & Data Extraction

**Persona:** "A Resume Analyzer focused on extracting skills and understanding candidate profiles"

**Goal:**
- Extract all technical and soft skills from resume
- Categorize skills by domain
- Provide structured skill data for analysis

**Restricted Toolset:**
1. `query_knowledge_base` - Access domain knowledge
2. `extract_skills_from_resume` - Extract structured skills

**Cannot Access:** Scoring, matching, improvement, or submission tools

**Output Example:**
```
Extracted 15 skills from resume:
Programming Languages: Python, JavaScript, Java
AI/ML: TensorFlow, PyTorch
Cloud: AWS, Docker
```

---

## 2. Analyst Agent

**Role:** Job Matching & Quality Validation

**Persona:** "A Job Matching Specialist who calculates skill matches and ATS compatibility scores"

**Goal:**
- Calculate skill match percentage
- Compute ATS compatibility score (0-100)
- Identify skill gaps

**Restricted Toolset:**
1. `query_knowledge_base` - Access ATS guidelines
2. `calculate_skill_match_score` - Compare skills
3. `calculate_ats_score` - Evaluate ATS compatibility

**Cannot Access:** Skill extraction, improvement, or submission tools

**Output Example:**
```
Match Score: 70% (Good)
ATS Score: 75/100 (PASS)
Missing Skills: Kubernetes, PostgreSQL
```

---

## 3. Writer Agent

**Role:** Improvement Recommendation Generation

**Persona:** "A Resume Improvement Advisor who generates actionable suggestions using STAR method"

**Goal:**
- Generate specific improvement actions
- Provide STAR method templates
- Create ATS optimization strategies

**Restricted Toolset:**
1. `query_knowledge_base` - Access best practices
2. `generate_improvement_suggestions` - Create action plan

**Cannot Access:** Extraction, scoring, or submission tools

**Output Example:**
```
MISSING SKILLS:
  • Kubernetes: Add to Skills + mention in project
  
STAR METHOD:
  Template: [Action] + [Result] + [Metrics]
```

---

## 4. Executor Agent

**Role:** Application Submission (HIGH-RISK)

**Persona:** "An Application Submitter who handles final resume submission after human approval"

**Goal:**
- Submit approved resume to employer
- Confirm successful delivery
- Log submission

**Restricted Toolset:**
1. `send_resume_to_employer` - ⚠️ HIGH-RISK (requires HITL approval)

**Cannot Access:** Any analysis or modification tools

**Safety:** Requires explicit human approval before execution

---

## Collaboration Flow

```
Researcher → Analyst → Writer → [HUMAN REVIEW] → Executor
```

Each agent completes its task and hands over state to the next agent.