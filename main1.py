"""
=============================================================
  Lab 3 – Interactive Resume Assistant
  File: main.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

Interactive interface that:
1. Asks user to upload/paste resume
2. Asks for job description
3. Runs autonomous analysis using the agent
=============================================================
"""

import os
from pathlib import Path
from graph import agent_graph
from langchain_core.messages import HumanMessage


def read_resume_from_file(filepath: str) -> str:
    """Read resume from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def get_resume_input() -> str:
    """Get resume text from user - either upload or paste."""
    print("\n" + "="*60)
    print("  STEP 1: Provide Your Resume")
    print("="*60)
    print("\nHow would you like to provide your resume?")
    print("  1. Paste resume text directly")
    print("  2. Load from a .txt file")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        print("\nPaste your resume text below.")
        print("(Press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done)\n")
        print("-" * 60)
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        resume_text = "\n".join(lines)
        
        if len(resume_text.strip()) < 50:
            print("\n⚠️  Resume seems too short. Please provide a complete resume.")
            return get_resume_input()
        
        return resume_text
    
    elif choice == "2":
        filepath = input("\nEnter path to resume file (e.g., resume.txt): ").strip()
        
        if not os.path.exists(filepath):
            print(f"\n❌ File not found: {filepath}")
            return get_resume_input()
        
        resume_text = read_resume_from_file(filepath)
        
        if resume_text.startswith("Error"):
            print(f"\n❌ {resume_text}")
            return get_resume_input()
        
        print(f"\n✓ Loaded resume ({len(resume_text)} characters)")
        return resume_text
    
    else:
        print("\n❌ Invalid choice. Please enter 1 or 2.")
        return get_resume_input()


def get_job_description() -> str:
    """Get job description from user."""
    print("\n" + "="*60)
    print("  STEP 2: Provide Job Description")
    print("="*60)
    print("\nHow would you like to provide the job description?")
    print("  1. Paste job description text")
    print("  2. Load from a .txt file")
    print("  3. Skip (get general resume advice)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nPaste the job description below.")
        print("(Press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done)\n")
        print("-" * 60)
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        return "\n".join(lines)
    
    elif choice == "2":
        filepath = input("\nEnter path to job description file: ").strip()
        
        if not os.path.exists(filepath):
            print(f"\n❌ File not found: {filepath}")
            return get_job_description()
        
        jd_text = read_resume_from_file(filepath)
        
        if jd_text.startswith("Error"):
            print(f"\n❌ {jd_text}")
            return get_job_description()
        
        print(f"\n✓ Loaded job description ({len(jd_text)} characters)")
        return jd_text
    
    elif choice == "3":
        return ""
    
    else:
        print("\n❌ Invalid choice. Please enter 1, 2, or 3.")
        return get_job_description()


def select_analysis_type() -> str:
    """Let user choose what analysis to perform."""
    print("\n" + "="*60)
    print("  STEP 3: Choose Analysis Type")
    print("="*60)
    print("\nWhat would you like me to do?")
    print("  1. Full Analysis (extract skills, calculate match score, ATS score, suggestions)")
    print("  2. Extract Skills Only")
    print("  3. Calculate ATS Score")
    print("  4. Get Improvement Suggestions")
    print("  5. Ask a Custom Question")
    
    choice = input("\nEnter choice (1-5): ").strip()
    return choice


def build_agent_prompt(resume: str, job_desc: str, analysis_type: str) -> str:
    """Build the prompt for the agent based on user choices."""
    
    base_context = f"""
I have the following resume:

{resume[:2000]}  {'...[truncated]' if len(resume) > 2000 else ''}
"""
    
    if job_desc:
        base_context += f"""

And I'm applying for a job with this description:

{job_desc[:1500]}  {'...[truncated]' if len(job_desc) > 1500 else ''}
"""
    
    # Build specific analysis request
    if analysis_type == "1":
        if job_desc:
            prompt = base_context + """

Please perform a FULL analysis:
1. Extract all skills from my resume
2. Calculate the skill match score between my resume and the job
3. Calculate the ATS compatibility score
4. Generate specific improvement suggestions

Be thorough and provide actionable feedback.
"""
        else:
            prompt = base_context + """

Please analyze my resume:
1. Extract all skills from my resume
2. Calculate the ATS compatibility score (general assessment)
3. Provide improvement suggestions for making my resume stronger

Since no job description was provided, give general advice.
"""
    
    elif analysis_type == "2":
        prompt = base_context + """

Please extract all skills from my resume and categorize them into:
- Programming Languages
- AI/ML Technologies
- Web Development
- Cloud & DevOps
- Databases
- Soft Skills
"""
    
    elif analysis_type == "3":
        if job_desc:
            prompt = base_context + """

Calculate the ATS (Applicant Tracking System) compatibility score for my resume
against this specific job description. Identify keyword gaps and formatting issues.
"""
        else:
            prompt = base_context + """

Calculate a general ATS (Applicant Tracking System) compatibility score for my resume.
Check for formatting issues and common ATS problems.
"""
    
    elif analysis_type == "4":
        if job_desc:
            prompt = base_context + """

Based on the job requirements, generate specific improvement suggestions for my resume.
Focus on:
- Missing skills I should highlight
- Keyword optimization
- STAR method bullet points
- ATS formatting improvements
"""
        else:
            prompt = base_context + """

Analyze my resume and generate improvement suggestions to make it stronger.
Focus on general best practices, formatting, and content quality.
"""
    
    elif analysis_type == "5":
        custom_q = input("\nEnter your question about the resume: ").strip()
        prompt = base_context + f"\n\n{custom_q}"
    
    else:
        prompt = base_context + "\n\nAnalyze my resume and provide feedback."
    
    return prompt


def run_interactive_session():
    """Run the full interactive resume analysis session."""
    print("""
    =================================================================
      Agentic AI Resume & Job Application Assistant
      Lab 3 - Interactive Mode
    =================================================================
    
    This assistant will:
    ✓ Analyze your resume using autonomous AI agents
    ✓ Calculate skill match and ATS compatibility scores
    ✓ Provide actionable improvement suggestions
    
    =================================================================
    """)
    
    # Step 1: Get resume
    resume_text = get_resume_input()
    
    print("\n✓ Resume received!")
    print(f"   Length: {len(resume_text)} characters")
    print(f"   Lines: {len(resume_text.splitlines())}")
    
    # Step 2: Get job description (optional)
    job_desc = get_job_description()
    
    if job_desc:
        print("\n✓ Job description received!")
        print(f"   Length: {len(job_desc)} characters")
    else:
        print("\n✓ Skipping job description (will provide general advice)")
    
    # Step 3: Choose analysis type
    analysis_type = select_analysis_type()
    
    # Build the agent prompt
    agent_prompt = build_agent_prompt(resume_text, job_desc, analysis_type)
    
    # Run the agent
    print("\n" + "="*60)
    print("  Running AI Agent Analysis")
    print("="*60)
    print("\nThe agent will autonomously decide which tools to use...")
    print("This may take 10-30 seconds.\n")
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=agent_prompt)]
    }
    
    # Run agent with streaming
    print("[Agent] Starting reasoning loop...\n")
    
    step_count = 0
    for event in agent_graph.stream(initial_state):
        for node_name, output in event.items():
            step_count += 1
            if "messages" in output:
                last_msg = output["messages"][-1]
                
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"[Step {step_count}] Agent decided to use tools:")
                    for tc in last_msg.tool_calls:
                        print(f"         → {tc['name']}")
                
                elif hasattr(last_msg, "content") and last_msg.content and node_name == "tools":
                    print(f"[Step {step_count}] Tools executed successfully")
    
    # Get final result
    final_state = agent_graph.invoke(initial_state)
    final_answer = final_state["messages"][-1].content
    
    # Display result
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print()
    print(final_answer)
    print()
    print("="*60)
    
    # Save option
    save = input("\nWould you like to save this analysis to a file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Enter filename (e.g., resume_analysis.txt): ").strip()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("  Resume Analysis Report\n")
            f.write("="*60 + "\n\n")
            f.write(final_answer)
        print(f"\n✓ Analysis saved to {filename}")


def run_knowledge_query_mode():
    """Run in knowledge query mode - no resume needed."""
    print("""
    =================================================================
      Knowledge Base Query Mode
    =================================================================
    
    Ask questions about:
    - ATS formatting rules
    - How the Resume Improver agent works
    - Industry skill keywords
    - System architecture
    
    =================================================================
    """)
    
    query = input("\nYour question: ").strip()
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print("\n[Agent] Thinking...\n")
    
    final_state = agent_graph.invoke(initial_state)
    final_answer = final_state["messages"][-1].content
    
    print("\n" + "="*60)
    print("  ANSWER")
    print("="*60)
    print()
    print(final_answer)
    print()
    print("="*60 + "\n")


def main():
    """Main entry point."""
    print("""
    =================================================================
      Agentic AI Resume & Job Application Assistant
      Lab 3 - LangGraph ReAct Agent
    =================================================================
    """)
    
    print("\nChoose mode:")
    print("  1. Full Resume Analysis (upload resume + job description)")
    print("  2. Knowledge Base Query (no resume needed)")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_interactive_session()
    elif choice == "2":
        run_knowledge_query_mode()
    elif choice == "3":
        print("\nGoodbye!")
        return
    else:
        print("\n❌ Invalid choice")
        return
    
    # Ask if user wants to continue
    again = input("\nWould you like to do another analysis? (y/n): ").strip().lower()
    if again == 'y':
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running (should be automatic)")
        print("  2. llama3.2 model is pulled")
        print("  3. ChromaDB from Lab 2 exists")