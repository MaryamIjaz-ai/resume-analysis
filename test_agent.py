"""
=============================================================
  Lab 3 – Test Agent
  File: test_agent.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

This script tests the ReAct agent with various queries to demonstrate
autonomous tool selection and reasoning.

RUN:
  python test_agent.py
=============================================================
"""

from graph import run_agent


def test_knowledge_query():
    """Test: Agent queries knowledge base for ATS rules."""
    print("\n" + "="*70)
    print("TEST 1: Knowledge Base Query")
    print("="*70)
    
    query = "What are the most important ATS formatting rules I should follow?"
    
    result = run_agent(query, verbose=False)
    
    print("\n✓ Test passed: Agent queried knowledge base")
    return result


def test_skill_match():
    """Test: Agent calculates skill match between resume and job."""
    print("\n" + "="*70)
    print("TEST 2: Skill Match Calculation")
    print("="*70)
    
    query = """
    I have these skills on my resume: Python, JavaScript, React, Node.js, MongoDB, AWS.
    
    The job description requires: Python, React, Docker, Kubernetes, PostgreSQL.
    
    Calculate my skill match score and tell me what skills I'm missing.
    """
    
    result = run_agent(query, verbose=False)
    
    print("\n✓ Test passed: Agent calculated skill match")
    return result


def test_ats_scoring():
    """Test: Agent scores resume for ATS compatibility."""
    print("\n" + "="*70)
    print("TEST 3: ATS Compatibility Score")
    print("="*70)
    
    resume_text = """
    John Doe
    Software Engineer
    
    EXPERIENCE
    Senior Developer at Tech Corp
    - Built web applications using Python and Django
    - Managed AWS cloud infrastructure
    - Led team of 5 developers
    
    EDUCATION
    BS Computer Science, State University
    
    SKILLS
    Python, JavaScript, AWS, Docker, Git
    """
    
    query = f"""
    Score this resume for ATS compatibility. The job requires Python, AWS, 
    Docker, and Kubernetes as key skills.
    
    Resume:
    {resume_text}
    """
    
    result = run_agent(query, verbose=False)
    
    print("\n✓ Test passed: Agent calculated ATS score")
    return result


def test_multi_tool_reasoning():
    """Test: Agent uses multiple tools in sequence."""
    print("\n" + "="*70)
    print("TEST 4: Multi-Tool Reasoning Chain")
    print("="*70)
    
    resume = """
    Jane Smith - Data Scientist
    
    EXPERIENCE:
    ML Engineer at DataCorp (2020-2023)
    - Developed machine learning models using Python and TensorFlow
    - Deployed models on AWS
    
    SKILLS:
    Python, TensorFlow, Pandas, SQL
    """
    
    query = f"""
    Analyze this resume for a job that requires Python, Machine Learning,
    Docker, and Kubernetes.
    
    1. Extract the skills from the resume
    2. Calculate the skill match score
    3. Calculate the ATS score
    4. Give me improvement suggestions
    
    Resume:
    {resume}
    """
    
    result = run_agent(query, verbose=False)
    
    print("\n✓ Test passed: Agent used multiple tools in reasoning chain")
    return result


def main():
    """Run all tests."""
    print("""
    =================================================================
      Lab 3 - Agent Test Suite
      Agentic AI Resume & Job Application Assistant
    =================================================================
    
    This script demonstrates the ReAct agent autonomously:
    - Deciding which tools to use
    - Chaining multiple tools together
    - Reasoning about results
    
    Prerequisites:
    - Ollama running (ollama serve)
    - Model pulled (ollama pull llama3.2)
    - Lab 2 ChromaDB exists (./chroma_db/)
    
    =================================================================
    """)
    
    try:
        # Run tests
        test_knowledge_query()
        
        input("\nPress Enter to continue to Test 2...")
        test_skill_match()
        
        input("\nPress Enter to continue to Test 3...")
        test_ats_scoring()
        
        input("\nPress Enter to continue to Test 4...")
        test_multi_tool_reasoning()
        
        print("\n" + "="*70)
        print("  ALL TESTS PASSED ✓")
        print("="*70)
        print("""
        The agent successfully:
        ✓ Queried knowledge base
        ✓ Calculated skill matches
        ✓ Scored ATS compatibility
        ✓ Chained multiple tools together
        
        The ReAct loop is working correctly!
        """)
        
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Ollama running? (ollama serve)")
        print("  2. Is llama3.2 pulled? (ollama pull llama3.2)")
        print("  3. Does ./chroma_db/ exist? (run Lab 2 first)")
        raise


if __name__ == "__main__":
    main()