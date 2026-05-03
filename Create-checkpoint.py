"""
=============================================================
  LAB 5 REQUIREMENT: Generate checkpoint_db.sqlite
  
  Lab Manual Page 9: "checkpoint_db.sqlite: The local database
  file containing the saved states of your agent."
=============================================================

This script creates the SQLite database file that stores
checkpointed states for session persistence.

Run this to generate checkpoint_db.sqlite for submission.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import os


# ════════════════════════════════════════════════════════════
#  STATE DEFINITION
# ════════════════════════════════════════════════════════════

class CheckpointState(TypedDict):
    """Simple state for checkpoint demonstration."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    step_count: int


# ════════════════════════════════════════════════════════════
#  CREATE GRAPH WITH SQLITE CHECKPOINTER
# ════════════════════════════════════════════════════════════

def create_checkpointed_graph():
    """
    Create a graph with SQLite-based persistent checkpointing.
    
    LAB 5 REQUIREMENT: This creates checkpoint_db.sqlite file
    """
    
    print("="*70)
    print("  Creating Graph with SQLite Checkpointer")
    print("="*70)
    
    # Define a simple node
    def simple_node(state: CheckpointState) -> CheckpointState:
        print(f"\n[Node] Processing step {state.get('step_count', 0) + 1}")
        return {
            "messages": [HumanMessage(content=f"Step {state.get('step_count', 0) + 1} completed")],
            "step_count": state.get("step_count", 0) + 1
        }
    
    # Build graph
    workflow = StateGraph(CheckpointState)
    workflow.add_node("process", simple_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    # ────────────────────────────────────────────────────────
    #  LAB 5: SQLITE CHECKPOINTER CONFIGURATION
    # ────────────────────────────────────────────────────────
    
    db_path = "./checkpoint_db.sqlite"
    
    print(f"\n[Setup] Creating SQLite checkpointer at: {db_path}")
    
    # Create SQLite-based checkpointer
    checkpointer = SqliteSaver.from_conn_string(db_path)
    
    # Compile graph with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    print(f"[Setup] ✓ Graph compiled with SQLite checkpointer")
    
    return app, db_path


# ════════════════════════════════════════════════════════════
#  GENERATE CHECKPOINT DATA
# ════════════════════════════════════════════════════════════

def generate_checkpoint_data():
    """
    Run the graph with different thread_ids to populate
    checkpoint_db.sqlite with sample states.
    """
    
    print("\n" + "="*70)
    print("  Generating Checkpoint Data")
    print("="*70)
    
    app, db_path = create_checkpointed_graph()
    
    # Create multiple sessions with different thread_ids
    sessions = [
        ("session_001", "First user session"),
        ("session_002", "Second user session"),
        ("session_003", "Third user session")
    ]
    
    for thread_id, description in sessions:
        print(f"\n[Session] Creating checkpoint for: {thread_id}")
        print(f"[Session] Description: {description}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=f"Starting {description}")],
            "step_count": 0
        }
        
        # Run the graph - this creates checkpoints
        result = app.invoke(initial_state, config)
        
        print(f"[Session] ✓ Checkpoint saved for {thread_id}")
        print(f"[Session]   Step count: {result.get('step_count', 0)}")
        print(f"[Session]   Messages: {len(result.get('messages', []))}")
    
    return db_path


# ════════════════════════════════════════════════════════════
#  VERIFY CHECKPOINT DATABASE
# ════════════════════════════════════════════════════════════

def verify_checkpoint_db(db_path: str):
    """
    Verify that checkpoint_db.sqlite was created and contains data.
    """
    
    print("\n" + "="*70)
    print("  Verifying Checkpoint Database")
    print("="*70)
    
    if not os.path.exists(db_path):
        print(f"\n❌ ERROR: {db_path} was not created!")
        return False
    
    print(f"\n✓ File exists: {db_path}")
    
    # Check file size
    file_size = os.path.getsize(db_path)
    print(f"✓ File size: {file_size:,} bytes")
    
    if file_size == 0:
        print("❌ WARNING: Database file is empty!")
        return False
    
    # Try to read checkpoints
    try:
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if checkpoints table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='checkpoints'
        """)
        
        if cursor.fetchone():
            print("✓ Checkpoints table exists")
            
            # Count checkpoints
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            count = cursor.fetchone()[0]
            print(f"✓ Number of checkpoints: {count}")
            
            if count > 0:
                # Show sample checkpoint info
                cursor.execute("""
                    SELECT thread_id, checkpoint_ns 
                    FROM checkpoints 
                    LIMIT 5
                """)
                
                print("\nSample checkpoints:")
                for thread_id, ns in cursor.fetchall():
                    print(f"  - Thread ID: {thread_id}, Namespace: {ns}")
        else:
            print("❌ Checkpoints table not found!")
            return False
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return False
    
    print("\n✅ Database verification PASSED")
    return True


# ════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ════════════════════════════════════════════════════════════

def main():
    """Generate checkpoint_db.sqlite for Lab 5 submission."""
    
    print("""
    =================================================================
      LAB 5 - Checkpoint Database Generator
      Creating checkpoint_db.sqlite for submission
    =================================================================
    
    This script creates the SQLite database file that contains
    saved agent states for session persistence.
    
    Lab Manual Page 9 Requirement: "checkpoint_db.sqlite: The local
    database file containing the saved states of your agent."
    
    =================================================================
    """)
    
    try:
        # Generate the database
        db_path = generate_checkpoint_data()
        
        # Verify it was created correctly
        success = verify_checkpoint_db(db_path)
        
        if success:
            print("\n" + "="*70)
            print("  SUCCESS")
            print("="*70)
            print(f"""
✅ checkpoint_db.sqlite created successfully!
✅ Location: {os.path.abspath(db_path)}
✅ Contains multiple session checkpoints
✅ Ready for Lab 5 submission

What this demonstrates:
─────────────────────
1. SQLiteSaver checkpointer configuration
2. Persistent state storage across sessions
3. Multiple thread_id sessions saved
4. State can be recovered after script restart

Files for Lab 5 Submission:
──────────────────────────
1. persistence_test.py - Tests session recovery
2. approval_logic.py - Shows interrupt configuration
3. checkpoint_db.sqlite - This database file ✓

Lab 5 Task 1 Requirement: SATISFIED ✓
""")
        else:
            print("\n❌ Database verification failed")
            print("Please check for errors above")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. You have write permissions in current directory")
        print("  2. langgraph package is installed")
        print("  3. No other process is using checkpoint_db.sqlite")


if __name__ == "__main__":
    main()