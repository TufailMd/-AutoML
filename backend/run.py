import uvicorn
import os
import sys

def main():
    # Append workspace root to path to resolve absolute imports
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)
        
    print("--------------------------------------------------")
    print("AutoML Cloud Platform Backend Service Initializing")
    print("Database Layer: SQLite active (automl.db)")
    print("S3 Simulator: active (.s3_mock/)")
    print("Address: http://127.0.0.1:8000")
    print("Swagger docs available at: http://127.0.0.1:8000/docs")
    print("--------------------------------------------------")
    
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
