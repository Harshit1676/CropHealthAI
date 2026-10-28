"""
Render deployment entry point for CropHealth AI backend.

This file is used by Render to start the FastAPI server.
It is a thin shim that runs uvicorn programmatically so that
Render's assigned PORT environment variable is respected.
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
