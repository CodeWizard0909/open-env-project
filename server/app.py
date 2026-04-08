import uvicorn
import os
from hospital_bed_env.server.app import app

def main():
    """Main entry point for the OpenEnv validator."""
    uvicorn.run("hospital_bed_env.server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
