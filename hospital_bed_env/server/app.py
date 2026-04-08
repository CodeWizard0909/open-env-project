# Copyright (c) 2026 Hospital Bed Env Team
# FastAPI server application

from openenv.core.env_server import create_app
from openenv.core.env_server.types import ServerMode
from hospital_bed_env.models import HospitalAction, HospitalObservation
from fastapi.responses import JSONResponse

from .hospital_environment import HospitalEnvironment


app = create_app(
    HospitalEnvironment,
    HospitalAction,
    HospitalObservation,
    env_name="hospital_bed_env",
    max_concurrent_envs=10,
)


@app.get("/")
async def root():
    """Root route for Hugging Face Spaces health check."""
    return JSONResponse(content={
        "name": "hospital-bed-env",
        "status": "running",
        "description": "Hospital Bed & Resource Allocation OpenEnv",
        "endpoints": ["/health", "/reset", "/step", "/state"]
    })


def main():
    import uvicorn
    import os
    # Get the path to this file and construct the correct module path
    server_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(server_dir)
    os.chdir(project_root)
    uvicorn.run("hospital_bed_env.server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
