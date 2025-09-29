from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from typing import AsyncIterator

from app.config import settings
from app.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting InnoTrain API server")
    yield
    # Shutdown
    logger.info("🛑 Shutting down InnoTrain API server")



def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to InnoTrain - EC2 Docker Training Simulator",
            "version": settings.app_version,  # Using version from settings
            "documentation": "/docs",
            "endpoints": {
                "train": "/api/v1/train - Start a training job",
                "status": "/api/v1/status/{job_id} - Check job status",
                "jobs": "/api/v1/jobs - List all jobs",
            },
        }

    # Include API routes
    app.include_router(api_router)

    return app

# Create the FastAPI application
app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)