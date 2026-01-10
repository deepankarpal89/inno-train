from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from typing import AsyncIterator
from dotenv import load_dotenv
import os
import asyncio

from app.config import settings
from app.api import api_router
from app.database import init_db, close_db
from app.api.endpoints import TRAINING_EXECUTOR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting InnoTrain API server")
    logger.info("📊 Initializing database connection...")
    await init_db()
    logger.info("✅ Database connected successfully")
    yield
    # Shutdown
    logger.info("🛑 Shutting down InnoTrain API server")
    TRAINING_EXECUTOR.shutdown(wait=True)
    logger.info("Closing ThreadPoolExecutor")
    logger.info("📊 Closing database connections...")
    await close_db()
    logger.info("✅ Database connections closed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to InnoTrain - EC2 Docker Training Simulator",
            "version": settings.app_version,
            "documentation": "/docs",
            "endpoints": {
                "start_training": "POST /api/v1/training/start - Start a new training job",
                "job_status": "GET /api/v1/jobs/{job_uuid} - Get job status",
                "job_details": "GET /api/v1/jobs/{job_uuid}/details - Get detailed job info",
                "list_jobs": "GET /api/v1/jobs - List all training jobs",
                "cancel_job": "DELETE /api/v1/jobs/{job_uuid} - Cancel a running job",
                "iteration_epochs": "GET /api/v1/iterations/{job_uuid}/{iteration_number}/epochs - Get epoch metrics",
                "list_evaluations": "GET /api/v1/evaluations - List all evaluations",
            },
        }

    # Include API routes
    app.include_router(api_router)

    return app


# Create the FastAPI application
app = create_application()

if __name__ == "__main__":
    import uvicorn

    load_dotenv()
    print("Port: ", os.getenv("PORT"))
    port = int(os.getenv("PORT"))
    print("Port: ", port)
    uvicorn.run("app.main:app", host=settings.host, port=port, reload=True, workers=4)
