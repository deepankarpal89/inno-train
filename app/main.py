from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from typing import AsyncIterator
from dotenv import load_dotenv
import os

from app.config import settings
from app.api import api_router
from app.db import init_db, close_db

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

    # Cancel all active background tasks
    from app.api.endpoints import active_background_tasks, active_orchestrators

    if active_background_tasks:
        logger.info(
            f"📋 Cancelling {len(active_background_tasks)} active background tasks..."
        )
        for job_uuid, task in active_background_tasks.items():
            if not task.done():
                logger.info(f"⏹️ Cancelling task for job {job_uuid}")
                task.cancel()

        # Wait for tasks to complete cancellation
        if active_background_tasks:
            import asyncio

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *active_background_tasks.values(), return_exceptions=True
                    ),
                    timeout=30.0,
                )
                logger.info("✅ All background tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Some background tasks did not cancel within timeout")

    # Clean up orchestrators
    if active_orchestrators:
        logger.info(f"🧹 Cleaning up {len(active_orchestrators)} orchestrators...")
        for job_uuid, orchestrator in list(active_orchestrators.items()):
            try:
                # Attempt graceful cleanup
                if hasattr(orchestrator, "cleanup"):
                    await orchestrator.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup orchestrator {job_uuid}: {e}")
        active_orchestrators.clear()
        logger.info("✅ Orchestrators cleaned up")

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
    uvicorn.run("app.main:app", host=settings.host, port=port, reload=True)
