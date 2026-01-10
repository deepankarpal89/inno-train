"""
API endpoints and dependencies for the InnoTrain application.
"""

from fastapi import APIRouter
from .endpoints import router

# Create main router
api_router = APIRouter()

# Include all endpoints
api_router.include_router(router, prefix="/api", tags=["jobs"])

__all__ = ["api_router"]
