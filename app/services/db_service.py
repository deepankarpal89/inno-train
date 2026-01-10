"""
Database Service - Handles database operations using SQLAlchemy
"""

import logging
from typing import TypeVar, Type, List, Optional, Any, Dict
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session_maker

T = TypeVar("T")


class DatabaseService:
    """Service for database operations using SQLAlchemy."""

    def __init__(self):
        """Initialize the database service."""
        self.logger = logging.getLogger("DatabaseService")

    async def get(self, model_class: Type[T], **kwargs) -> Optional[T]:
        """Get a single record by filters.

        Args:
            model_class: The SQLAlchemy model class
            **kwargs: Filter criteria as keyword arguments

        Returns:
            The model instance if found, None otherwise
        """
        async with async_session_maker() as session:
            stmt = select(model_class).filter_by(**kwargs)
            result = await session.execute(stmt)
            return result.scalars().first()

    async def get_by_id(self, model_class: Type[T], id_value: Any) -> Optional[T]:
        """Get a record by primary key.

        Args:
            model_class: The SQLAlchemy model class
            id_value: The primary key value

        Returns:
            The model instance if found, None otherwise
        """
        async with async_session_maker() as session:
            return await session.get(model_class, id_value)

    async def get_all(self, model_class: Type[T], **kwargs) -> List[T]:
        """Get all records matching the filters.

        Args:
            model_class: The SQLAlchemy model class
            **kwargs: Filter criteria as keyword arguments

        Returns:
            List of model instances
        """
        async with async_session_maker() as session:
            stmt = select(model_class).filter_by(**kwargs)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def create(self, model_class: Type[T], **kwargs) -> T:
        """Create a new record.

        Args:
            model_class: The SQLAlchemy model class
            **kwargs: Model attributes as keyword arguments

        Returns:
            The created model instance
        """
        async with async_session_maker() as session:
            obj = model_class(**kwargs)
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj

    async def update(self, model_obj: T, **kwargs) -> T:
        """Update an existing record.

        Args:
            model_obj: The model instance to update
            **kwargs: Attributes to update as keyword arguments

        Returns:
            The updated model instance
        """
        async with async_session_maker() as session:
            # Add the object to the session
            session.add(model_obj)

            # Update attributes
            for key, value in kwargs.items():
                setattr(model_obj, key, value)

            # Commit changes
            await session.commit()
            await session.refresh(model_obj)
            return model_obj

    async def delete(self, model_obj: T) -> bool:
        """Delete a record.

        Args:
            model_obj: The model instance to delete

        Returns:
            True if successful
        """
        async with async_session_maker() as session:
            await session.delete(model_obj)
            await session.commit()
            return True

    async def execute_raw(self, query, **params):
        """Execute a raw SQL query.

        Args:
            query: The SQL query to execute
            **params: Query parameters

        Returns:
            Query result
        """
        async with async_session_maker() as session:
            result = await session.execute(query, params)
            return result


# Create a global instance
db_service = DatabaseService()
