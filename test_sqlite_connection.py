#!/usr/bin/env python3
"""
Test script to verify SQLAlchemy is working with SQLite.
"""

import asyncio
import os
import uuid
from dotenv import load_dotenv

# Set environment variables for SQLite before importing app modules
os.environ["DB_TYPE"] = "sqlite"
os.environ["DB_NAME"] = "test_sqlite.db"

# Now import the app modules
from app.database import init_db, close_db, async_session_maker
from models.training_job import TrainingJob, TrainingJobStatus
from sqlalchemy import select


async def test_sqlite_connection():
    """Test SQLite connection and basic CRUD operations."""
    print("\n" + "=" * 70)
    print("TESTING SQLALCHEMY WITH SQLITE".center(70))
    print("=" * 70)

    # Initialize database
    print("\n🔌 Initializing database...")
    await init_db()

    try:
        # Create a test job
        print("\n📝 Creating test job...")
        job_uuid = str(uuid.uuid4())

        async with async_session_maker() as session:
            # Create job
            job = TrainingJob(
                uuid=job_uuid,
                project_id="sqlite-test-project",
                training_run_id="sqlite-test-run",
                status=TrainingJobStatus.PENDING,
                machine_config={
                    "instance_type": "test",
                    "instance_id": "test-123",
                },
                training_request={"test": True},
            )

            session.add(job)
            await session.commit()
            print(f"✅ Job created with UUID: {job_uuid}")

        # Read the job back
        print("\n🔍 Reading job from database...")
        async with async_session_maker() as session:
            stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
            result = await session.execute(stmt)
            fetched_job = result.scalars().first()

            if fetched_job:
                print(f"✅ Job retrieved successfully:")
                print(f"   UUID: {fetched_job.uuid}")
                print(f"   Project: {fetched_job.project_id}")
                print(f"   Status: {fetched_job.status.value}")
            else:
                print("❌ Failed to retrieve job")
                return False

        # Update the job
        print("\n✏️ Updating job status...")
        async with async_session_maker() as session:
            stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
            result = await session.execute(stmt)
            job_to_update = result.scalars().first()

            if job_to_update:
                job_to_update.status = TrainingJobStatus.RUNNING
                await session.commit()
                print(f"✅ Job status updated to: {job_to_update.status.value}")
            else:
                print("❌ Failed to update job")
                return False

        # Delete the job
        print("\n🗑️ Deleting job...")
        async with async_session_maker() as session:
            stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
            result = await session.execute(stmt)
            job_to_delete = result.scalars().first()

            if job_to_delete:
                await session.delete(job_to_delete)
                await session.commit()
                print(f"✅ Job deleted")
            else:
                print("❌ Failed to delete job")
                return False

        # Verify deletion
        async with async_session_maker() as session:
            stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
            result = await session.execute(stmt)
            deleted_job = result.scalars().first()

            if deleted_job is None:
                print("✅ Verified job was deleted")
            else:
                print("❌ Job still exists after deletion")
                return False

        print("\n🎉 All SQLite tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during SQLite test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Close database
        print("\n🔌 Closing database connection...")
        await close_db()


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_sqlite_connection())

    # Exit with appropriate code
    exit_code = 0 if success else 1
    exit(exit_code)
