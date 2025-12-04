import asyncio
from models.database import init_db, close_db
from models.training_job import TrainingJob

async def get_job(job_uuid: str):
    # Initialize database connection
    await init_db()
    
    try:
        # Get the job
        job = await TrainingJob.get(uuid=job_uuid)
        print(f"Job found: {job}")
        print(f"Job UUID: {job.uuid}")
        print(f"Status: {job.status}")
        print(f"Created at: {job.created_at}")
        if hasattr(job, 'completed_at') and job.completed_at:
            print(f"Completed at: {job.completed_at}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close the database connection
        await close_db()

async def get_connection_info():
    """Get database connection information with detailed debugging"""
    from tortoise.connection import connections
    conn = connections.get("default")
    
    info = {
        "db_type": "sqlite",
        "file_path": getattr(conn, 'filename', 'in-memory'),
        "connection_type": "direct"
    }
    
    try:
        # Test a simple query
        test_query = await conn.execute_query("SELECT 'test' as result;")
        info["test_query"] = dict(test_query[1][0]) if test_query[1] else "No results"
        
        # Get individual PRAGMAs
        async def get_pragma(name):
            try:
                result = await conn.execute_query(f"PRAGMA {name};")
                return dict(result[1][0]) if result[1] else None
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Get PRAGMA values
        info["journal_mode"] = await get_pragma("journal_mode")
        info["synchronous"] = await get_pragma("synchronous")
        info["page_size"] = await get_pragma("page_size")
        info["cache_size"] = await get_pragma("cache_size")
        
        # Get SQLite version and other info
        version = await conn.execute_query("SELECT sqlite_version() as version;")
        info["sqlite_version"] = dict(version[1][0]) if version[1] else "unknown"
        
    except Exception as e:
        info["error"] = f"Error: {str(e)}"
        import traceback
        info["traceback"] = traceback.format_exc()
    
    return info

if __name__ == "__main__":
    import asyncio
    from models.database import init_db, close_db
    
    async def main():
        await init_db()
        try:
            job_uuid = "5cf64a13-1e52-49b3-8b52-aca8fb77a2ac"
            await get_job(job_uuid)
            
            # Get and display connection info
            conn_info = await get_connection_info()
            print("\nDatabase Connection Info:")
            for key, value in conn_info.items():
                print(f"{key}: {value}")
                
        finally:
            await close_db()
    
    asyncio.run(main())