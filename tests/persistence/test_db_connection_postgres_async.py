import asyncio
import logging
import sys
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Set up logging to see detailed connection information
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    stream=sys.stdout)

# Your actual database connection parameters
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# Additional connection parameters
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

async def test_async_connection():
    print(f"Testing async connection to PostgreSQL: {DB_URI}")
    
    try:
        # Create an async connection pool
        print("Creating async connection pool...")
        async with AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=5,
            kwargs=connection_kwargs,
        ) as pool:
            # Create the checkpointer
            print("Creating async checkpointer...")
            checkpointer = AsyncPostgresSaver(pool)
            
            # Setup tables if needed (first time only)
            print("Setting up tables...")
            await checkpointer.setup()
            
            # Test registering a thread
            thread_id = "test-async-thread"
            print(f"Creating test config with thread_id: {thread_id}")
            config = {"configurable": {"thread_id": thread_id}}
            
            # Test with a simple data write/read
            test_data = {
                "test_key": "test_value",
                "timestamp": "2024-04-21T12:00:00Z"
            }
            
            print(f"Writing test data: {test_data}")
            updated_config = await checkpointer.aput(config, test_data)
            print(f"Data written with checkpoint_id: {updated_config['configurable'].get('checkpoint_id')}")
            
            # Retrieve the data
            print("Reading data back...")
            checkpoint = await checkpointer.aget(updated_config)
            print(f"Retrieved data: {checkpoint}")
            
            # List checkpoints for the thread
            print("Listing checkpoints...")
            checkpoint_tuples = [c async for c in checkpointer.alist(config, limit=5)]
            print(f"Found {len(checkpoint_tuples)} checkpoints")
            
            print("✅ Async connection test completed successfully!")
            
    except Exception as e:
        print(f"❌ Error during async connection test: {e}")
        import traceback
        traceback.print_exc()

# Run the async test
if __name__ == "__main__":
    asyncio.run(test_async_connection())