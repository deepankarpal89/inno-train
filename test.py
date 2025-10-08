import asyncio

async def fetch_data():
    print("Start fetching")
    await asyncio.sleep(2)  # simulate IO work
    print("Done fetching")
    return {"data": 123}

async def main():
    print("Before await")
    result = await fetch_data()  # execution pauses here until fetch_data finishes
    print("After await:", result)

asyncio.run(main())
