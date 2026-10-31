import asyncio
import time
import json
import logging

logging.basicConfig(level=logging.INFO)

from mcp_server_main import search_bedesten_semantic

async def run_test():
    start_time = time.time()
    print("Starting test with court_types=['YARGITAYKARARI'], initial_keyword='tahliye taahhütnamesi'...")
    
    # FastMCP wraps tools in FunctionTool, underlying coroutine is .fn
    func = getattr(search_bedesten_semantic, "fn", search_bedesten_semantic)
    
    result = await func(
        initial_keyword="tahliye taahhütnamesi",
        query="Kiracının kiralananı belli bir tarihte tahliye edeceğine dair taahhütname vermesi ve süresinde icra takibi yapılması",
        court_types=["YARGITAYKARARI"],
        top_k=10
    )
    
    elapsed = time.time() - start_time
    print(f"\n--- TEST COMPLETED IN {elapsed:.2f} SECONDS ---")
    print(f"Status: {result.get('status')}")
    print(f"Total documents processed: {result.get('total_documents_processed')}")
    print(f"Embedding model: {result.get('embedding_model')}")
    print(f"Stats: {result.get('stats')}")
    results = result.get("results", [])
    print(f"Results count: {len(results)}\n")
    
    with open("test_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    asyncio.run(run_test())
