import time
from typing import List, Optional

from aiocache import Cache, cached
from aiocache.plugins import HitMissRatioPlugin
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, validator
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from tickr_agent.main import TickrAgent
import uvicorn

app = FastAPI()

# Loguru setup
logger.add("tickr_agent.log", rotation="500 MB", retention="10 days")

# Configure Cache (Redis is recommended in production, using in-memory for simplicity here)
cache = Cache.MEMORY


class AnalysisRequest(BaseModel):
    """
    Pydantic model for input validation of analysis requests.
    """

    stocks: List[str] = Field(
        ...,
        example=["AAPL", "GOOGL", "MSFT"],
        description="List of stock tickers to analyze.",
    )
    query: str = Field(
        ...,
        example="Conduct an analysis on this stock and show me if it's a buy or not and why",
        description="The query for stock analysis.",
    )
    max_loops: Optional[int] = Field(
        1, description="Max loops the agent should run."
    )
    workers: Optional[int] = Field(
        10, description="Number of workers for concurrent execution."
    )
    retry_attempts: Optional[int] = Field(
        1, description="Number of retry attempts in case of failure."
    )
    context_length: Optional[int] = Field(
        16000, description="Context length for agent processing."
    )
    return_json_on: Optional[bool] = Field(
        False,
        description="Flag to control whether results should be returned as JSON.",
    )

    @validator("stocks")
    def validate_stocks(cls, v):
        if not v:
            raise ValueError("At least one stock ticker is required.")
        return v


class AnalysisResponse(BaseModel):
    """
    Pydantic model for output validation of analysis responses.
    """

    success: bool
    message: str
    result: Optional[dict] = None


# Caching Decorator (key: query + stocks combination)
@cached(ttl=300, cache=Cache.MEMORY, plugins=[HitMissRatioPlugin()])
async def run_cached_analysis(agent: TickrAgent, query: str) -> dict:
    """
    Run the analysis with caching. The result is cached for 300 seconds to improve performance.
    """
    return agent.run(query)


# Retry Decorator for Analysis (3 attempts, wait 1 second between each)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def analyze_with_retry(agent: TickrAgent, query: str) -> dict:
    """
    Attempt to run the analysis with a retry mechanism in case of failure.
    """
    return await run_cached_analysis(agent, query)


@app.post("/v1/agent/analyze", response_model=AnalysisResponse)
async def analyze_stocks(request: AnalysisRequest):
    """
    Endpoint that accepts a list of stock tickers and query for analysis, with retry and caching mechanisms.

    Args:
        request (AnalysisRequest): Contains the list of stocks and other parameters for the analysis.

    Returns:
        AnalysisResponse: The result of the analysis in a structured format.
    """
    logger.info(f"Received analysis request: {request}")

    start_time = time.time()  # For timing the execution

    try:
        # Initialize the TickrAgent with the provided stocks and parameters
        agent = TickrAgent(
            stocks=request.stocks,
            max_loops=request.max_loops,
            workers=request.workers,
            retry_attempts=request.retry_attempts,
            context_length=request.context_length,
            return_json_on=request.return_json_on,
        )

        # Execute the analysis with retry and caching
        result = await analyze_with_retry(agent, request.query)

        end_time = time.time()  # Timing finished

        # Log execution time
        logger.info(
            f"Analysis completed in {end_time - start_time:.2f} seconds"
        )

        # Return the result with success message
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully.",
            result=result,
        )

    except RetryError as e:
        logger.error(f"Analysis failed after retries: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed after multiple retries.",
        )

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        )


@app.get("/v1/agent/status", response_model=dict)
async def get_status():
    """
    Basic endpoint to check the status of the API, including cache status.
    """
    logger.info("Status check endpoint hit.")

    # Check the cache hit/miss ratio
    cache_stats = cache.plugins[0].hit_miss_ratio()

    return {
        "status": "TickrAgent API is running",
        "version": "1.0",
        "cache_stats": cache_stats,
    }


@app.on_event("shutdown")
async def on_shutdown():
    """
    Function to gracefully shut down the application and clean up resources.
    """
    logger.info("Shutting down the TickrAgent API...")


# Middleware to log requests/response duration
@app.middleware("http")
async def log_request_data(request, call_next):
    """
    Middleware to log request details and response time.
    """
    logger.info(f"Request: {request.method} {request.url}")
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} completed in {process_time:.2f} seconds"
    )

    return response

# The rest of your code remains the same...

if __name__ == "__main__":
    logger.info("Starting TickrAgent API with Uvicorn...")

    uvicorn.run(
        "main:app",  # This refers to the FastAPI app created earlier
        host="0.0.0.0",  # Bind to all available interfaces
        port=8080,  # Port to run the app
        reload=True  # Enable auto-reloading for development
    )
