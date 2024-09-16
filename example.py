from tickr_agent.main import TickrAgent
from loguru import logger

# Example Usage
if __name__ == "__main__":
    try:
        # Define stock tickers
        stocks = ["NVDA", "CEG"]

        # Run the financial analysis and save to JSON
        # result = run_financial_analysis(stocks, output_file)
        agent = TickrAgent(
            stocks=stocks,
            max_loops=1,
            workers=10,
            retry_attempts=1,
            context_length=16000,
        )

        result = agent.run("Conduct an analysis of this summary")

        # Output the result
        print(result)
    except Exception as e:
        logger.critical(
            f"Critical error in financial agent execution: {e}"
        )
