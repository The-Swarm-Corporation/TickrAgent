from tickr_agent.main import TickrAgent

# Example Usage
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

result = agent.run(
    "Conduct an analysis on this stock and show me if it's a buy or not and why"
)

# Output the result
print(result)
