from tickr_agent.main import TickrAgent

# Example Usage
# Define stock tickers
stocks = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "TSLA",
    "AMZN",
    # "FB",
    # "BRK-A",
    # "BRK-B",
    # "JPM",
    # "JNJ",
    # "V",
    # "PG",
    # "UNH",
    # "DIS",
    # "MA",
    # "HD",
    # "BAC",
    # "INTC",
    # "VZ",
    # "CMCSA",
    # "PFE",
    # "PEP",
    # "CSCO",
    # "KO",
    # "NFLX",
    # "T",
    # "MRK",
    # "WMT",
    # "ABT",
    # "NVDA",
    # "ADBE",
    # "XOM",
    # "NKE",
    # "PYPL",
    # "CRM",
    # "ABBV",
    # "ACN",
    # "CVX",
    # "TMO",
    # "LLY",
    # "DHR",
    # "MDT",
    # "QCOM",
    # "NEE",
    # "UNP",
    # "TXN",
    # "HON",
    # "LIN",
    # "LOW",
    # "BMY",
    # "SBUX",
    # "AMT",
    # "ORCL",
    # "UPS",
    # "LMT",
    # "COST",
    # "RTX",
    # "GS",
    # "BDX",
    # "MO",
    # "AXP",
    # "MS",
    # "IBM",
    # "MMM",
    # "CAT",
    # "CVS",
    # "TGT",
    # "DE",
    # "ISRG",
    # "FIS",
    # "SPGI",
    # "TJX",
    # "GILD",
    # "CME",
    # "ANTM",
    # "SO",
    # "CL",
    # "CI",
    # "BDX",
    # "ZTS",
    # "CSX",
    # "PLD",
    # "SCHW",
    # "ICE",
    # "ADI",
]

# Run the financial analysis and save to JSON
# result = run_financial_analysis(stocks, output_file)
agent = TickrAgent(
    stocks=stocks,
    max_loops=1,
    workers=10,
    retry_attempts=1,
    context_length=16000,
    return_json_on=False,
)

result = agent.run(
    "Conduct an analysis on this stock and show me if it's a buy or not and why"
)

# Output the result
print(result)
