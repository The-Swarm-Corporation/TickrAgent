# Tickr Agent by TGSC


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)


`tickr-agent` is an enterprise-ready, scalable Python library for building swarms of financial agents that conduct comprehensive stock analysis and produce insights. Powered by `yfinance`, `loguru` for logging, and `pydantic` for data validation, `tickr-agent` is designed for businesses that need robust financial data processing, multithreading, and seamless integration with AI-powered models (via OpenAI) to derive actionable insights.

With `tickr-agent`, you can automate the retrieval of stock market data, perform technical analysis (e.g., RSI, moving averages), and integrate insights into AI-driven workflows. This solution is built to handle real-time data processing, analysis, and reporting at an enterprise scale.

## Key Features

- **Multithreaded Stock Data Fetching**: Retrieve and analyze financial data for multiple stocks concurrently for faster performance.
- **Advanced Logging**: Built with `loguru`, offering superior logging capabilities for debugging, auditing, and monitoring.
- **Enterprise-Ready**: Production-grade, modular design, and scalable for high-performance environments.
- **AI-Integrated**: Leverage OpenAI models to generate comprehensive financial analysis and predictions based on real-time stock data.
- **Pydantic Validation**: Ensures reliable and validated stock data, minimizing errors in data processing.

## Installation

To install `tickr-agent`, simply run:

```bash
$ pip3 install -U tickr-agent
```

## Getting Started

### Basic Example: Running Financial Analysis on a Single Stock

```python
from tickr_agent.main import TickrAgent
from loguru import logger

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

result = agent.run("Conduct an analysis on this stock and show me if it's a buy or not and why")

# Output the result
print(result)

```

### How It Works

1. **Stock Data Fetching**: The agent fetches real-time stock data from multiple financial APIs using the `yfinance` library. Technical indicators such as **RSI**, **50-day moving average**, and **200-day moving average** are calculated.
  
2. **Multithreading**: Multiple stock tickers are processed concurrently using Python’s `ThreadPoolExecutor`, ensuring high performance and efficiency.
  
3. **Data Validation**: Each piece of stock data is validated using `pydantic`, ensuring the reliability of the data processed.

4. **AI-Powered Analysis**: After gathering and validating stock data, the agent passes the summary of stock performance to an OpenAI-powered model, which can provide deeper insights, forecasts, or personalized reports based on the stock performance.

## Enterprise Use Case: Swarms of Financial Agents

For large enterprises, `tickr-agent` supports creating swarms of financial agents, each focusing on different sectors, regions, or investment strategies. These swarms can analyze hundreds or even thousands of stocks concurrently, generate reports, and trigger AI-driven insights for decision-making.

### Example: Running a Swarm of Agents for Multiple Stocks

```python
from tickr_agent.main import TickrAgent
from loguru import logger

# Example Usage
if __name__ == "__main__":
    try:
        # Define multiple stock tickers
        stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        # Initialize the agent for multi-stock analysis
        agent = TickrAgent(
            stocks=stocks,
            max_loops=2,          # Increased loops for more in-depth analysis
            workers=20,           # Number of threads for concurrent execution
            retry_attempts=2,      # Retry logic for reliability
            context_length=32000,  # Maximum context length for AI models
        )

        # Run the financial analysis
        result = agent.run("Provide a detailed financial health report for the selected stocks.")

        # Output the result
        print(result)

    except Exception as e:
        logger.critical(f"Critical error in financial agent execution: {e}")
```

### Advanced Customization

`tickr-agent` can be customized based on your enterprise needs, including:

- **Custom AI Models**: Integrate custom models or fine-tuned versions of GPT for industry-specific insights.
- **Data Pipelines**: Use `tickr-agent` as part of a larger financial data pipeline, feeding real-time stock analysis into dashboards, reporting systems, or decision-making tools.
- **Scalable Architecture**: Deploy swarms of agents in cloud environments, utilizing Kubernetes or Docker for containerized scaling.

## Logging and Monitoring

`tickr-agent` uses `loguru` for logging, providing robust, enterprise-grade logging capabilities:

- **File Rotation**: Log files are rotated automatically after reaching 1 MB in size.
- **Detailed Error Tracking**: Comprehensive error logging, including stack traces and timestamps, ensures that failures are easily traceable.
- **Custom Log Levels**: Adjust the verbosity of logs as needed (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

Logs are saved to `financial_agent_log.log` by default. Customize the logging configuration to integrate with your enterprise logging systems.

## Contributing

Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request to the GitHub repository. We follow standard Python development practices and require tests for all new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Join our discord for real-time support or email me at kye@swarms.world


# Todo

- [ ] Implement a better prompt with instructions and multi-shot examples
- [ ] Implement more stock data and or richer financial calculations
- [ ] Implement a multi-agent implementation to predict stock price will go up or down


# License
MIT