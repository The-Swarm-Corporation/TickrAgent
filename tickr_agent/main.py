import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, OpenAIChat
from swarms.models.base_llm import BaseLLM
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.utils.file_processing import create_file_in_folder

load_dotenv()


# Loguru logger configuration
logger.add(
    "financial_agent_log.log",
    rotation="1 MB",
    level="INFO",
    backtrace=True,
    diagnose=True,
)

model = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.1,
)


# Pydantic schema for stock data validation and serialization
class IndividualTickrAgentOutput(BaseModel):
    stock_data: Dict[str, Any]
    summary: str
    time_stamp: str = Field(
        default=time.strftime("%Y-%m-%d %H:%M:%S")
    )


class MultiStockTickrAgentOutput(BaseModel):
    logs: List[IndividualTickrAgentOutput]
    time_stamp: str = Field(
        default=time.strftime("%Y-%m-%d %H:%M:%S")
    )


class TickrAgent:
    def __init__(
        self,
        agent_name: str = "Financial-Analysis-Agent",
        system_prompt: str = FINANCIAL_AGENT_SYS_PROMPT,
        llm: BaseLLM = model,
        stocks: List[str] = [],
        max_loops: int = 1,
        workers: int = 10,
        retry_attempts: int = 1,
        context_length: int = 16000,
        output_file: str = None,
    ):
        self.stocks = stocks
        self.max_loops = max_loops
        self.workers = workers
        self.output_file = output_file
        self.system_prompt = system_prompt
        self.llm = llm
        self.retry_attempts = retry_attempts
        self.context_length = context_length
        self.agent_name = agent_name

        self.mult_stock_log = MultiStockTickrAgentOutput(
            logs=[], time_stamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # self.agent = self.initialize_agent()
        self.output_file = (
            "tickr_agent_run_"
            + time.strftime("%Y-%m-%d_%H-%M-%S")
            + ".json"
        )

    def fetch_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches stock data for a given ticker, including price, market cap, RSI, and more.

        Parameters:
        ticker (str): The stock ticker symbol.

        Returns:
        dict: A dictionary containing stock information for the given ticker.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Fetch stock history and calculate indicators
            history = stock.history(period="1y")
            if history.empty:
                logger.warning(
                    f"No historical data available for ticker: {ticker}"
                )
                return {"error": f"No historical data for {ticker}"}

            price = history["Close"][-1]

            # Calculate RSI (14-day)
            delta = history["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            average_gain = up.rolling(
                window=14, min_periods=14
            ).mean()
            average_loss = down.rolling(
                window=14, min_periods=14
            ).mean()
            rs = average_gain / average_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = (
                rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else "N/A"
            )

            # Moving averages
            ma_50 = (
                history["Close"].rolling(window=50).mean().iloc[-1]
            )
            ma_200 = (
                history["Close"].rolling(window=200).mean().iloc[-1]
            )
            short_term_trend = (
                "Uptrend" if ma_50 > ma_200 else "Downtrend"
            )

            # Prepare the data dictionary
            stock_data = {
                "ticker": ticker,
                "price": price,
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "rsi": current_rsi,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "short_term_trend": short_term_trend,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "company_name": info.get("longName", "N/A"),
                "summary": info.get("longBusinessSummary", "N/A"),
            }

            logger.info(f"Successfully retrieved data for {ticker}")

            # Conver the stock_data into a string
            stock_data_string = json.dumps(stock_data)

            return stock_data  # , stock_data_string

        except Exception as e:
            logger.error(
                f"Error retrieving data for ticker {ticker}: {e}"
            )
            return {"error": f"Failed to retrieve data for {ticker}"}

    def get_stock_info_multithreaded(self) -> List[Dict[str, Any]]:
        """
        Retrieves stock data for a list of tickers using multithreading.

        Returns:
        List[Dict[str, Any]]: A list of dictionaries containing stock data.
        """
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.fetch_stock_data, ticker): ticker
                for ticker in self.stocks
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(
                        f"Error processing {futures[future]}: {e}"
                    )
        return results

    # def validate_and_save_stock_data(
    #     self, stocks: List[str], output_file: str
    # ) -> None:
    #     """
    #     Fetches stock data using multithreading, validates using Pydantic, and saves it to a JSON file.

    #     Parameters:
    #     stocks (List[str]): List of stock ticker symbols.
    #     output_file (str): The output filename for saving the data.
    #     """
    #     logger.info(
    #         "Starting stock data retrieval with multithreading"
    #     )
    #     raw_data = self.get_stock_info_multithreaded(stocks)

    #     for stock in raw_data:
    #         # if "error" not in stock:
    #         #     try:
    #         #         stock_data = IndividualTickrAgentOutput(
    #         #             stock_data=stock
    #         #         )  # Validate with Pydantic
    #         #         valid_data.append(stock_data.dict())
    #         #     except ValidationError as e:
    #         #         logger.error(
    #         #             f"Validation error for {stock['ticker']}: {e}"
    #         #         )
    #         try:
    #             stock_data = IndividualTickrAgentOutput(
    #                 stock_data=stock
    #             )  # Validate with Pydantic
    #             self.mult_stock_log.logs.append(stock_data)
    #         except ValidationError as e:
    #             logger.error(
    #                 f"Validation error for {stock['ticker']}: {e}"
    #             )

    #     # # Save valid data to JSON
    #     # self.save_to_json(
    #     #     self.mult_stock_log.model_dump_json(indent=2), output_file
    #     # )

    def summarize_stock_positions(self, stock: str) -> str:
        """
        Summarizes stock positions using technical indicators and company information.

        Parameters:
        stocks_data (List[Dict[str, Any]]): A list of dictionaries containing stock data.

        Returns:
        str: A summary of the stock positions.

        Raises:
        ValueError: If the stock data list is empty or invalid.
        """
        try:
            summary = "Stock Positions Summary:\n"
            # for stock in stocks_data:
            #     if "error" not in stock:
            #         summary += (
            #             f"Ticker: {stock['ticker']}\n"
            #             f"Current Price: ${stock['price']:,.2f}\n"
            #             f"Market Cap: ${stock['market_cap']:,}\n"
            #             f"P/E Ratio: {stock['pe_ratio']}\n"
            #             f"EPS: {stock['eps']}\n"
            #             f"Dividend Yield: {stock['dividend_yield']}\n"
            #             f"RSI: {stock['rsi']}\n"
            #             f"50-Day MA: ${stock['ma_50']:,.2f}\n"
            #             f"200-Day MA: ${stock['ma_200']:,.2f}\n"
            #             f"Short-Term Trend: {stock['short_term_trend']}\n"
            #             f"Sector: {stock['sector']}\n"
            #             f"Industry: {stock['industry']}\n"
            #             f"Company Name: {stock['company_name']}\n"
            #             f"Summary: {stock['summary']}\n"
            #             f"{'-' * 40}\n"
            #         )
            # summary += self.fetch_stock_data(stock)
            stock_data = self.fetch_stock_data(stock)
            summary += json.dumps(stock_data)
            summary += "Note: Stocks with RSI > 70 may be overbought; those with RSI < 30 may be oversold.\n"
            logger.info(summary)
            return summary
        except Exception as e:
            logger.error(f"Error summarizing stock positions: {e}")
            raise

    # Initialize the financial agent
    def initialize_agent(self) -> Agent:
        """
        Initializes the financial analysis agent with the OpenAIChat model and configuration.

        Returns:
        Agent: The initialized financial analysis agent.
        """
        try:
            logger.info("Initializing financial analysis agent.")

            # Get the OpenAI API key from the environment variable
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                logger.error(
                    "OpenAI API key not found in environment variables."
                )
                raise ValueError("OpenAI API key not found.")

            agent = Agent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                llm=self.llm,
                max_loops=self.max_loops,
                autosave=True,
                dashboard=False,
                verbose=True,
                dynamic_temperature_enabled=True,
                saved_state_path="finance_agent.json",
                user_name="swarms_corp",
                retry_attempts=1,
                context_length=self.context_length,
                return_step_meta=False,
            )

            logger.info(
                "Financial analysis agent initialized successfully."
            )
            return agent
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            raise

    def run_one_stock(self, task: str, stock: str) -> str:
        """
        Runs a full financial analysis using the stock data and GPT-based agent.

        Parameters:
        output_file (str): The filename to save the stock data in JSON format.
        task (str): The task to pass to the agent for further analysis.

        Returns:
        str: The agent's summary and any relevant financial advice.

        Raises:
        ValueError: If there is an issue retrieving stock data or running the agent.
        """
        try:
            # logger.info("Running financial analysis.")

            # Summarize stock positions
            logger.info(f"Fetching stock infor for {stock}")
            stock_data = self.fetch_stock_data(stock)
            stock_summary = self.summarize_stock_positions(stock)

            # Initialize the agent
            agent = self.initialize_agent()

            # Send the stock summary to the agent as context for the task
            full_task = f"{task}\n\nHere is the summary of the stock positions:\n{stock_summary}"

            # Example prompt: Asking the agent about stock performance or other financial advice
            logger.info(f"Running {self.agent_name} on {stock} data")
            summary = agent.run(full_task)

            logger.info("Financial analysis completed successfully.")

            # Log
            log = IndividualTickrAgentOutput(
                stock_data=stock_data,
                summary=summary,
            )

            # Append to multi logs
            self.mult_stock_log.logs.append(log)

            # return f"Stock data saved to {output_file}.\n\nAnalysis and Advice:\n{summary}"
        except Exception as e:
            logger.error(f"Error running financial analysis: {e}")
            raise

    def run_many_stocks_concurrently(self, task: str):
        # logger.info(f"Fetching data for stocks: {self.stocks}")
        logger.info(
            f"Running financial analysis on {len(self.stocks)}"
        )
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.run_one_stock, task, stock
                ): stock
                for stock in self.stocks
            }
            for future in as_completed(futures):
                stock = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    logger.error(
                        f"Error running financial analysis for stock {stock}: {exc}"
                    )
                else:
                    logger.info(
                        f"Stock {stock} analysis completed successfully."
                    )
                    logger.info(data)

    def run(self, task: str, *args, **kwargs):
        self.run_many_stocks_concurrently(task, *args, **kwargs)

        output_data = self.mult_stock_log.model_dump_json(indent=4)

        if self.output_file is not None:
            create_file_in_folder(
                folder_path=os.getenv("WORKSPACE_DIR"),
                file_name=self.output_file,
                content=output_data,
            )

        return output_data
