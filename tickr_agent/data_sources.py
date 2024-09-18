import yfinance as yf
from loguru import logger
import pandas as pd
from typing import Dict, Any, List
import ffn


def fetch_stock_data_yfinance(
    ticker: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Fetches stock data from Yahoo Finance, including price, market cap, RSI, and more.
    Accepts a start and end date for historical data.

    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :param start_date: Start date for fetching historical data (e.g., '2020-01-01').
    :param end_date: End date for fetching historical data (e.g., '2020-12-31').
    :return: A dictionary with stock data and indicators.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Fetch stock history with start and end dates
        history = stock.history(start=start_date, end=end_date)
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
        average_gain = up.rolling(window=14, min_periods=14).mean()
        average_loss = down.rolling(window=14, min_periods=14).mean()
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = (
            rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else "N/A"
        )

        # Moving averages
        ma_50 = history["Close"].rolling(window=50).mean().iloc[-1]
        ma_200 = history["Close"].rolling(window=200).mean().iloc[-1]
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
            "volume": info.get("volume", "N/A"),
            "implied_volatility": info.get(
                "impliedVolatility", "N/A"
            ),
        }

        logger.info(f"Successfully retrieved data for {ticker}")
        return stock_data

    except Exception as e:
        logger.error(
            f"Error fetching stock data from Yahoo Finance for {ticker}: {e}"
        )
        return {"error": str(e)}


def get_stock_data(
    tickers: List[str], start_date: str, end_date: str
) -> str:
    """
    Fetches stock data from both ffn and Yahoo Finance for given tickers between start_date and end_date
    and returns a comprehensive analysis.

    :param tickers: List of stock ticker symbols (e.g., ['aapl', 'msft']).
    :param start_date: Start date for fetching data (e.g., '2010-01-01').
    :param end_date: End date for fetching data (e.g., '2020-01-01').
    :return: Comprehensive stock data and analysis as a string.
    """
    result = ""
    try:
        # Fetch stock data from ffn (Yahoo Finance default)
        logger.info(
            f"Fetching stock data for {tickers} from {start_date} to {end_date}"
        )
        data = ffn.get(
            ",".join(tickers), start=start_date, end=end_date
        )
        logger.info(f"Data fetched successfully for {tickers}")

        # Step 1: Stock data as string
        data_str = data.to_string()

        # # Step 2: Calculate monthly returns
        # monthly_returns = data.to_monthly_returns().dropna()
        # logger.info(f"Monthly returns calculated for {tickers}")
        # monthly_returns_str = monthly_returns.to_string()

        # Step 3: Performance statistics
        perf_stats = data.calc_stats()
        logger.info(
            f"Performance statistics calculated for {tickers}"
        )
        perf_stats_display = perf_stats.display()

        # Step 4: Fetch detailed stock data from Yahoo Finance (via yfinance)
        for ticker in tickers:
            stock_data = fetch_stock_data_yfinance(
                ticker, start_date, end_date
            )
            stock_data_str = "\n".join(
                [
                    f"{key}: {value}"
                    for key, value in stock_data.items()
                ]
            )
            result += f"\n\nYahoo Finance Data for {ticker}:\n{stock_data_str}"

        # Combine ffn and Yahoo Finance data
        result += f"\n\nStock Data from ffn:\n{data_str}\n\n"
        # result += f"Monthly Returns:\n{monthly_returns_str}\n\n"
        result += f"Performance Statistics:\n{perf_stats_display}\n\n"

        return result

    except Exception as e:
        logger.error(f"Error fetching stock data for {tickers}: {e}")
        return f"Error fetching data: {e}"


def fetch_data_for_stocks(tickers: List[str], *args, **kwargs) -> str:
    response = ""
    for ticker in tickers:
        response += get_stock_data(tickers=[ticker], *args, **kwargs)

    return response


stock_data = fetch_data_for_stocks(
    tickers=["aapl", "msft"],
    start_date="2020-01-01",
    end_date="2020-12-31",
)
print(stock_data)
