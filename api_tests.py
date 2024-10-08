import requests

# Swappable base URL for your FastAPI app
BASE_URL = "https://tickragent-285321057562.us-central1.run.app"


def analyze_stocks():
    """
    Function to send a POST request to the `/v1/agent/analyze` endpoint.
    """
    url = f"{BASE_URL}/v1/agent/analyze"
    
    # Example request payload
    data = {
        "stocks": ["AAPL", "GOOGL", "MSFT"],
        "query": "Conduct an analysis on this stock and show me if it's a buy or not and why",
        "max_loops": 1,
        "workers": 10,
        "retry_attempts": 1,
        "context_length": 16000,
        "return_json_on": True
    }

    try:
        # Send the POST request
        response = requests.post(url, json=data)
        # Print the response status and JSON content
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except Exception as e:
        print(f"Error occurred while calling analyze_stocks: {e}")


def get_status():
    """
    Function to send a GET request to the `/v1/agent/status` endpoint.
    """
    url = f"{BASE_URL}/v1/agent/status"

    try:
        # Send the GET request
        response = requests.get(url)
        # Print the response status and JSON content
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except Exception as e:
        print(f"Error occurred while calling get_status: {e}")


def main():
    """
    Main function to execute all the tests.
    """
    print("Running tests for FastAPI app endpoints...\n")
    
    # Call each endpoint test
    print("Testing `/v1/agent/analyze` endpoint:")
    analyze_stocks()
    print("\n")
    
    print("Testing `/v1/agent/status` endpoint:")
    get_status()
    print("\n")


if __name__ == "__main__":
    main()
