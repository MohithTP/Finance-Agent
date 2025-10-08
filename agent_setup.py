from textwrap import dedent
import requests
import os
import json # Added json import for better tool handling
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.google import Gemini
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.toolkit import Toolkit
from agno.tools import tool
from agno.utils.log import log_error
from dotenv import load_dotenv

# --- Configuration ---
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

# Use a harmonized key name for consistency
FMP_API_KEY_ENV = "FMP_API_KEY"
# Use a generic base URL for the custom API
BASE_URL = "https://api.financialdatasets.ai" 


class FinancialDatasetsTools(Toolkit):
    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
    
        # Removed unused 'enable_xxx' flags for cleaner code
        super().__init__(name="financial_datasets_tools", **kwargs)

        self.api_key: Optional[str] = api_key or os.getenv(FMP_API_KEY_ENV)
        if not self.api_key:
            log_error(
                f"{FMP_API_KEY_ENV} not set. Please set the {FMP_API_KEY_ENV} environment variable."
            )

        self.base_url = BASE_URL 

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Makes a request to the Financial Datasets API.
        """
        if not self.api_key:
            log_error("No API key provided. Cannot make request.")
            return json.dumps({"error": "API key not set"}) # Return JSON error for easier LLM parsing

        # Headers assuming the custom API uses an X-API-KEY header
        headers = {"X-API-KEY": self.api_key} 
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            # Return JSON structure for easier LLM processing (assuming API returns JSON)
            return response.text
        except requests.exceptions.RequestException as e:
            # Enhanced error message for LLM visibility
            error_details = {
                "error": "Request Failed",
                "url": url,
                "details": str(e),
                "response_text": response.text if 'response' in locals() else None
            }
            log_error(f"Error making request to {url}: {str(e)}")
            return json.dumps(error_details)
    
    # --- ENHANCEMENT 1: Dedicated Indian Market Screener Tool (Addresses the core failure) ---
    @tool(name="get_indian_market_screen", 
          description="Finds top-performing stocks. Use this tool *first* to identify Indian stocks (e.g., set 'country=IN', 'change_over_period=1d', and 'min_change_percent=3.0').")
    def get_indian_market_screen(
        self, 
        country: str = "IN", 
        change_over_period: str = "1d",
        min_change_percent: float = 3.0,
        limit: int = 10
    ) -> str:
        """
        Retrieves a list of top gainers for a specific market by filtering for high momentum stocks.
        Note: The 'country' parameter is crucial for Indian stocks.
        """
        params = {
            "country": country, 
            "min_change_percent": min_change_percent,
            "period": change_over_period,
            "limit": limit
        }
        # Assuming the API has a /market/screener endpoint
        return self._make_request("market/screener", params)

    # --- Existing Financial Statement Tools (with explicit Indian market context) ---

    @tool(name="get_income_statements", description="Get income statements for an Indian ticker. Use .NS or .BO suffix (e.g., TCS.NS).")
    def get_income_statements(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/income-statements", params)

    @tool(name="get_balance_sheets", description="Get balance sheets for an Indian ticker. Use .NS or .BO suffix.")
    def get_balance_sheets(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/balance-sheets", params)

    @tool(name="get_cash_flow_statements", description="Get cash flow statements for an Indian ticker. Use .NS or .BO suffix.")
    def get_cash_flow_statements(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/cash-flow-statements", params)
    
    # ... (Keep get_segmented_financials, get_financial_metrics) ...

    @tool(name="get_company_info", description="Get company information for an Indian ticker (use .NS or .BO).")
    def get_company_info(self, ticker: str) -> str:
        params = {"ticker": ticker}
        return self._make_request("company", params)

    # ... (Keep get_crypto_prices, get_earnings, get_insider_trades, get_institutional_ownership) ...

    @tool(name="get_news", description="Get market news. Use 'IN' or filter by specific Indian ticker (e.g., RELIANCE.NS) to focus on the Indian market.")
    def get_news(self, ticker: Optional[str] = None, limit: int = 50) -> str:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._make_request("news", params)

    @tool(name="get_stock_prices", description="Get stock prices for an Indian ticker. Use .NS or .BO suffix (e.g., RELIANCE.NS).")
    def get_stock_prices(self, ticker: str, interval: str = "1d", limit: int = 100) -> str:
        params = {"ticker": ticker, "interval": interval, "limit": limit}
        return self._make_request("prices", params)

    @tool(name="search_tickers", description="Search for Indian tickers based on a query (e.g., 'Reliance Industries India').")
    def search_tickers(self, query: str, limit: int = 10) -> str:
        params = {"query": query, "limit": limit}
        return self._make_request("search", params)

    @tool(name="get_sec_filings", description="Get SEC filings. Primarily for Indian companies with US listings (ADRs).")
    def get_sec_filings(self, ticker: str, form_type: Optional[str] = None, limit: int = 50) -> str:
        params: Dict[str, Any] = {"ticker": ticker, "limit": limit}
        if form_type:
            params["form_type"] = form_type
        return self._make_request("sec-filings", params)


# --- AGENT DEFINITIONS ---

financial_analyst_agent = Agent(
    name="Financial Analyst Agent",
    role="Analyzes financial data, market trends, and company performance to provide investment insights for the **Indian Market (NSE/BSE)**.",
    model=Gemini(id=os.environ["DEFAULT_MODEL"]),
    tools=[FinancialDatasetsTools(api_key=os.environ[FMP_API_KEY_ENV])],
    instructions=dedent("""
        1. **Initial Screen:** Use the 'get_indian_market_screen' tool first to identify top gainers from the previous day's trends (country=IN, 1d period).
        2. **Filter & Select:** Choose 3-5 of the most promising large-cap/mid-cap companies from the screener results.
        3. **Fundamental Analysis (Tools):** Use financial tools (e.g., `get_income_statements`, `get_financial_metrics`) for deep fundamental analysis of the selected tickers (remember the .NS/.BO suffix).
        4. **Qualitative Analysis (Delegation):** If necessary, request the 'Web Search Agent' to find the latest news, sector outlook, and future growth drivers for the selected companies.
        5. **Novel Feature (The Analyst Score):** In your final output table, include an **Analyst Score (1-10)** based on a combination of: 
           - **Fundamental Health (50% weight):** Profitability, Debt/Equity, Cash Flow.
           - **Future Outlook & News Sentiment (50% weight):** Market trends, sector growth, and recent news gathered by the Web Search Agent.
        6. Use tables to display data and provide a concise rationale for each recommendation.
    """),
    #add_datetime_to_instructions=True,
)

web_agent = Agent(
    name="Web Search Agent",
    role="Handle web search requests for real-time and unstructured data, especially **recent news, sector trends, and future outlook for Indian companies**.",
    model=Gemini(id=os.environ["DEFAULT_MODEL"]),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources. Focus search queries on the **Indian stock market** (e.g., 'future of Indian IT sector', 'recent news for TCS India'). Do not attempt to scrape lists of top gainers, rely on the Analyst Agent's market screen for that.",
    #add_datetime_to_instructions=True,
)


team_leader = Team(
    name="Reasoning Finance Team Leader",
    #mode="coordinate",
    model=Gemini(id=os.environ["DEFAULT_MODEL"]),
    members=[web_agent, financial_analyst_agent],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Use tables to display data.",
        "Provide the analysis and recommendations clearly, using the requested structure (including the Analyst Score).",
    ],
    markdown=True,
    show_members_responses=True,
    enable_agentic_state=True,
    #add_datetime_to_instructions=True,
    #success_criteria="The team has successfully identified and analyzed promising Indian stocks for long-term investment, providing a rationale and the Analyst Score.",
)


task = """\
Identify top-performing Indian companies based on the previous day's stock trends.
Analyze their financial performance, market position, and future outlook. Based on this analysis,
provide a list of promising stocks for long-term investment, along with a brief rationale for each, including the new 'Analyst Score' (1-10)."""


if __name__ == "__main__":
    print("Starting AGNO agent team to analyze Indian stocks...")
    team_leader.print_response(
        task,
        stream=True,
        stream_intermediate_steps=True,
        show_full_reasoning=True,
    )