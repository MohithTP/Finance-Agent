# Finance Agent - Indian Stock Market Analyst

A multi-agent financial analysis system built with **AGNO** and **Google Gemini**, specialized in identifying and analyzing long-term investment opportunities in the Indian stock market (NSE/BSE).

## 🚀 Features

- **Indian Market Focus:** Automatically screens for top-performing Indian stocks using a dedicated market screener.
- **Multi-Agent Collaboration:** 
    - **Financial Analyst Agent:** Performs deep fundamental analysis using income statements, balance sheets, and cash flow data.
    - **Web Search Agent:** Gathers real-time news and sector trends using DuckDuckGo.
- **Analyst Scoring:** Provides a proprietary **Analyst Score (1-10)** based on fundamental health (50%) and future outlook/sentiment (50%).
- **Interactive Web Interface:** Built with FastAPI and Jinja2 for easy interaction.

## 🛠️ Technical Stack

- **Framework:** [AGNO](https://github.com/agno-ai/agno)
- **Model:** Google Gemini
- **Backend:** FastAPI, Uvicorn
- **Frontend Templates:** Jinja2
- **Data Sources:** Financial Datasets API, DuckDuckGo

## ⚙️ Setup

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
FMP_API_KEY=your_financial_datasets_api_key
DEFAULT_MODEL=gemini-2.5-flash(use any gemini text-out model)
```

### 3. Installation
Using `uv`:
```bash
uv sync
```
Or using `pip`:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### Run the Agent Team (CLI)
To run the automated analysis task defined in `agent_setup.py`:
```bash
python agent_setup.py
```

## 📁 Project Structure

- `agent_setup.py`: Definitions for agents, tools, and the finance team.
- `templates/`: HTML templates for the web interface.
- `pyproject.toml` / `requirements.txt`: Project dependencies.
