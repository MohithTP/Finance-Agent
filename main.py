import os
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Import the logic and definitions from your agent file
from agent_setup import financial_analyst_agent, web_agent, Team, ReasoningTools, FMP_API_KEY_ENV

# --- Configuration ---
load_dotenv() 

# Initialize FastAPI app
app = FastAPI(title="Indian Financial Agent API", version="1.0")

# Configure templates directory
# You must create a directory named 'templates' in your project root
templates = Jinja2Templates(directory="templates") 

# --- Agent Initialization (same as before) ---
def initialize_team_leader():
    # ... (Keep the exact implementation of your initialize_team_leader function) ...
    fa_agent = financial_analyst_agent  
    fa_agent.tools[0].api_key = os.environ.get(FMP_API_KEY_ENV) 
    wa_agent = web_agent

    team_leader = Team(
        name="Reasoning Finance Team Leader",
        model=fa_agent.model,
        members=[wa_agent, fa_agent],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=[
            "Use tables to display data.",
            "Provide the analysis and recommendations clearly, including the Analyst Score.",
        ],
        markdown=True,
        show_members_responses=False, 
        enable_agentic_state=True,
    )
    return team_leader

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main web interface."""
    # Pass an empty context to the template initially
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "analysis_result": None}
    )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_indian_stocks(request: Request, task: str = Form(...)):
    """
    Triggers the multi-agent system and returns the result to the main page.
    Uses Form(...) to read the 'task' field from the HTML form post.
    """
    if not os.environ.get(FMP_API_KEY_ENV) or not os.environ.get("GEMINI_API_KEY"):
        error_detail = "Required API keys are not configured in the environment."
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "analysis_result": f"<div class='error-message'>{error_detail}</div>"}
        )

    try:
        team_leader = initialize_team_leader()
        
        # This will be the full markdown string analysis
        final_response = team_leader.run(
            task,
            stream=False, 
            show_full_reasoning=False,
        )
        
        # Return the analysis back to the index.html template
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "analysis_result": final_response}
        )

    except Exception as e:
        error_detail = f"Agent execution failed. Internal error: {str(e)}"
        print(f"Agent Execution Error: {e}")
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "analysis_result": f"<div class='error-message'>{error_detail}</div>"}
        )

# Keep the health check for deployment services
@app.get("/health")
def health_check():
    return {
        "message": "Indian Finance Agent API is running.",
        "endpoints": {"health_check": "/health (GET)", "analysis": "/analyze (POST)"}
    }