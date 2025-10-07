# main.py (Ensure this file is exactly as provided previously, including the initialize_team_leader function)

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import the logic and definitions from your agent file
from agent_setup import financial_analyst_agent, web_agent, Team, ReasoningTools, FMP_API_KEY_ENV

# Load environment variables (Render injects them, but this is good for local testing)
load_dotenv() 

# Initialize FastAPI app
app = FastAPI(title="Indian Financial Agent API", version="1.0")

# --- (initialize_team_leader function goes here - copy from previous response) ---
def initialize_team_leader():
    # Environment variables are managed by Render in production
    
    # 1. Initialize Agents
    # Note: Using .get() for safety.
    fa_agent = financial_analyst_agent  
    fa_agent.tools[0].api_key = os.environ.get(FMP_API_KEY_ENV) 

    wa_agent = web_agent

    # 2. Re-create Team Leader with correct members
    team_leader = Team(
        name="Reasoning Finance Team Leader",
        mode="coordinate",
        model=fa_agent.model,
        members=[wa_agent, fa_agent],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=[
            "Use tables to display data.",
            "Provide the analysis and recommendations clearly, including the Analyst Score.",
        ],
        markdown=True,
        show_members_responses=False, 
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        success_criteria="The team has successfully identified and analyzed promising Indian stocks for long-term investment, providing a rationale and the Analyst Score.",
    )
    return team_leader


@app.post("/analyze")
async def analyze_indian_stocks(task: str):
    """
    Triggers the multi-agent system to perform Indian stock analysis.
    """
    # Note: os.environ.get("GEMINI_API_KEY") must be set in Render's environment variables.
    if not os.environ.get(FMP_API_KEY_ENV) or not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Required API keys (FMP_API_KEY, GEMINI_API_KEY) are not configured in the environment.")

    try:
        team_leader = initialize_team_leader()
        
        # Run the agent synchronously. The Free Tier timeout is generous but not infinite.
        final_response = team_leader.run(
            task,
            stream=False, 
            show_full_reasoning=False,
        )
        
        return JSONResponse(
            content={"analysis": final_response, "status": "Success"}, 
            status_code=200
        )

    except Exception as e:
        print(f"Agent Execution Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Agent execution failed. Internal error details: {str(e)}"
        )
        
@app.get("/health")
def health_check():
    # The health check keeps the service from spinning down if regularly pinged (but consumes 750 free hours)
    return {"status": "ok"}