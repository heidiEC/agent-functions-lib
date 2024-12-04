from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import json

app = FastAPI(title="Agent Functions API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FunctionRequest(BaseModel):
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]

@app.post("/execute")
async def execute_function(request: FunctionRequest):
    """Execute a Python agent function"""
    try:
        # Import the function dynamically
        module_path, function_name = request.function_name.rsplit('.', 1)
        module = __import__(module_path, fromlist=[function_name])
        func = getattr(module, function_name)
        
        # Execute the function
        result = func(*request.args, **request.kwargs)
        
        # Handle non-serializable results
        try:
            json.dumps(result)
        except TypeError:
            result = str(result)
            
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_functions")
async def list_functions():
    """List all available Python agent functions"""
    from agent_functions import AgentFunction
    
    functions = []
    for func in AgentFunction.registry:
        functions.append({
            "name": func.__name__,
            "category": getattr(func, "category", None),
            "description": getattr(func, "description", None),
            "agent_triggers": getattr(func, "agent_triggers", [])
        })
    return {"functions": functions}

def start_service(host: str = "localhost", port: int = 8000):
    """Start the FastAPI service"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_service()
