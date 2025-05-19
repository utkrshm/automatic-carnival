from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates # For a slightly cleaner HTML serving
from pydantic import BaseModel
import uuid
from typing import Dict, Any

# Assuming akinator_logic.py is in the same directory
from akinator_logic import Akinator 

app = FastAPI()

# In-memory storage for game sessions. For production, use a more persistent store.
game_sessions: Dict[str, Akinator] = {}

# Paths to your data files
DATASET_PATH = "indian_personalities_dataset_30.json"
QUESTIONS_PATH = "questions_30.json"

# For serving HTML
templates = Jinja2Templates(directory="templates") # Create a 'templates' directory

class AnswerPayload(BaseModel):
    session_id: str
    attribute_key: str
    answer_value: float # 1.0 (yes), 0.0 (no), 0.75 (probably), 0.25 (probably not)

class GuessConfirmationPayload(BaseModel):
    session_id: str
    guessed_character_name: str
    user_confirms_correct: bool


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # Simple HTML page content
    # For a real app, serve this from a file using Jinja2Templates or StaticFiles
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/start_game")
async def start_game_session():
    session_id = str(uuid.uuid4())
    try:
        akinator_instance = Akinator(dataset_path=DATASET_PATH, questions_path=QUESTIONS_PATH)
        game_sessions[session_id] = akinator_instance
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Akinator: {str(e)}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset or questions file not found. Ensure 'indian_personalities_dataset_30.json' and 'questions_30.json' exist.")

    initial_state = akinator_instance.start_game()
    return {"session_id": session_id, **initial_state}

@app.post("/answer")
async def submit_answer(payload: AnswerPayload):
    if payload.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    akinator_instance = game_sessions[payload.session_id]
    
    if not (0.0 <= payload.answer_value <= 1.0):
         raise HTTPException(status_code=400, detail="Invalid answer value. Must be 0, 0.25, 0.75, or 1.")

    game_state = akinator_instance.process_answer(payload.attribute_key, payload.answer_value)
    return {"session_id": payload.session_id, **game_state}

@app.post("/confirm_guess")
async def confirm_akinator_guess(payload: GuessConfirmationPayload):
    if payload.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    akinator_instance = game_sessions[payload.session_id]

    if payload.user_confirms_correct:
        # Game won
        response_data = {
            "session_id": payload.session_id,
            "status": "finished",
            "message": f"Great! I knew it was {payload.guessed_character_name}!",
            "guess": payload.guessed_character_name,
            "certainty": 1.0, # Or actual certainty if available and relevant
            "top_candidates": akinator_instance._get_top_candidates(5)
        }
        # Clean up session
        del game_sessions[payload.session_id] 
        return response_data
    else:
        # Akinator was wrong, ask it to process mistaken guess
        game_state = akinator_instance.process_mistaken_guess(payload.guessed_character_name)
        return {"session_id": payload.session_id, **game_state}

# To run: uvicorn main:app --reload
# Ensure you have 'templates/index.html'    