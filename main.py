from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # If you add CSS later
import uuid # For session IDs
import sqlite3
import json

# Assuming akinator_logic.py is in the same directory
from akinator_logic import AkinatorEfficientWeb 

# --- Configuration ---
DATASET_PATH = "indian_personalities_dataset_30.json" # Make sure this file exists
DEFAULT_STRATEGY = "entropy_sampled" # Choose 'entropy_sampled' or 'simple_heuristic' for Railway
DB_PATH = 'sessions.db'  # You can use ':memory:' for pure in-memory, but file is more robust for dev

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Initialize SQLite DB ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Helper functions for session DB ---
def save_game_session(session_id: str, game: AkinatorEfficientWeb):
    # Serialize game state (probabilities, asked_attributes, questions_asked_count)
    state = {
        'probabilities': game.probabilities,
        'asked_attributes': list(game.asked_attributes),
        'questions_asked_count': game.questions_asked_count
    }
    state_json = json.dumps(state)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('REPLACE INTO sessions (session_id, state_json) VALUES (?, ?)', (session_id, state_json))
    conn.commit()
    conn.close()

def load_game_session(session_id: str, game: AkinatorEfficientWeb):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT state_json FROM sessions WHERE session_id = ?', (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        state = json.loads(row[0])
        game.probabilities = {k: float(v) for k, v in state['probabilities'].items()}
        game.asked_attributes = set(state['asked_attributes'])
        game.questions_asked_count = int(state['questions_asked_count'])
        return True
    return False

def delete_game_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

# --- Dependency to get or create game instance ---
def get_game_session(session_id: str) -> AkinatorEfficientWeb:
    game = AkinatorEfficientWeb(DATASET_PATH, question_selection_strategy=DEFAULT_STRATEGY)
    loaded = load_game_session(session_id, game)
    if not loaded:
        game.reset_game_state()
        save_game_session(session_id, game)
    return game

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Create a new session ID for a new game    
    session_id = str(uuid.uuid4())
    game = get_game_session(session_id) # This will create and reset it
    
    next_question = game.select_next_question()

    if not next_question: # Should not happen on first load unless dataset is tiny/problematic
        return templates.TemplateResponse("game.html", {
            "request": request, 
            "session_id": session_id,
            "question": "I'm already stumped! Dataset issue?", 
            "game_over": True, 
            "message": "Could not start the game properly."
        })

    return templates.TemplateResponse("game.html", {
        "request": request, 
        "session_id": session_id, 
        "question_text": f"Q{game.questions_asked_count + 1}: Is the person {next_question.replace('_', ' ')}?", # +1 because we haven't officially asked it yet
        "attribute_name": next_question,
        "game_over": False
    })

@app.post("/answer", response_class=HTMLResponse)
async def handle_answer(request: Request, session_id: str = Form(...), attribute_name: str = Form(...), answer: str = Form(...)):
    game = get_game_session(session_id)
    if not game or not game.probabilities: # Game state not found or not initialized
         return templates.TemplateResponse("game.html", {
            "request": request, "session_id": "error", "question": "Session error. Please restart.", 
            "game_over": True, "message": "Game session error."})


    user_answer_binary = 1 if answer == "yes" else 0
    
    # Record answer - questions_asked_count is incremented inside record_answer_and_update
    update_result = game.record_answer_and_update(attribute_name, user_answer_binary)
    save_game_session(session_id, game)  # Save after update

    if update_result.get("status") == "contradiction":
        delete_game_session(session_id)
        return templates.TemplateResponse("game.html", {
            "request": request, "session_id": session_id, 
            "question": "", "game_over": True, "message": update_result["message"]
        })

    guess_info = game.get_guess_if_ready()

    if guess_info["type"] == "guess":
        delete_game_session(session_id)
        message = ""
        if guess_info["name"]:
            message = f"I am {guess_info['certainty']*100:.1f}% sure. My guess is: {guess_info['name']}!"
            # Here you could add a form to confirm if the guess was correct
            # For simplicity, we just display the guess.
            # If you want to allow "was I right?", you'd need another endpoint or logic here.
            # For now, if it's a guess, the game ends.
        else:
            message = guess_info.get("message", "I'm stumped!")
        
        return templates.TemplateResponse("game.html", {
            "request": request, "session_id": session_id, 
            "question": "", "game_over": True, "message": message
        })
    else: # Ask more
        next_question = game.select_next_question()
        if not next_question:
            # This means no more distinguishing questions, but we weren't ready to guess
            # (e.g. didn't meet min_questions or certainty threshold)
            # Try a final Hail Mary guess based on current state
            final_name, final_cert = game._get_current_guess_info() # Use internal method for raw info
            message = "I've run out of good questions to ask."
            if final_name and final_cert > 0.01:
                message += f" My best feeling is it might be {final_name} ({final_cert*100:.1f}%)."
            else:
                message += " I'm truly stumped!"
            delete_game_session(session_id)
            return templates.TemplateResponse("game.html", {
                "request": request, "session_id": session_id,
                "question": "", "game_over": True, "message": message
            })

        save_game_session(session_id, game)
        return templates.TemplateResponse("game.html", {
            "request": request, 
            "session_id": session_id, 
            "question_text": f"Q{game.questions_asked_count + 1}: Is the person {next_question.replace('_', ' ')}?",
            "attribute_name": next_question,
            "game_over": False
        })

# --- To run this FastAPI app: uvicorn main:app --reload ---