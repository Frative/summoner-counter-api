from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import champion, team

import champion
import team
from fastapi import Query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["*"] si te da igual
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/champion/{champion_name}/counters")
def get_counters(champion_name: str):
    counters = champion.get_counters_for_champion(champion_name)
    return {"counters": counters}

@app.get("/team/counters")
def get_team_counters(
    team_a: list[str] = Query(..., description="Lista de campeones del equipo A"),
    team_b: list[str] = Query(..., description="Lista de campeones del equipo B")
):
    if len(team_a) != 5 or len(team_b) != 5:
        return {"error": "Cada equipo debe tener exactamente 5 campeones."}
    matrix = team.predict_team_matchup(team_a, team_b)
    return {"counters": matrix.tolist()}