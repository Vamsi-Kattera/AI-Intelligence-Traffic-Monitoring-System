from fastapi import FastAPI
from backend.database.analytics_data import analytics_data
from backend.routes.stream_routes import router as stream_router
from fastapi.middleware.cors import CORSMiddleware
import json
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(stream_router)
@app.get("/")
def home():
    return {
        "message": "Traffic Intelligence System API"
    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
@app.get("/analytics")
def get_analytics():
    with open(
        "backend/database/live_analytics.json",
        "r"
    ) as file:
        data = json.load(file)
    return data