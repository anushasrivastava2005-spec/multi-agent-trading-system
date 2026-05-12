"""
FastAPI application entry point.

Run with:
    cd backend
    uvicorn main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.connection import init_db, SessionLocal
from database.seed import seed_companies
from routers import companies, market_data, predictions

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("quantagent")


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and seed companies on startup."""
    logger.info("Initializing database...")
    init_db()

    db = SessionLocal()
    try:
        count = seed_companies(db)
        total = db.query(__import__("database.models", fromlist=["Company"]).Company).count()
        logger.info(f"Database ready — {total} companies ({count} new)")
    finally:
        db.close()

    yield  # App runs here

    logger.info("Shutting down QuantAgent")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QuantAgent API",
    description="LLM-powered multi-agent quantitative trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(companies.router)
app.include_router(market_data.router)
app.include_router(predictions.router)


@app.get("/")
def root():
    return {
        "name": "QuantAgent API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "companies": "/api/companies",
            "market_data": "/api/data/{ticker}",
            "predict": "/api/predict/{ticker}",
            "predictions": "/api/predictions",
            "backtest": "/api/backtest/{ticker}",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
