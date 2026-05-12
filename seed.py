"""
Seed the database with company metadata parsed from CSV filenames.
Market data stays in CSV files and is loaded on-demand by data_service.
"""

import re
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from database.models import Company
from config import DATA_DIR

logger = logging.getLogger(__name__)

# Map ticker patterns to sectors (best-effort)
SECTOR_MAP = {
    "HDFCBANK|ICICIBANK|SBIN|KOTAKBANK|AXISBANK|INDUSINDBK|BANDHANBNK|FEDERALBNK|IDFCFIRSTB|RBLBANK|KARURVYSYA|SOUTHBANK": "Banking",
    "BAJFINANCE|BAJAJFINSV|CHOLAFIN|MUTHOOTFIN|LICHSGFIN|CANFINHOME|MANAPPURAM|PFC|RECLTD|HUDCO": "Finance",
    "RELIANCE|ONGC|BPCL|IOC|HPCL|GAIL|PETRONET|IGL|GUJGASLTD|GSPL|MGL": "Oil & Gas",
    "TCS|INFY|WIPRO|HCLTECH|TECHM|LTIM|COFORGE|MPHASIS|PERSISTENT|KPITTECH|TATAELXSI|ECLERX|HAPPSTMNDS|SONATSOFTW|NEWGEN": "IT",
    "SUNPHARMA|DRREDDY|CIPLA|DIVISLAB|LUPIN|AUROPHARMA|BIOCON|TORNTPHARM|ALKEM|GLENMARK|ZYDUSLIFE": "Pharma",
    "TATAMOTORS|MARUTI|BAJAJ-AUTO|EICHERMOT|HEROMOTOCO|TVSMOTOR|ASHOKLEY|ESCORTS|M&M|APOLLOTYRE|CEATLTD|JKTYRE|BALKRISIND": "Auto",
    "TATASTEEL|JSWSTEEL|HINDALCO|VEDL|SAIL|JINDALSTEL|NATIONALUM|NMDC|COALINDIA|HINDZINC|MOIL|WELCORP": "Metals & Mining",
    "ASIANPAINT|BERGEPAINT|PIDILITIND|ASTRAL|SUPREMEIND|APLAPOLLO": "Paints & Building",
    "HINDUNILVR|ITC|BRITANNIA|NESTLEIND|DABUR|MARICO|COLPAL|GODREJCP|TATACONSUM|EMAMILTD|RADICO|UNITDSPR": "FMCG",
    "ADANIENT|ADANIPORTS|ADANIGREEN|ADANIPOWER|ADANITRANS": "Adani Group",
    "TITAN|BATAINDIA|TRENT|PAGEIND|SHOPERSTOP": "Retail & Fashion",
    "DLF|OBEROIRLTY|PRESTIGE|GODREJIND": "Real Estate",
    "POWERGRID|NTPC|NHPC|SJVN|TATAPOWER|IRFC": "Power & Utilities",
    "LT|SIEMENS|ABB|BOSCHLTD|CUMMINSIND|THERMAX|HAL|BEL|BHEL|GRINFRA|KEC|PNCINFRA|KNRCON|DILIPBUILD|IRB": "Engineering & Defence",
    "NAUKRI|ZOMATO|PAYTM|NYKAA|POLICYBZR|DELHIVERY|JUSTDIAL|AFFLE|DEVYANI": "Internet & Tech",
    "HDFCLIFE|SBILIFE|ICICIBANK": "Insurance",
    "INDIGO|IRCTC": "Transport",
    "APOLLOHOSP|INDHOTEL|EIHOTEL|CHALET|LEMONTREE|WESTLIFE": "Hospitality & Health",
    "ACC|AMBUJACEM|ULTRACEMCO|SHREECEM|GRASIM": "Cement",
    "BHARTIARTL|CROMPTON|HAVELLS|VOLTAS|BLUESTARCO|WHIRLPOOL|VGUARD|KEI|POLYCAB": "Telecom & Consumer Durables",
}


def _ticker_from_filename(filename: str) -> str:
    """Extract ticker from 'cleaned_NSE_RELIANCE-EQ.csv' → 'RELIANCE'."""
    match = re.match(r"cleaned_NSE_(.+)-EQ\.csv", filename)
    if match:
        return match.group(1)
    return filename.replace("cleaned_NSE_", "").replace("-EQ.csv", "").replace(".csv", "")


def _sector_for_ticker(ticker: str) -> str:
    for pattern, sector in SECTOR_MAP.items():
        if ticker in pattern.split("|"):
            return sector
    return "Other"


def seed_companies(db: Session):
    """Scan the data directory and register all valid companies."""
    if not DATA_DIR.exists():
        logger.warning(f"Data directory not found: {DATA_DIR}")
        return 0

    csv_files = sorted(DATA_DIR.glob("cleaned_NSE_*-EQ.csv"))
    added = 0

    for csv_path in csv_files:
        # Skip empty files (header-only, ≤100 bytes)
        if csv_path.stat().st_size <= 100:
            continue

        ticker = _ticker_from_filename(csv_path.name)
        existing = db.query(Company).filter_by(ticker=ticker).first()
        if existing:
            continue

        company = Company(
            name=ticker.replace("-", " ").title(),
            ticker=ticker,
            sector=_sector_for_ticker(ticker),
            file_path=str(csv_path),
        )
        db.add(company)
        added += 1

    db.commit()
    logger.info(f"Seeded {added} companies from {len(csv_files)} CSV files")
    return added
