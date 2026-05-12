"""
Company listing and detail endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Company

router = APIRouter(prefix="/api/companies", tags=["Companies"])


@router.get("")
def list_companies(
    search: str = Query(None, description="Search by ticker or name"),
    sector: str = Query(None, description="Filter by sector"),
    db: Session = Depends(get_db),
):
    """List all registered companies with optional search/filter."""
    q = db.query(Company)

    if search:
        term = f"%{search.upper()}%"
        q = q.filter(
            (Company.ticker.ilike(term)) | (Company.name.ilike(term))
        )

    if sector:
        q = q.filter(Company.sector == sector)

    companies = q.order_by(Company.ticker).all()
    return {
        "count": len(companies),
        "companies": [c.to_dict() for c in companies],
    }


@router.get("/sectors")
def list_sectors(db: Session = Depends(get_db)):
    """List all unique sectors."""
    sectors = (
        db.query(Company.sector)
        .distinct()
        .order_by(Company.sector)
        .all()
    )
    return {"sectors": [s[0] for s in sectors]}


@router.get("/{ticker}")
def get_company(ticker: str, db: Session = Depends(get_db)):
    """Get details for a specific company."""
    company = db.query(Company).filter(Company.ticker == ticker.upper()).first()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    return company.to_dict()
