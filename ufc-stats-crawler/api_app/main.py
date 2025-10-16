from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")
FIGHTER_DIR = DATA_DIR / "fighter_stats"
FIGHT_INFO_DIR = DATA_DIR / "fight_info"
UPCOMING_DIR = DATA_DIR / "upcoming"

app = FastAPI(title="UFC Stats API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def latest_csv(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv"))
    return files[-1] if files else None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(name: str = Query(..., min_length=2)):
    """Search fighters by (case-insensitive) substring."""
    csv = latest_csv(FIGHTER_DIR)
    if not csv or not csv.exists():
        raise HTTPException(503, "No fighter data yet. Run the spiders.")
    df = pd.read_csv(csv)
    mask = df.apply(lambda col: col.astype(str).str.contains(name, case=False, na=False) if col.dtype == "object" else False).any(axis=1)
    out = df[mask].copy()
    # keep the most common/useful columns if present
    cols = [c for c in out.columns if c.lower() in {
        "fighter", "name", "height", "weight", "reach", "stance", "dob",
        "slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc",
        "td_def", "sub_avg", "win", "loss", "draw"
    }]
    if cols:
        out = out[cols]
    return {"count": int(out.shape[0]), "results": out.fillna("").to_dict(orient="records")}

@app.get("/upcoming")
def upcoming():
    csv = latest_csv(UPCOMING_DIR)
    if not csv or not csv.exists():
        raise HTTPException(503, "No upcoming data yet. Run the upcoming spider.")
    df = pd.read_csv(csv)
    return {"count": int(df.shape[0]), "results": df.fillna("").to_dict(orient="records")}
