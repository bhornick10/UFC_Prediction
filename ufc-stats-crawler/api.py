from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import pandas as pd
import glob, os, subprocess, shlex
from rapidfuzz import process, fuzz

DATA_DIR = os.path.join(os.getcwd(), "data")

app = FastAPI(title="UFC Stats API")

# Allow browser calls if you hit API directly; nginx proxy will make this unnecessary
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None

def _load_fighter_table() -> pd.DataFrame:
    # The fighter spider writes CSVs into data/fighter_stats/
    csv_path = _latest_file(os.path.join(DATA_DIR, "fighter_stats", "*.csv"))
    if not csv_path:
        raise FileNotFoundError("No fighter_stats CSV found. Run a crawl first.")
    df = pd.read_csv(csv_path)
    # normalize a name column guess
    for col in ["fighter", "fighter_name", "name", "Fighter", "Name"]:
        if col in df.columns:
            df.rename(columns={col: "name"}, inplace=True)
            break
    if "name" not in df.columns:
        # just create a name view if repo uses structured fields
        df["name"] = df.get("First Name", "") + " " + df.get("Last Name", "")
        df["name"] = df["name"].str.strip()
    df["name_norm"] = df["name"].fillna("").str.strip().str.lower()
    return df

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/fighter")
def fighter(name: str = Query(..., description="Fighter full or partial name"),
            limit: int = 3):
    try:
        df = _load_fighter_table()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Fast path: substring
    name_norm = name.strip().lower()
    subset = df[df["name_norm"].str.contains(name_norm, na=False)]
    if subset.empty:
        # Fuzzy match
        choices = df["name"].fillna("").tolist()
        matches = process.extract(name, choices, scorer=fuzz.WRatio, limit=limit)
        rows: List[Dict[str, Any]] = []
        for match_name, score, idx in matches:
            row = df.iloc[idx].to_dict()
            row["_match_score"] = int(score)
            rows.append(row)
        return {"query": name, "exact": False, "results": rows, "message": "Fuzzy matches"}
    else:
        # Return up to `limit` distinct names from substring match
        names = subset["name"].dropna().unique().tolist()[:limit]
        rows = [subset[subset["name"] == n].iloc[0].to_dict() for n in names]
        return {"query": name, "exact": True, "results": rows, "message": "Substring matches"}

def _run_scrapy(spider: str, extra_args: Optional[List[str]] = None) -> int:
    cmd = ["python", "-m", "scrapy", "crawl", spider]
    if extra_args: cmd += extra_args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd, cwd=os.getcwd())

@app.post("/scrape/fighters")
def scrape_fighters():
    os.makedirs(os.path.join(DATA_DIR, "fighter_stats"), exist_ok=True)
    rc = _run_scrapy("ufcFighters")
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"scrapy exited {rc}")
    return {"ok": True}

@app.post("/scrape/fights")
def scrape_fights():
    os.makedirs(os.path.join(DATA_DIR, "fight_info"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "fight_stats"), exist_ok=True)
    rc = _run_scrapy("ufcFights")
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"scrapy exited {rc}")
    return {"ok": True}

@app.post("/scrape/upcoming")
def scrape_upcoming():
    os.makedirs(os.path.join(DATA_DIR, "upcoming"), exist_ok=True)
    rc = _run_scrapy("upcoming")
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"scrapy exited {rc}")
    return {"ok": True}

