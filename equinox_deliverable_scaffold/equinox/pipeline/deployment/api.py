"""
Stage 6: Deployment API Service
FastAPI application serving incident intelligence, status updates, clearance re-checks,
and the interactive operator dashboard.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Equinox Corridor Watch API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST_DIR = BASE_DIR / "dashboard" / "frontend" / "dist"
VALID_STATUSES = ("Open", "Assigned", "In Action", "Cleared", "Verified")

# The React bundle is optional during development; when built, FastAPI serves
# its assets at the same origin as the API. The frontend can also use an
# explicit VITE_API_BASE_URL when deployed separately in the future.
if (FRONTEND_DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="dashboard_assets")
if (FRONTEND_DIST_DIR / "evidence-posters").is_dir():
    app.mount("/evidence-posters", StaticFiles(directory=str(FRONTEND_DIST_DIR / "evidence-posters")), name="evidence_posters")
if (FRONTEND_DIST_DIR / "detector-evidence").is_dir():
    app.mount("/detector-evidence", StaticFiles(directory=str(FRONTEND_DIST_DIR / "detector-evidence")), name="detector_evidence")


class Incident(BaseModel):
    incident_id: str
    incident_class: str          # stalled vehicle | illegal parking | debris | congestion | lane blockage
    location: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timestamp: str
    severity: Literal["Normal", "Moderate", "Severe", "Critical"]
    accessibility_score: int = Field(ge=0, le=100)  # 0-100
    recommended_action: str
    status: Literal["Open", "Assigned", "In Action", "Cleared", "Verified"]
    evidence_url: Optional[str] = None
    corridor: Optional[str] = "Main Emergency Route"
    contributing_factors: List[str] = Field(default_factory=list)
    clear_lane_width_m: float = Field(default=1.5, ge=0)
    road_occupancy_pct: float = Field(default=50.0, ge=0, le=100)
    dwell_time_s: float = Field(default=0.0, ge=0)


# Seed initial incidents for the prototype
_INCIDENTS: List[Incident] = [
    Incident(
        incident_id="INC-001",
        incident_class="stalled vehicle",
        location="Hosur Rd, Junction 4",
        lat=12.845,
        lon=77.663,
        timestamp="10:41:52",
        severity="Critical",
        accessibility_score=28,
        recommended_action="Dispatch traffic personnel to Hosur Rd Junction 4 immediately, or route emergency vehicles via Service Rd (Bl. 9) as alternate corridor. Clear lane width has dropped below the 1.5 m emergency threshold for 11 consecutive seconds.",
        status="Open",
        evidence_url="/api/video/clip_01_hosur_j4.mp4",
        corridor="Hosur Rd J4 · northbound",
        contributing_factors=[
            "Road occupancy: 82% of usable width",
            "Clear lane width: 1.1 m (below 1.5 m threshold)",
            "Obstruction dwell time: 11.2 s and rising",
            "Known emergency corridor: yes",
            "Queue length: 6 vehicles stacked"
        ],
        clear_lane_width_m=1.1,
        road_occupancy_pct=82.0,
        dwell_time_s=11.2
    ),
    Incident(
        incident_id="INC-002",
        incident_class="illegal parking",
        location="Service Rd, Bl. 9 Gate",
        lat=12.839,
        lon=77.678,
        timestamp="10:37:10",
        severity="Severe",
        accessibility_score=45,
        recommended_action="Dispatch towing unit to clear illegally parked delivery van at Bl. 9 service road entrance. Corridor passable with caution for narrow ambulances.",
        status="Assigned",
        evidence_url="/api/video/clip_02_service_rd.mp4",
        corridor="Service Rd Bl. 9 · southbound",
        contributing_factors=[
            "Road occupancy: 65% of usable width",
            "Clear lane width: 1.4 m",
            "Obstruction dwell time: 45.0 s",
            "Known emergency corridor: secondary"
        ],
        clear_lane_width_m=1.4,
        road_occupancy_pct=65.0,
        dwell_time_s=45.0
    ),
    Incident(
        incident_id="INC-003",
        incident_class="congestion",
        location="Neeladri Rd, Ramp 2",
        lat=12.851,
        lon=77.671,
        timestamp="10:22:45",
        severity="Moderate",
        accessibility_score=68,
        recommended_action="Activate green wave signal timing on Neeladri Rd Ramp 2 to alleviate slow-moving bottleneck.",
        status="In Action",
        evidence_url="/api/video/clip_03_neeladri_rd.mp4",
        corridor="Neeladri Rd Ramp 2 · eastbound",
        contributing_factors=[
            "Road occupancy: 48% of usable width",
            "Clear lane width: 2.1 m",
            "Obstruction dwell time: 8.5 s"
        ],
        clear_lane_width_m=2.1,
        road_occupancy_pct=48.0,
        dwell_time_s=8.5
    ),
    Incident(
        incident_id="INC-004",
        incident_class="vehicle",
        location="Wipro Junction",
        lat=12.842,
        lon=77.665,
        timestamp="09:58:03",
        severity="Normal",
        accessibility_score=95,
        recommended_action="Corridor clear. Continue routine automated monitoring.",
        status="Verified",
        evidence_url="/api/video/clip_04_wipro_jct.mp4",
        corridor="Wipro Junction · all approach lanes",
        contributing_factors=[
            "Road occupancy: 18% of usable width",
            "Clear lane width: 3.8 m",
            "Corridor status: Clear"
        ],
        clear_lane_width_m=3.8,
        road_occupancy_pct=18.0,
        dwell_time_s=0.0
    )
]


@app.get("/incidents", response_model=List[Incident])
def list_incidents():
    return _INCIDENTS


@app.get("/incidents/{incident_id}", response_model=Incident)
def get_incident(incident_id: str):
    for inc in _INCIDENTS:
        if inc.incident_id == incident_id:
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


@app.post("/incidents", response_model=Incident)
def create_incident(incident: Incident):
    if any(existing.incident_id == incident.incident_id for existing in _INCIDENTS):
        raise HTTPException(status_code=409, detail="Incident ID already exists")
    _INCIDENTS.append(incident)
    return incident


@app.patch("/incidents/{incident_id}/status")
def update_status(incident_id: str, status: str):
    if status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {list(VALID_STATUSES)}")

    for inc in _INCIDENTS:
        if inc.incident_id == incident_id:
            previous_status = inc.status
            inc.status = status
            # Clearance verification step (plan §1.6 / §11)
            if status == "Cleared" and previous_status != "Cleared":
                # Simulated auto re-check improving score
                inc.accessibility_score = min(100, inc.accessibility_score + 40)
                if inc.accessibility_score >= 80:
                    inc.severity = "Normal"
                    inc.recommended_action = "Re-check pass confirmed obstruction cleared. Verified safe for emergency vehicle passage."
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


@app.post("/incidents/{incident_id}/recheck")
def recheck_clearance(incident_id: str):
    for inc in _INCIDENTS:
        if inc.incident_id == incident_id:
            # Simulate re-checking the road corridor frame
            inc.status = "Verified"
            inc.severity = "Normal"
            inc.accessibility_score = 98
            inc.clear_lane_width_m = 3.5
            inc.road_occupancy_pct = 15.0
            inc.dwell_time_s = 0.0
            inc.recommended_action = "Verification scan complete: Corridor is fully clear. Passage verified."
            inc.contributing_factors = [
                "Re-check scan result: Passed",
                "Usable clear width: 3.5 m",
                "Obstruction cleared: yes"
            ]
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


@app.get("/health")
def health():
    models_dir = BASE_DIR / "models"
    model_exists = (models_dir / "severity_model.pkl").exists()
    return {
        "status": "ok",
        "incidents_tracked": len(_INCIDENTS),
        "severity_model_ready": model_exists
    }


# Serve raw video files for evidence playback
@app.get("/api/video/{filename}")
def serve_video(filename: str):
    video_dir = (BASE_DIR / "data" / "raw").resolve()
    # Only serve direct media files from data/raw; reject traversal and
    # arbitrary extensions before constructing the path.
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Video file not found")
    if Path(filename).suffix.lower() not in {".mp4", ".mov"}:
        raise HTTPException(status_code=404, detail="Video file not found")
    video_path = (video_dir / filename).resolve()
    try:
        video_path.relative_to(video_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="Video file not found")
    if video_path.is_file():
        media_type = "video/quicktime" if video_path.suffix.lower() == ".mov" else "video/mp4"
        return FileResponse(str(video_path), media_type=media_type)
    # Fallback to empty/mock if video file not yet generated
    raise HTTPException(status_code=404, detail="Video file not found")


# Root route serves the interactive operator dashboard UI
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    built_dashboard = FRONTEND_DIST_DIR / "index.html"
    if built_dashboard.is_file():
        return FileResponse(str(built_dashboard))
    html_path = BASE_DIR / "dashboard" / "dashboard_mockup.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return "<h1>Equinox API Running</h1><p>Dashboard HTML file missing.</p>"
