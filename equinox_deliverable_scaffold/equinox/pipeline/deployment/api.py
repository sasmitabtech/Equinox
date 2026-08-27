"""
Stage 6: Deployment API Service
FastAPI application serving incident intelligence, status updates, clearance re-checks,
and the interactive operator dashboard.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Equinox Corridor Watch API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Incident(BaseModel):
    incident_id: str
    incident_class: str          # stalled vehicle | illegal parking | debris | congestion | lane blockage
    location: str
    lat: float
    lon: float
    timestamp: str
    severity: str                 # Normal | Moderate | Severe | Critical
    accessibility_score: int      # 0-100
    recommended_action: str
    status: str                   # Open | Assigned | In Action | Cleared | Verified
    evidence_url: Optional[str] = None
    corridor: Optional[str] = "Main Emergency Route"
    contributing_factors: List[str] = []
    clear_lane_width_m: float = 1.5
    road_occupancy_pct: float = 50.0
    dwell_time_s: float = 0.0


# Seed initial incidents for the prototype
_INCIDENTS: List[Incident] = [
    Incident(
        incident_id="INC-001",
        incident_class="stalled vehicle",
        location="Hosur Rd — Junction 4",
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
        location="Service Rd — Bl. 9 Gate",
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
        location="Neeladri Rd — Ramp 2",
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
    _INCIDENTS.append(incident)
    return incident


@app.patch("/incidents/{incident_id}/status")
def update_status(incident_id: str, status: str):
    valid_statuses = ["Open", "Assigned", "In Action", "Cleared", "Verified"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")

    for inc in _INCIDENTS:
        if inc.incident_id == incident_id:
            inc.status = status
            # Clearance verification step (plan §1.6 / §11)
            if status == "Cleared":
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
    video_path = BASE_DIR / "data" / "raw" / filename
    if video_path.exists():
        return FileResponse(str(video_path), media_type="video/mp4")
    # Fallback to empty/mock if video file not yet generated
    raise HTTPException(status_code=404, detail="Video file not found")


# Root route serves the interactive operator dashboard UI
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = BASE_DIR / "dashboard" / "dashboard_mockup.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return "<h1>Equinox API Running</h1><p>Dashboard HTML file missing.</p>"
