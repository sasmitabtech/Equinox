export type Severity = "Normal" | "Moderate" | "Severe" | "Critical";
export type IncidentStatus = "Open" | "Assigned" | "In Action" | "Cleared" | "Verified";

export interface Incident {
  incident_id: string;
  incident_class: string;
  location: string;
  lat: number;
  lon: number;
  timestamp: string;
  severity: Severity;
  accessibility_score: number;
  recommended_action: string;
  status: IncidentStatus;
  evidence_url?: string | null;
  corridor?: string | null;
  contributing_factors: string[];
  clear_lane_width_m: number;
  road_occupancy_pct: number;
  dwell_time_s: number;
}

export interface Health {
  status: "ok" | string;
  incidents_tracked: number;
  severity_model_ready: boolean;
}
