import type { Health, Incident, IncidentStatus } from "@/lib/types";

const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE = configuredBase ? configuredBase.replace(/\/$/, "") : "";

function endpoint(path: string) {
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(endpoint(path), {
    ...options,
    headers: { Accept: "application/json", ...options?.headers },
  });
  if (!response.ok) {
    const message = await response.text().catch(() => "");
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getHealth: () => request<Health>("/health"),
  getIncidents: () => request<Incident[]>("/incidents"),
  updateStatus: (incidentId: string, status: IncidentStatus) =>
    request<Incident>(`/incidents/${encodeURIComponent(incidentId)}/status?status=${encodeURIComponent(status)}`, {
      method: "PATCH",
    }),
  recheck: (incidentId: string) =>
    request<Incident>(`/incidents/${encodeURIComponent(incidentId)}/recheck`, { method: "POST" }),
  mediaUrl: (path?: string | null) => (!path ? undefined : path.startsWith("http") ? path : endpoint(path)),
};
