import { motion } from "motion/react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Incident } from "@/lib/types";

const severityClasses = {
  Critical: "bg-[#fdebec] text-[#9f2f2d]",
  Severe: "bg-[#fceee3] text-[#a65328]",
  Moderate: "bg-[#fbf3db] text-[#956400]",
  Normal: "bg-[#edf3ec] text-[#346538]",
};

export function SeverityBadge({ severity }: { severity: Incident["severity"] }) {
  return <Badge className={cn("border-0 px-2 py-1 text-[10px] font-semibold tracking-[0.08em]", severityClasses[severity])}>{severity}</Badge>;
}

interface IncidentQueueProps {
  incidents: Incident[];
  selectedId?: string;
  onSelect: (id: string) => void;
  reduceMotion?: boolean | null;
}

export function IncidentQueue({ incidents, selectedId, onSelect, reduceMotion }: IncidentQueueProps) {
  return (
    <section aria-labelledby="incident-register-title" className="border-y border-border">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-1 py-4 sm:px-0">
        <h2 id="incident-register-title" className="min-w-0 text-lg font-semibold tracking-[-0.03em]">Incident register</h2>
        <span className="shrink-0 whitespace-nowrap text-xs text-stone-500">{incidents.length} declared records</span>
      </div>
      <div role="list" className="border-t border-border">
        {incidents.map((incident) => {
          const selected = incident.incident_id === selectedId;
          return (
            <button
              type="button"
              role="listitem"
              key={incident.incident_id}
              aria-pressed={selected}
              onClick={() => onSelect(incident.incident_id)}
              className={cn(
                "relative grid w-full grid-cols-[minmax(0,1fr)_auto] gap-x-4 border-b border-border px-1 py-4 text-left last:border-b-0 focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-stone-700 sm:grid-cols-[minmax(200px,1.6fr)_minmax(120px,0.8fr)_minmax(110px,0.6fr)_auto] sm:items-center sm:px-3",
                selected ? "text-stone-950" : "text-stone-700 hover:bg-stone-50"
              )}
            >
              {selected ? <motion.span layoutId="selected-incident" transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 480, damping: 36 }} className="absolute inset-x-0 inset-y-1 -z-10 rounded-md bg-[#efeee9]" aria-hidden="true" /> : null}
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold tracking-[-0.015em]">{incident.location}</span>
                <span className="mt-1 block truncate text-xs text-stone-500">{incident.corridor || "Corridor not supplied"}</span>
              </span>
              <span className="hidden text-sm sm:block">{incident.incident_class}</span>
              <span className="hidden text-xs text-stone-500 sm:block">{incident.status}</span>
              <span className="flex items-center gap-3"><span className="hidden text-xs text-stone-500 sm:inline">{incident.timestamp}</span><SeverityBadge severity={incident.severity} /></span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
