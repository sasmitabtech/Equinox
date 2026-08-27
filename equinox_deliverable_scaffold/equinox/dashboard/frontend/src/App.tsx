import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from "motion/react";
import {
  ArrowClockwiseIcon,
  CheckCircleIcon,
  FilmStripIcon,
  SirenIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { MagicCard } from "@/components/ui/magic-card";
import Loader from "@/components/kokonutui/loader";
import DetectorEvidenceCarousel from "@/components/kokonutui/carousel-cards";
import { IncidentQueue, SeverityBadge } from "@/components/incident-queue";
import { api, API_BASE } from "@/lib/api";
import type { Health, Incident, IncidentStatus } from "@/lib/types";

const workflow: IncidentStatus[] = ["Open", "Assigned", "In Action", "Cleared", "Verified"];
const detectorSamples = [
  { src: "/detector-evidence/people-scene.webp", alt: "VisDrone frame with YOLO person detections", label: "Crowded aerial scene", detail: "Frame 005 · 41 predictions, including people and traffic classes." },
  { src: "/detector-evidence/vehicle-scene.webp", alt: "VisDrone frame with YOLO car, bus, and truck detections", label: "Vehicle scene", detail: "Frame 000 · 7 predictions, including car, bus, and truck classes." },
  { src: "/detector-evidence/edge-cow.webp", alt: "VisDrone frame with YOLO cow edge-case prediction", label: "False-positive edge case", detail: "Frame 009 · includes a cow prediction; shown to make the generic-model limitation visible." },
];

function accessibilityLabel(score: number) {
  if (score >= 80) return "Passable corridor";
  if (score >= 55) return "Restricted passage";
  return "Major delay likely";
}

function scoreClass(score: number) {
  if (score >= 80) return "text-[#346538]";
  if (score >= 55) return "text-[#956400]";
  return "text-[#9f2f2d]";
}

function Evidence({ incident, reduceMotion }: { incident: Incident; reduceMotion: boolean | null }) {
  const source = api.mediaUrl(incident.evidence_url);
  const poster = incident.evidence_url ? `/evidence-posters/${incident.evidence_url.split("/").pop()?.replace(/\.mp4$/i, ".jpg")}` : undefined;
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    // Some headless and low-power browsers defer video decode. Keep the
    // poster and controls usable rather than masking the finite evidence.
    const fallback = window.setTimeout(() => setState((current) => current === "loading" ? "ready" : current), 2_500);
    return () => window.clearTimeout(fallback);
  }, [incident.evidence_url]);

  return (
    <motion.figure
      key={incident.incident_id}
      initial={reduceMotion ? false : { opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 360, damping: 32 }}
      className="m-0"
    >
      <div className="mb-3 flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2 text-sm">
        <figcaption className="flex min-w-0 flex-wrap items-center gap-2 font-medium"><FilmStripIcon className="shrink-0" size={18} weight="fill" aria-hidden="true" /> Evidence fixture <Badge variant="outline" className="shrink-0 border-[#d9b7b1] bg-[#fdebec] text-[10px] font-semibold uppercase tracking-[0.08em] text-[#9f2f2d]">Simulated</Badge></figcaption>
        <span className="max-w-full break-words text-xs text-stone-500">{incident.timestamp} · {incident.incident_id}</span>
      </div>
      <div className="relative aspect-video min-w-0 max-w-full overflow-hidden rounded-md border border-stone-300 bg-stone-200">
        {source ? <video
          key={`${incident.incident_id}-${source}`}
          controls
          muted
          autoPlay
          loop
          playsInline
          preload="metadata"
          src={source}
          poster={poster}
          onLoadedMetadata={() => setState("ready")}
          onCanPlay={() => setState("ready")}
          onLoadedData={() => setState("ready")}
          onError={() => setState("error")}
          className="absolute inset-0 h-full w-full bg-stone-900 object-contain"
        ><track kind="captions" /></video> : null}
        {state === "loading" ? <div className="absolute inset-0 grid place-items-center bg-stone-100/80"><Loader title="Loading evidence" subtitle="Requesting the local MP4 fixture" size="sm" /></div> : null}
        {state === "error" ? <div className="absolute inset-0 grid place-items-center p-6 text-center"><div><WarningCircleIcon className="mx-auto text-stone-500" size={26} weight="fill" aria-hidden="true" /><p className="mt-2 text-sm font-medium">Evidence video is unavailable</p><p className="mt-1 text-xs text-stone-600">Check the FastAPI service and local fixture path.</p></div></div> : null}
      </div>
      <p className="mt-3 text-xs leading-5 text-stone-600">Generated demonstration footage. It is not a live drone feed and contains no model-generated overlays.</p>
    </motion.figure>
  );
}

function StatusTimeline({ status, reduceMotion }: { status: IncidentStatus; reduceMotion: boolean | null }) {
  const currentIndex = workflow.indexOf(status);
  return (
    <ol className="mt-5 flex flex-wrap gap-x-5 gap-y-3" aria-label="Incident response progression">
      {workflow.map((step, index) => {
        const reached = index <= currentIndex;
        return <motion.li key={step} layout transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 460, damping: 32 }} className={reached ? "flex items-center gap-2 text-sm font-medium text-stone-900" : "flex items-center gap-2 text-sm text-stone-500"}>
          <span className={reached ? "grid size-5 place-items-center rounded-full bg-stone-900 text-[10px] text-white" : "grid size-5 place-items-center rounded-full border border-stone-300 text-[10px]"}>{reached ? <CheckCircleIcon size={13} weight="fill" aria-hidden="true" /> : index + 1}</span>{step}
        </motion.li>;
      })}
    </ol>
  );
}

function App() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedId, setSelectedId] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string>();
  const [error, setError] = useState<string>();
  const [pending, setPending] = useState<string>();
  const [detectorSample, setDetectorSample] = useState(0);
  const reduceMotion = useReducedMotion();

  const load = useCallback(async () => {
    try {
      const [nextIncidents, nextHealth] = await Promise.all([api.getIncidents(), api.getHealth()]);
      setIncidents(nextIncidents);
      setHealth(nextHealth);
      setSelectedId((current) => nextIncidents.some((item) => item.incident_id === current) ? current : nextIncidents[0]?.incident_id);
      setError(undefined);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to contact the Equinox API.");
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const interval = window.setInterval(() => void load(), 20_000);
    return () => window.clearInterval(interval);
  }, [load]);

  const selected = useMemo(() => incidents.find((item) => item.incident_id === selectedId), [incidents, selectedId]);
  const activeCount = incidents.filter((item) => !["Cleared", "Verified"].includes(item.status)).length;

  const saveIncident = (next: Incident) => setIncidents((current) => current.map((item) => item.incident_id === next.incident_id ? next : item));
  const updateStatus = async (status: IncidentStatus) => {
    if (!selected) return;
    setPending(status); setNotice(undefined);
    try { const next = await api.updateStatus(selected.incident_id, status); saveIncident(next); setNotice(`${selected.incident_id} moved to ${status}.`); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Status update failed."); }
    finally { setPending(undefined); }
  };
  const recheck = async () => {
    if (!selected) return;
    setPending("recheck"); setNotice(undefined);
    try { const next = await api.recheck(selected.incident_id); saveIncident(next); setNotice(`${selected.incident_id} re-check completed. The current prototype endpoint applies a simulated verification result.`); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Clearance re-check failed."); }
    finally { setPending(undefined); }
  };

  return (
    <main className="min-h-screen overflow-x-hidden bg-background text-foreground">
      <header className="border-b border-border bg-[#fbfaf7] px-4 py-4 sm:px-6 lg:px-8">
        <div className="mx-auto flex w-full min-w-0 max-w-screen-2xl flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3"><span className="grid size-9 shrink-0 place-items-center rounded-md bg-stone-900 text-white"><SirenIcon size={19} weight="fill" aria-hidden="true" /></span><div className="min-w-0"><h1 className="text-lg font-semibold tracking-[-0.035em]">Equinox</h1><p className="text-xs text-stone-500">Emergency corridor operations</p></div></div>
          <div className="flex min-w-0 flex-wrap items-center gap-x-5 gap-y-2 text-xs text-stone-600"><span className="flex items-center gap-2 whitespace-nowrap"><i className={health?.status === "ok" ? "size-2 shrink-0 rounded-full bg-[#346538]" : "size-2 shrink-0 rounded-full bg-[#9f2f2d]"} />{health?.status === "ok" ? "API connected" : "API unavailable"}</span><span className="whitespace-nowrap"><strong className="font-medium text-stone-900">{activeCount}</strong> active responses</span><Tooltip><TooltipTrigger className="shrink-0 cursor-help border-b border-dotted border-stone-400">Prototype data</TooltipTrigger><TooltipContent>Incident intelligence and clearance outcomes are seeded for this demo.</TooltipContent></Tooltip></div>
        </div>
      </header>
      <div className="border-b border-[#d8c8b5] bg-[#fff6e9] px-4 py-2.5 text-sm text-[#6f4c16] sm:px-6 lg:px-8"><div className="mx-auto flex w-full min-w-0 max-w-screen-2xl items-start gap-2"><WarningCircleIcon className="mt-0.5 shrink-0" size={18} weight="fill" aria-hidden="true" /><span className="min-w-0 break-words"><strong>Demo disclosure.</strong> Incident records, recommendations, and clearance results are seeded prototype data. The separate YOLO test is not connected to this dashboard.</span></div></div>

      <LayoutGroup>
        <div className="mx-auto w-full min-w-0 max-w-screen-2xl px-4 py-6 sm:px-6 lg:px-8">
          {error ? <Alert variant="destructive" className="mb-5 bg-[#fdebec]"><WarningCircleIcon size={18} weight="fill" /><AlertTitle>Connection problem</AlertTitle><AlertDescription>{error} <button className="ml-1 underline underline-offset-2" onClick={() => void load()}>Try again</button></AlertDescription></Alert> : null}
          <AnimatePresence>{notice ? <motion.div initial={reduceMotion ? false : { opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><Alert className="mb-5 border-[#cbdcc8] bg-[#edf3ec] text-[#346538]"><CheckCircleIcon size={18} weight="fill" /><AlertTitle>Action recorded</AlertTitle><AlertDescription>{notice}</AlertDescription></Alert></motion.div> : null}</AnimatePresence>
          {loading ? <div className="space-y-3"><Skeleton className="h-8 w-44" /><Skeleton className="h-48 w-full" /></div> : <IncidentQueue incidents={incidents} selectedId={selectedId} onSelect={setSelectedId} reduceMotion={reduceMotion} />}

          <AnimatePresence mode="wait">
            {selected ? <motion.section key={selected.incident_id} initial={reduceMotion ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={reduceMotion ? {} : { opacity: 0, y: -5 }} transition={{ type: "spring", stiffness: 360, damping: 32 }} className="mt-8">
              <div className="grid min-w-0 gap-8 xl:grid-cols-[minmax(0,8fr)_minmax(280px,4fr)] xl:items-start">
                <div className="min-w-0"><Evidence incident={selected} reduceMotion={reduceMotion} /></div>
                <MagicCard disabled={Boolean(reduceMotion)} className="min-w-0 border-t border-stone-300 p-5 xl:border-t xl:border-l xl:pl-8 xl:pt-5">
                  <div className="flex items-start justify-between gap-3"><div><p className="text-xs text-stone-500">{selected.incident_id} · {selected.corridor || selected.location}</p><h2 className="mt-2 font-serif text-3xl font-semibold leading-none tracking-[-0.04em] text-stone-900">{selected.location}</h2><p className="mt-3 text-sm text-stone-600">Declared class: <span className="font-medium text-stone-900">{selected.incident_class}</span></p></div><SeverityBadge severity={selected.severity} /></div>
                  <Separator className="my-5" />
                  <div className="flex items-end justify-between"><div><p className="text-xs text-stone-500">Emergency accessibility</p><p className={"mt-1 text-5xl font-semibold tracking-[-0.06em] " + scoreClass(selected.accessibility_score)}>{selected.accessibility_score}<span className="ml-1 text-base font-normal text-stone-500">/100</span></p></div><p className="max-w-28 text-right text-sm text-stone-600">{accessibilityLabel(selected.accessibility_score)}</p></div>
                  <p className="mt-6 border-t border-stone-200 pt-4 text-sm leading-6 text-stone-800">{selected.recommended_action}</p>
                  <Button className="mt-5 w-full bg-stone-900 text-white hover:bg-stone-700" disabled={pending === "recheck"} onClick={() => void recheck()}>{pending === "recheck" ? <ArrowClockwiseIcon className="animate-spin" size={16} aria-hidden="true" /> : <ArrowClockwiseIcon size={16} weight="bold" aria-hidden="true" />} Verify clearance <span className="ml-auto text-xs text-stone-300">Simulated check</span></Button>
                </MagicCard>
              </div>
              <section className="mt-8 border-t border-border pt-6">
                <h2 className="text-lg font-semibold tracking-[-0.03em]">Response movement</h2>
                <StatusTimeline status={selected.status} reduceMotion={reduceMotion} />
                <div className="mt-5 flex flex-wrap gap-2"><Button variant="outline" size="sm" disabled={pending === "Assigned" || selected.status === "Assigned"} onClick={() => void updateStatus("Assigned")}>{pending === "Assigned" ? <ArrowClockwiseIcon className="animate-spin" /> : null} Assign response</Button><Button variant="outline" size="sm" disabled={pending === "In Action" || selected.status === "In Action"} onClick={() => void updateStatus("In Action")}>{pending === "In Action" ? <ArrowClockwiseIcon className="animate-spin" /> : null} Begin intervention</Button><Button variant="outline" size="sm" disabled={pending === "Cleared" || ["Cleared", "Verified"].includes(selected.status)} onClick={() => void updateStatus("Cleared")}>{pending === "Cleared" ? <ArrowClockwiseIcon className="animate-spin" /> : null} Mark clear</Button></div>
              </section>
              <section className="mt-8 border-t border-border pt-6">
                <Tabs defaultValue="factors"><TabsList className="h-auto flex-wrap rounded-md bg-stone-100 p-1"><TabsTrigger value="factors" className="rounded-sm px-3 py-1.5 text-xs">Contributing factors</TabsTrigger><TabsTrigger value="provenance" className="rounded-sm px-3 py-1.5 text-xs">Evidence & provenance</TabsTrigger><TabsTrigger value="detector" className="rounded-sm px-3 py-1.5 text-xs">Detector evidence</TabsTrigger></TabsList>
                  <TabsContent value="factors" className="mt-5"><div className="grid gap-x-8 gap-y-3 md:grid-cols-2">{selected.contributing_factors.map((factor) => <p key={factor} className="border-b border-border pb-3 text-sm leading-5 text-stone-700">{factor}</p>)}</div></TabsContent>
                  <TabsContent value="provenance" className="mt-5"><dl className="grid gap-x-8 gap-y-4 text-sm md:grid-cols-2"><div><dt className="text-stone-500">Video source</dt><dd className="mt-1 text-stone-800">Generated MP4 fixture</dd></div><div><dt className="text-stone-500">Coordinates</dt><dd className="mt-1 text-stone-800">{selected.lat.toFixed(3)}, {selected.lon.toFixed(3)}</dd></div><div><dt className="text-stone-500">API base</dt><dd className="mt-1 break-all text-stone-800">{API_BASE || "same origin"}</dd></div><div><dt className="text-stone-500">Dashboard detector connection</dt><dd className="mt-1 text-stone-800">Not implemented</dd></div></dl></TabsContent>
                  <TabsContent value="detector" className="mt-5"><div className="grid min-w-0 gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(220px,3fr)]"><div className="min-w-0"><div className="flex flex-wrap items-baseline gap-x-5 gap-y-2 border-b border-border pb-4"><p className="text-4xl font-semibold tracking-[-0.05em] text-stone-900">612</p><p className="text-sm text-stone-600">predictions from 50 VisDrone2019-DET validation images</p><p className="text-sm text-stone-600">11.057 s wall time · approximately 221 ms/image</p></div><div className="mt-4 overflow-hidden rounded-md border border-stone-300 bg-stone-100"><AnimatePresence mode="wait"><motion.img key={detectorSamples[detectorSample].src} initial={reduceMotion ? false : { opacity: 0, scale: 0.99 }} animate={{ opacity: 1, scale: 1 }} exit={reduceMotion ? {} : { opacity: 0 }} transition={{ type: "spring", stiffness: 360, damping: 32 }} src={detectorSamples[detectorSample].src} alt={detectorSamples[detectorSample].alt} className="block h-auto w-full" /></AnimatePresence></div><p className="mt-3 text-sm text-stone-700"><strong>{detectorSamples[detectorSample].label}.</strong> {detectorSamples[detectorSample].detail}</p><DetectorEvidenceCarousel items={detectorSamples} selectedIndex={detectorSample} onSelect={setDetectorSample} /></div><aside className="min-w-0 border-t border-border pt-5 lg:border-t-0 lg:border-l lg:pl-6 lg:pt-0"><h3 className="text-base font-semibold tracking-[-0.02em]">Smoke-test provenance</h3><dl className="mt-4 space-y-4 text-sm"><div><dt className="text-stone-500">Model</dt><dd className="mt-1 text-stone-800">YOLO11n, generic COCO-pretrained weights</dd></div><div><dt className="text-stone-500">Source</dt><dd className="mt-1 text-stone-800">VisDrone2019-DET validation subset</dd></div><div><dt className="text-stone-500">Evaluation boundary</dt><dd className="mt-1 text-stone-800">No fine-tune; no precision, recall, or mAP claim.</dd></div></dl><p className="mt-6 border-t border-[#d8c8b5] pt-4 text-sm leading-6 text-[#6f4c16]"><strong>Important:</strong> This is an object-detection smoke test, not blockage or severity validation. The detector is not wired into the seeded incident operations shown above.</p></aside></div></TabsContent>
                </Tabs>
              </section>
            </motion.section> : null}
          </AnimatePresence>
        </div>
      </LayoutGroup>
    </main>
  );
}

export default App;
