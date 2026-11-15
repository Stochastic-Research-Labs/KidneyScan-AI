"use client";

import { useRef, useState } from "react";
import { Columns2, Maximize, RotateCcw, RotateCw } from "lucide-react";
import type { PredictionResult } from "@/lib/api";
import { KidneyScene } from "./KidneyScene";

/** Two panels side by side — same layout as NeuroScan's ComparisonViewer.
 *  Left: original kidney, no overlay. Right: predicted, kidney grey,
 *  tumor yellow/red, cyst blue. Each panel has its own OrbitControls. */
export function ComparisonViewer({ result }: { result: PredictionResult }) {
  const [autoRotate, setAutoRotate] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);

  const fullscreen = (ref: React.RefObject<HTMLDivElement | null>) => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void ref.current?.requestFullscreen();
  };

  const panel = (
    key: string,
    title: string,
    sub: string,
    showOverlay: boolean,
    ref: React.RefObject<HTMLDivElement | null>
  ) => (
    <div ref={ref} className="min-w-0 rounded-xl border border-border bg-card/40 p-2">
      <div className="flex items-center justify-between px-2 pb-2 pt-2">
        <p className="text-xs font-medium">
          {title} <span className="ml-2 text-text-muted">{sub}</span>
        </p>
        <button
          type="button"
          onClick={() => fullscreen(ref)}
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text-muted hover:text-text"
          aria-label={`Fullscreen ${title}`}
        >
          <Maximize size={13} aria-hidden="true" /> Fullscreen
        </button>
      </div>
      <KidneyScene lesions={result.lesions} showOverlay={showOverlay} autoRotate={autoRotate} resetKey={resetKey} />
    </div>
  );

  return (
    <section className="min-w-0 space-y-4" aria-label="Kidney comparison viewer">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="flex items-center gap-2 text-xs text-text-muted">
          <Columns2 size={14} className="text-primary" aria-hidden="true" />
          Drag to rotate 360° · Scroll to zoom · Right-drag to move
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setAutoRotate((v) => !v)}
            aria-pressed={autoRotate}
            className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-border px-3 text-xs hover:text-text"
          >
            <RotateCw size={14} aria-hidden="true" /> {autoRotate ? "Stop rotate" : "Auto-rotate"}
          </button>
          <button
            type="button"
            onClick={() => setResetKey((k) => k + 1)}
            className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-border px-3 text-xs hover:text-text"
          >
            <RotateCcw size={14} aria-hidden="true" /> Reset view
          </button>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {panel("left", "Original kidney", "/ no overlay", false, leftRef)}
        {panel("right", result.is_dummy ? "Dummy prediction (TEST)" : "Predicted segmentation", result.is_dummy ? "/ test data only" : "/ model output", true, rightRef)}
      </div>

      <div className="flex flex-wrap gap-4 text-xs text-text-muted" aria-label="Legend">
        <span><i className="mr-1 inline-block h-3 w-3 rounded-full bg-[#9aa7b0] align-middle" /> Kidney (grey)</span>
        <span><i className="mr-1 inline-block h-3 w-3 rounded-full bg-[#facc15] align-middle" /> Tumor (yellow, red core)</span>
        <span><i className="mr-1 inline-block h-3 w-3 rounded-full bg-[#38bdf8] align-middle" /> Cyst (blue)</span>
      </div>
      <p className="text-xs leading-6 text-text-muted">
        {result.is_dummy
          ? "DUMMY / TEST data. These shapes are generated for UI testing only — not a real prediction. Switch the backend to real mode for real inference."
          : "The overlay is a model prediction, not a reference annotation. A normal case shows no tumor and no cyst here."}
      </p>
    </section>
  );
}
