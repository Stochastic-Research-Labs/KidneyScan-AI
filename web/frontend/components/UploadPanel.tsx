"use client";

import { useRef, useState } from "react";
import type { BackendStatus } from "../lib/api";

export type Mode = "dummy" | "real";
export type Stage = "idle" | "reading" | "ready" | "uploading" | "inferring" | "decoding" | "done" | "error";

type Props = {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  stage: Stage;
  progress: number; // 0..1 while uploading
  file: File | null;
  fileInfo: string | null;
  error: string | null;
  backend: BackendStatus;
  onPickFile: (f: File) => void;
  onRun: () => void;
  onCancel: () => void;
  onLoadDummy: () => void;
};

const STAGE_TEXT: Partial<Record<Stage, string>> = {
  reading: "Reading the scan in your browser",
  inferring: "The model is segmenting your scan. The first request can take a minute while the server wakes up.",
  decoding: "Preparing the 3D view",
};

export default function UploadPanel(p: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const busy = ["reading", "uploading", "inferring", "decoding"].includes(p.stage);
  const canRun = p.mode === "real" && !!p.file && (p.stage === "ready" || p.stage === "done" || p.stage === "error");

  const take = (f?: File | null) => {
    if (f) p.onPickFile(f);
  };

  return (
    <section className="ks-card ks-input" aria-labelledby="input-title">
      <div className="ks-tabs" role="tablist" aria-label="Analysis mode">
        {(["real", "dummy"] as const).map((m) => (
          <button
            key={m}
            role="tab"
            type="button"
            aria-selected={p.mode === m}
            className="ks-tab"
            disabled={busy}
            onClick={() => p.onModeChange(m)}
          >
            {m === "real" ? "Real model" : "Dummy demo"}
          </button>
        ))}
      </div>

      {p.mode === "real" ? (
        <>
          <h2 id="input-title">Bring in a CT scan</h2>
          <p className="ks-muted">One 3D CT volume as .nii.gz. The scan is read in your browser, then sent to the model server.</p>

          <div
            className={`ks-drop${drag ? " is-drag" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              if (!busy) take(e.dataTransfer.files?.[0]);
            }}
          >
            <input
              ref={input}
              type="file"
              accept=".nii,.gz,.nii.gz"
              hidden
              onChange={(e) => {
                take(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
            {p.file ? (
              <div className="ks-file">
                <strong title={p.file.name}>{p.file.name}</strong>
                <span>{(p.file.size / 1048576).toFixed(1)} MB{p.fileInfo ? `. ${p.fileInfo}` : ""}</span>
              </div>
            ) : (
              <div className="ks-file">
                <strong>Drop a .nii.gz file here</strong>
                <span>or choose one from your device</span>
              </div>
            )}
            <button type="button" className="ks-btn ks-btn-ghost" disabled={busy} onClick={() => input.current?.click()}>
              {p.file ? "Choose another file" : "Browse files"}
            </button>
          </div>

          <div className="ks-actions">
            <button type="button" className="ks-btn ks-btn-primary" disabled={!canRun} onClick={p.onRun}>
              {p.stage === "done" ? "Run segmentation again" : "Run segmentation"}
            </button>
            {(p.stage === "uploading" || p.stage === "inferring") && (
              <button type="button" className="ks-btn ks-btn-ghost" onClick={p.onCancel}>
                Cancel
              </button>
            )}
          </div>

          <p className={`ks-backend ks-backend-${p.backend}`}>
            <i aria-hidden />
            {p.backend === "checking" && "Checking the model server"}
            {p.backend === "online" && "Model server online"}
            {p.backend === "offline" && "Model server not reachable. It may be waking up. Try again shortly."}
          </p>
        </>
      ) : (
        <>
          <h2 id="input-title">Try the viewer without a scan</h2>
          <p className="ks-muted">
            Loads a synthetic torso with two kidneys, one sphere labelled tumor and one labelled cyst. It tests the 3D viewer only.
            No model runs and nothing is sent anywhere.
          </p>
          <div className="ks-actions">
            <button type="button" className="ks-btn ks-btn-primary" onClick={p.onLoadDummy}>
              Load dummy demo
            </button>
          </div>
        </>
      )}

      <div aria-live="polite">
        {p.stage === "uploading" && (
          <div className="ks-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p.progress * 100)}>
            <div style={{ width: `${Math.max(3, p.progress * 100)}%` }} />
            <span>Uploading {Math.round(p.progress * 100)}%</span>
          </div>
        )}
        {STAGE_TEXT[p.stage] && (
          <p className="ks-status">
            <span className="ks-spinner" aria-hidden />
            {STAGE_TEXT[p.stage]}
          </p>
        )}
        {p.error && (
          <p className="ks-note ks-note-error" role="alert">
            {p.error}
          </p>
        )}
      </div>
    </section>
  );
}
