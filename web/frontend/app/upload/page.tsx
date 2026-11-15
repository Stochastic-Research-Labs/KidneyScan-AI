"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Upload } from "lucide-react";
import { ApiRequestError, api, saveResult } from "@/lib/api";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [groundTruth, setGroundTruth] = useState<File | null>(null);
  const [mode, setMode] = useState<"dummy" | "real">("dummy");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const runPredict = async (f: File, gt: File | null, m: "dummy" | "real") => {
    setBusy(true);
    setError("");
    try {
      const result = await api.predict(f, gt, m);
      saveResult(result);
      router.push("/viewer");
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.message : "Upload failed. Is the backend running on localhost:8000?");
    } finally {
      setBusy(false);
    }
  };

  const loadDemo = async (kind: "tumor" | "both" | "normal") => {
    setBusy(true);
    setError("");
    try {
      const result = await api.demoSample(kind);
      saveResult(result);
      router.push("/viewer");
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.message : "Demo failed. Is the backend running on localhost:8000?");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-shell max-w-3xl space-y-6">
      <div>
        <p className="eyebrow">Upload</p>
        <h1 className="mt-2 text-3xl font-bold">Upload a kidney CT</h1>
        <p className="mt-2 text-sm leading-6 text-text-muted">
          One CT file (.nii.gz). The file is only the input — the model decides
          what it finds. A normal scan shows no tumor and no cyst.
        </p>
      </div>

      <div className="glass-panel space-y-4 p-5">
        <label className="block text-sm font-medium">
          CT file (.nii.gz) *
          <input
            type="file"
            accept=".nii.gz,.gz"
            className="mt-2 block w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="block text-sm font-medium">
          Ground-truth mask (.nii.gz) — optional, only for Dice / HD95 scores
          <input
            type="file"
            accept=".nii.gz,.gz"
            className="mt-2 block w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm"
            onChange={(e) => setGroundTruth(e.target.files?.[0] ?? null)}
          />
        </label>
        <fieldset>
          <legend className="text-sm font-medium">Mode</legend>
          <div className="mt-2 flex gap-4 text-sm">
            <label className="flex items-center gap-2">
              <input type="radio" name="mode" checked={mode === "dummy"} onChange={() => setMode("dummy")} />
              Dummy — UI testing only
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" name="mode" checked={mode === "real"} onChange={() => setMode("real")} />
              Real model
            </label>
          </div>
          {mode === "dummy" && (
            <p className="mt-2 text-xs leading-5 text-yellow-200">
              Dummy mode generates a clearly labelled TEST mask so the viewer can
              be tested without a trained model. It is not a real prediction.
            </p>
          )}
          {mode === "real" && (
            <p className="mt-2 text-xs leading-5 text-text-muted">
              Real mode needs the trained model file in the backend
              (models_store/model.pt). If it is missing, the backend returns an error — it never falls back to dummy.
            </p>
          )}
        </fieldset>
        {error && <p role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>}
        <button type="button" className="btn-primary" disabled={!file || busy} onClick={() => file && runPredict(file, groundTruth, mode)}>
          {busy ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : <Upload size={16} aria-hidden="true" />}
          {busy ? "Running…" : "Run prediction"}
        </button>
      </div>

      <div className="glass-panel space-y-3 p-5">
        <h2 className="font-semibold">No CT file? Try the dummy demo</h2>
        <p className="text-sm leading-6 text-text-muted">
          These samples need no upload. They are clearly labelled DUMMY / TEST.
        </p>
        <div className="flex flex-wrap gap-3">
          <button type="button" className="btn-ghost" disabled={busy} onClick={() => loadDemo("tumor")}>Dummy: tumor</button>
          <button type="button" className="btn-ghost" disabled={busy} onClick={() => loadDemo("both")}>Dummy: tumor + cyst</button>
          <button type="button" className="btn-ghost" disabled={busy} onClick={() => loadDemo("normal")}>Dummy: normal (no highlight)</button>
        </div>
      </div>
    </div>
  );
}
