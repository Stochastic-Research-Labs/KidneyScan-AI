"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { loadResult, type PredictionResult } from "@/lib/api";

function ScoreTable({ result }: { result: PredictionResult }) {
  if (!result.scores) {
    return <p className="text-sm leading-6 text-text-muted">No scores — upload a ground-truth mask (.nii.gz) with the CT to get Dice, Surface Dice, and HD95 per class.</p>;
  }
  const rows = ["kidney", "tumor", "cyst"];
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-text-muted"><th className="py-2">Class</th><th>Dice</th><th>Surface Dice (1 mm)</th><th>HD95 (mm)</th></tr>
      </thead>
      <tbody>
        {rows.map((c) => (
          <tr key={c} className="border-t border-border">
            <td className="py-2 font-medium capitalize">{c}</td>
            <td>{result.scores!.dice[c] ?? "—"}</td>
            <td>{result.scores!.surface_dice_mm1[c] ?? "—"}</td>
            <td>{result.scores!.hd95_mm[c] ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ResultsPage() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  useEffect(() => setResult(loadResult()), []);

  return (
    <div className="page-shell max-w-4xl space-y-6">
      <div>
        <p className="eyebrow">Results</p>
        <h1 className="mt-2 text-3xl font-bold">Volumes &amp; scores</h1>
      </div>
      {!result ? (
        <div className="glass-panel p-5 text-sm leading-7 text-text-muted">
          <p>No result yet. Upload a CT first, or load a dummy demo sample.</p>
          <Link href="/upload" className="btn-primary mt-4">Go to Upload</Link>
        </div>
      ) : (
        <>
          {result.is_dummy && (
            <p role="status" className="rounded-lg border border-yellow-500/40 bg-yellow-500/10 px-4 py-3 text-sm font-semibold text-yellow-200">
              DUMMY / TEST — not a real prediction. Case: {result.case_id}
            </p>
          )}
          <div className="glass-panel p-5">
            <h2 className="font-semibold">Volumes (cm³)</h2>
            <table className="mt-3 w-full text-sm">
              <thead><tr className="text-left text-text-muted"><th className="py-2">Class</th><th>Volume (cm³)</th><th>Voxels</th></tr></thead>
              <tbody>
                {(["kidney", "tumor", "cyst"] as const).map((c) => (
                  <tr key={c} className="border-t border-border">
                    <td className="py-2 font-medium capitalize">{c}</td>
                    <td>{result.volumes_cm3[c]}</td>
                    <td>{result.voxel_counts[c].toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-sm text-text-muted">
              Tumor: {result.findings.tumor_present ? `found, ${result.findings.tumor_volume_cm3} cm³` : "not found"} ·
              Cyst: {result.findings.cyst_present ? `found, ${result.findings.cyst_count} cyst(s), ${result.findings.cyst_volume_cm3} cm³` : "not found"}
            </p>
          </div>

          <div className="glass-panel p-5">
            <h2 className="font-semibold">Lesions found by the model</h2>
            {result.lesions.length === 0 ? (
              <p className="mt-2 text-sm text-text-muted">None. A normal case shows no highlight — the model found no tumor and no cyst.</p>
            ) : (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-text-muted">
                {result.lesions.map((l, i) => (
                  <li key={i}>{l.class} — {l.volume_cm3} cm³, radius {l.radius_mm} mm, centre (normalised) {l.center_xyz.join(", ")}</li>
                ))}
              </ul>
            )}
          </div>

          <div className="glass-panel p-5">
            <h2 className="font-semibold">Scores vs ground truth</h2>
            <div className="mt-3"><ScoreTable result={result} /></div>
          </div>

          <div className="glass-panel p-5 text-sm leading-7 text-text-muted">
            <h2 className="font-semibold text-text">Details</h2>
            <p>Case: {result.case_id} · Mode: {result.mode} · Model: {result.model.name}</p>
            <p>Shape: {result.shape.join(" × ")} · Spacing: {result.spacing_mm.join(" × ")} mm</p>
            <p>{result.note}</p>
            <p className="font-medium text-yellow-200">{result.warning}</p>
            <div className="mt-4 flex flex-wrap gap-3">
              <a
                className="btn-ghost"
                href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`}
                download={`${result.case_id}-kidneyscan-result.json`}
              >
                Download result (JSON)
              </a>
              <Link href="/viewer" className="btn-primary">Back to 3D viewer</Link>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
