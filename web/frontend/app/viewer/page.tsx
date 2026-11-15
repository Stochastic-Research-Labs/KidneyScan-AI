"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { loadResult, type PredictionResult } from "@/lib/api";

// Three.js needs the browser — never render the viewer on the server.
const ComparisonViewer = dynamic(
  () => import("@/components/viewer/ComparisonViewer").then((m) => m.ComparisonViewer),
  { ssr: false, loading: () => <p className="text-sm text-text-muted">Loading 3D viewer…</p> }
);

export default function ViewerPage() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  useEffect(() => setResult(loadResult()), []);

  return (
    <div className="page-shell space-y-6">
      <div>
        <p className="eyebrow">Viewer</p>
        <h1 className="mt-2 text-3xl font-bold">3D viewer</h1>
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
              DUMMY / TEST — this is not a real prediction. Case: {result.case_id}
            </p>
          )}
          <ComparisonViewer result={result} />
          <div className="flex flex-wrap gap-3">
            <Link href="/results" className="btn-primary">See volumes &amp; scores</Link>
            <Link href="/upload" className="btn-ghost">Upload another CT</Link>
          </div>
        </>
      )}
    </div>
  );
}
