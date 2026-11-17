"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import UploadPanel, { type Mode, type Stage } from "../../components/UploadPanel";
import ResultsPanel from "../../components/ResultsPanel";
import { checkHealth, predictScan, type BackendStatus } from "../../lib/api";
import { extentsDiffer, labelVolumesCm3, readNifti, toGrid, type AxisMap, type Grid } from "../../lib/nifti";
import { makeDummy } from "../../lib/phantom";
import type { ClassScores, ClassVolumes } from "../../lib/types";

const Viewer3D = dynamic(() => import("../../components/Viewer3D"), {
  ssr: false,
  loading: () => <div className="ks-card ks-empty">Loading the 3D viewer</div>,
});

const MAX_MB = 600;
type CtMeta = { dims: [number, number, number]; spacing: [number, number, number]; map: AxisMap };

const mm = (a: number[]) => a.map((x) => +x.toFixed(2)).join(" × ");

export default function Workspace() {
  const [mode, setMode] = useState<Mode>("real");
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [fileInfo, setFileInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [backend, setBackend] = useState<BackendStatus>("checking");
  const [ct, setCt] = useState<Grid | null>(null);
  const [mask, setMask] = useState<Grid | null>(null);
  const [volumes, setVolumes] = useState<ClassVolumes | null>(null);
  const [scores, setScores] = useState<ClassScores | null>(null);
  const meta = useRef<CtMeta | null>(null);
  const abort = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abort.current?.abort();
    setStage("idle");
    setProgress(0);
    setFile(null);
    setFileInfo(null);
    setError(null);
    setWarning(null);
    setCt(null);
    setMask(null);
    setVolumes(null);
    setScores(null);
    meta.current = null;
  }, []);

  const loadDummy = useCallback(() => {
    reset();
    const d = makeDummy();
    setCt(d.ct);
    setMask(d.mask);
    setVolumes(d.volumes);
    setStage("done");
  }, [reset]);

  const changeMode = useCallback(
    (m: Mode) => {
      reset();
      setMode(m);
    },
    [reset],
  );

  // Open straight into the dummy demo from the home page link.
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("mode") === "dummy") {
      setMode("dummy");
      loadDummy();
    }
    return () => abort.current?.abort();
  }, [loadDummy]);

  useEffect(() => {
    if (mode !== "real") return;
    const ctrl = new AbortController();
    setBackend("checking");
    checkHealth(ctrl.signal).then((s) => !ctrl.signal.aborted && setBackend(s));
    return () => ctrl.abort();
  }, [mode]);

  const pickFile = useCallback(
    async (f: File) => {
      reset();
      if (!/\.nii(\.gz)?$/i.test(f.name)) {
        setError("Use a 3D CT scan saved as .nii or .nii.gz. Single JPG, PNG or DICOM slices are not supported.");
        return;
      }
      if (f.size > MAX_MB * 1048576) {
        setError(`This file is larger than ${MAX_MB} MB. Use a smaller or cropped scan.`);
        return;
      }
      setFile(f);
      setStage("reading");
      await new Promise((r) => setTimeout(r, 30)); // let the status paint before heavy work
      try {
        const raw = readNifti(await f.arrayBuffer());
        meta.current = { dims: raw.dims, spacing: raw.spacing, map: raw.map };
        setCt(toGrid(raw, "ct"));
        setFileInfo(`${raw.dims.join(" × ")} voxels, ${mm(raw.spacing)} mm`);
        setStage("ready");
      } catch (e) {
        setFile(null);
        setStage("error");
        setError(e instanceof Error ? e.message : "Could not read this file.");
      }
    },
    [reset],
  );

  const run = useCallback(async () => {
    if (!file || !meta.current) return;
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(null);
    setWarning(null);
    setMask(null);
    setVolumes(null);
    setScores(null);
    setProgress(0);
    setStage("uploading");
    try {
      const res = await predictScan(file, {
        signal: ctrl.signal,
        onUploadProgress: setProgress,
        onUploaded: () => setStage("inferring"),
      });
      setStage("decoding");
      await new Promise((r) => setTimeout(r, 30));

      const m = meta.current!;
      const rawMask = readNifti(res.maskBuffer);
      const sameGrid = rawMask.dims.every((d, i) => d === m.dims[i]);
      if (sameGrid) {
        // Same voxel grid as the scan: reuse its orientation and voxel size.
        rawMask.spacing = m.spacing;
        rawMask.map = m.map;
      }
      const grid = toGrid(rawMask, "mask");
      if (!sameGrid) {
        setWarning(
          `The returned mask (${rawMask.dims.join(" × ")}) is on a different grid than your scan (${m.dims.join(" × ")}). ` +
            `Volumes use the mask's own voxel size, and the overlay may not line up exactly.`,
        );
      } else if (ct && extentsDiffer(ct, grid)) {
        setWarning("The mask and scan cover different physical sizes, so the overlay may not line up exactly.");
      }
      setMask(grid);
      setVolumes(labelVolumesCm3(rawMask));
      setScores(res.scores);
      setStage("done");
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setStage("ready");
        return;
      }
      setStage("error");
      setError(e instanceof Error ? e.message : "Segmentation failed.");
    }
  }, [file, ct]);

  const isDummy = mode === "dummy";

  return (
    <div className="ks-wrap ks-page">
      <header className="ks-page-head">
        <div>
          <h1>Your workspace</h1>
          <p className="ks-lede">Load a scan, run the model, and compare the original with the prediction.</p>
        </div>
      </header>

      <div className="ks-workgrid">
        <div className="ks-area-input">
          <UploadPanel
            mode={mode}
            onModeChange={changeMode}
            stage={stage}
            progress={progress}
            file={file}
            fileInfo={fileInfo}
            error={error}
            backend={backend}
            onPickFile={pickFile}
            onRun={run}
            onCancel={() => abort.current?.abort()}
            onLoadDummy={loadDummy}
          />
        </div>
        <div className="ks-area-viewer">
          <Viewer3D ct={ct} mask={mask} isDummy={isDummy} warning={warning} />
        </div>
        <div className="ks-area-results">
          <ResultsPanel volumes={volumes} scores={scores} isDummy={isDummy} />
          <p className="ks-small ks-disclaimer">
            Research prototype. Predictions need expert review. Best results so far: foreground Dice 0.4043 (kidney 0.787, tumor 0.378, cyst
            0.048).
          </p>
        </div>
      </div>
    </div>
  );
}
