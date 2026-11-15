// Single source of truth for segmentation class colors.
// Must stay in sync by value with backend/main.py CLASS_INFO and
// docs/API_CONTRACT.md — same "keep in sync" convention as NeuroScan.
export const CLASS_COLORS = {
  kidney: "#9aa7b0", // grey
  tumor: "#facc15", // yellow (viewer core uses #ef4444 red for the tumor centre)
  tumorCore: "#ef4444",
  cyst: "#38bdf8", // blue
} as const;

export type Lesion = {
  class: "tumor" | "cyst";
  center_xyz: [number, number, number]; // normalised 0..1, D/H/W order
  radius_mm: number;
  volume_cm3: number;
};

export type PredictionResult = {
  status: string;
  case_id: string;
  mode: "dummy" | "real";
  model: {
    name: string;
    checkpoint: string | null;
    base_channels: number;
    num_classes: number;
    classes: string[];
  };
  is_dummy: boolean;
  note: string;
  input: { file: string; shape: number[]; spacing_mm: number[] };
  shape: number[];
  spacing_mm: number[];
  finding: string;
  volumes_cm3: { kidney: number; tumor: number; cyst: number };
  voxel_counts: { kidney: number; tumor: number; cyst: number };
  lesions: Lesion[];
  findings: {
    tumor_present: boolean;
    cyst_present: boolean;
    cyst_count: number;
    tumor_volume_cm3: number;
    cyst_volume_cm3: number;
  };
  scores: {
    dice: Record<string, number | null>;
    surface_dice_mm1: Record<string, number | null>;
    hd95_mm: Record<string, number | null>;
    note?: string;
  } | null;
  mask: { encoding: string; file: string | null; shape: number[] };
  mask_grid: number;
  mask_base64: string;
  labels: Record<string, string>;
  inference_seconds: number;
  warning: string;
};

export type HealthStatus = {
  status: string;
  version: string;
  mode: string;
  model_loaded: boolean;
  checkpoint_present: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/backend";

export class ApiRequestError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiRequestError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers: { Accept: "application/json", ...(init?.headers ?? {}) } });
  } catch {
    throw new ApiRequestError("Can't reach the KidneyScan backend. Is it running on localhost:8000?", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch { /* ignore */ }
    throw new ApiRequestError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/api/health"),
  demoSample: (kind: "tumor" | "both" | "normal" = "tumor") =>
    request<PredictionResult>(`/api/demo/sample?kind=${kind}`),
  async predict(file: File, groundTruth: File | null, mode: "dummy" | "real"): Promise<PredictionResult> {
    const form = new FormData();
    form.append("file", file, file.name);
    if (groundTruth) form.append("ground_truth", groundTruth, groundTruth.name);
    form.append("mode", mode);
    return request<PredictionResult>("/api/predict", { method: "POST", body: form });
  },
};

// The current result lives in sessionStorage so Upload -> Viewer -> Results
// share it without a server-side case store (v1; NeuroScan has /api/cases).
const RESULT_KEY = "kidneyscan-result";
export function saveResult(r: PredictionResult) {
  sessionStorage.setItem(RESULT_KEY, JSON.stringify(r));
}
export function loadResult(): PredictionResult | null {
  try {
    const raw = sessionStorage.getItem(RESULT_KEY);
    return raw ? (JSON.parse(raw) as PredictionResult) : null;
  } catch {
    return null;
  }
}
