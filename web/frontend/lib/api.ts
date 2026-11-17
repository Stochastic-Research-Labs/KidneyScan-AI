import type { ClassScores, PredictResult } from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE ?? "https://m-wajid-t-v--kidneyscan-ai-serve.us-east.modal.run"
).replace(/\/$/, "");

export type BackendStatus = "checking" | "online" | "offline";

export async function checkHealth(signal?: AbortSignal): Promise<BackendStatus> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000); // a cold start can be slow
  signal?.addEventListener("abort", () => ctrl.abort());
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: ctrl.signal, cache: "no-store" });
    return res.ok ? "online" : "offline";
  } catch {
    return "offline";
  } finally {
    clearTimeout(timer);
  }
}

type Progress = {
  signal?: AbortSignal;
  /** 0..1 while the file is uploading. */
  onUploadProgress?: (fraction: number) => void;
  /** Called once the upload finished and the server is working. */
  onUploaded?: () => void;
};

/** POST /api/predict with form field `file`. Resolves with the mask and scores. */
export function predictScan(file: File, opts: Progress = {}): Promise<PredictResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/predict`);
    xhr.responseType = "arraybuffer";
    xhr.timeout = 15 * 60 * 1000;

    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) opts.onUploadProgress?.(ev.loaded / ev.total);
    };
    xhr.upload.onload = () => opts.onUploaded?.();
    xhr.onerror = () =>
      reject(new Error("Could not reach the model server. Check your connection, or try again in a minute."));
    xhr.ontimeout = () => reject(new Error("The model server took too long to answer. Try again."));
    xhr.onabort = () => reject(new DOMException("Cancelled", "AbortError"));
    opts.signal?.addEventListener("abort", () => xhr.abort());

    xhr.onload = async () => {
      const body = xhr.response as ArrayBuffer;
      const type = xhr.getResponseHeader("content-type") ?? "";
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(describeError(xhr.status, body)));
        return;
      }
      try {
        resolve(await parseResponse(body, type, (h) => xhr.getResponseHeader(h)));
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    };

    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

function describeError(status: number, body: ArrayBuffer) {
  let detail = "";
  try {
    const text = new TextDecoder().decode(body);
    try {
      const j = JSON.parse(text);
      detail = typeof j.detail === "string" ? j.detail : typeof j.error === "string" ? j.error : text.slice(0, 200);
    } catch {
      detail = text.slice(0, 200);
    }
  } catch {
    /* ignore */
  }
  return `The model server returned ${status}${detail ? `: ${detail}` : "."}`;
}

// ---------------------------------------------------------------------------
// Response handling. The server may answer with JSON (mask as URL or base64)
// or with the raw .nii.gz. Everything that depends on the exact shape lives
// here, so if API_CONTRACT.md differs, this is the only place to adjust.
// ---------------------------------------------------------------------------

const MASK_URL_KEYS = ["mask_url", "maskUrl", "segmentation_url", "mask_download_url", "download_url", "url"];
// NOTE: "mask_base64" is deliberately NOT here. In API_CONTRACT.md it is raw
// 32x32x32 voxel bytes (a viewer preview), not a .nii.gz file, so feeding it
// to readNifti would fail. The full mask always comes from mask.file above.
const MASK_B64_KEYS = ["mask_b64", "maskBase64", "segmentation_b64", "mask", "segmentation"];

async function parseResponse(
  body: ArrayBuffer,
  contentType: string,
  header: (name: string) => string | null,
): Promise<PredictResult> {
  if (contentType.includes("json")) {
    const json = JSON.parse(new TextDecoder().decode(body)) as Record<string, unknown>;
    return { maskBuffer: await resolveMask(json), scores: readScores(json) };
  }
  // Raw mask file. Scores may arrive in headers.
  let scores: ClassScores = { kidney: null, tumor: null, cyst: null };
  const packed = header("x-scores");
  if (packed) {
    try {
      scores = readScores(JSON.parse(packed));
    } catch {
      /* ignore */
    }
  } else {
    scores = readScores({
      kidney: header("x-kidney-score"),
      tumor: header("x-tumor-score"),
      cyst: header("x-cyst-score"),
    });
  }
  return { maskBuffer: body, scores };
}

async function resolveMask(json: Record<string, unknown>): Promise<ArrayBuffer> {
  // API_CONTRACT.md shape first: mask.file is the full-resolution .nii.gz
  // download URL (relative to the API base). This is the only full mask.
  const maskObj = json.mask as Record<string, unknown> | undefined;
  if (maskObj && typeof maskObj.file === "string" && maskObj.file) {
    const url = /^https?:/i.test(maskObj.file)
      ? maskObj.file
      : `${API_BASE}${maskObj.file.startsWith("/") ? "" : "/"}${maskObj.file}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download the mask (${res.status}).`);
    return res.arrayBuffer();
  }
  for (const k of MASK_URL_KEYS) {
    const v = json[k];
    if (typeof v === "string" && v) {
      const url = /^https?:/i.test(v) ? v : `${API_BASE}${v.startsWith("/") ? "" : "/"}${v}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Could not download the mask (${res.status}).`);
      return res.arrayBuffer();
    }
  }
  for (const k of MASK_B64_KEYS) {
    const v = json[k];
    if (typeof v === "string" && v.length > 100) return base64ToBuffer(v);
  }
  throw new Error(
    `The server answered, but the response has no mask file I can read (fields: ${Object.keys(json).join(", ") || "none"}). ` +
      `Note: mask_base64 in API_CONTRACT.md is raw 32x32x32 voxels, not a .nii.gz file, so it cannot be used here.`,
  );
}

function base64ToBuffer(b64: string): ArrayBuffer {
  const clean = b64.replace(/^data:[^,]*,/, "");
  const bin = atob(clean);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

function num(v: unknown): number | null {
  const n = typeof v === "string" ? parseFloat(v) : (v as number);
  return typeof n === "number" && Number.isFinite(n) ? n : null;
}

function readScores(src: unknown): ClassScores {
  const root = (src ?? {}) as Record<string, unknown>;
  // API_CONTRACT.md shape: scores = { dice: {kidney, tumor, cyst}, ... } or
  // null (null unless a ground-truth mask was uploaded). Read dice first.
  const scoresObj = root.scores as Record<string, unknown> | null | undefined;
  const dice = (scoresObj?.dice ?? null) as Record<string, unknown> | null;
  const s = ((dice ?? scoresObj ?? root.score ?? root) as Record<string, unknown>) || {};
  const pick = (k: string) => num(s[k] ?? s[`${k}_score`] ?? root[`${k}_score`]);
  return { kidney: pick("kidney"), tumor: pick("tumor"), cyst: pick("cyst") };
}
