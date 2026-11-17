import * as nifti from "nifti-reader-js";
import type { ClassVolumes } from "./types";

/**
 * How voxel axes map to display axes. Display axes are fixed:
 *   0 = patient Left  (viewer's right), 1 = Anterior, 2 = Superior.
 * perm[k] = which voxel axis feeds display axis k; flip[k] = reverse it.
 */
export type AxisMap = { perm: [number, number, number]; flip: [boolean, boolean, boolean] };

export type RawVolume = {
  dims: [number, number, number];
  spacing: [number, number, number]; // mm per voxel
  data: ArrayLike<number>;
  slope: number;
  inter: number;
  map: AxisMap;
};

/** A downsampled 8-bit volume ready to be uploaded as a 3D texture. */
export type Grid = {
  data: Uint8Array;
  size: [number, number, number];
  extentsMm: [number, number, number]; // physical size per display axis
};

const DESIRED_DIR = [-1, 1, 1]; // RAS+ world x is Right, so Left = -x

const IDENTITY: AxisMap = { perm: [0, 1, 2], flip: [true, false, false] }; // voxel axes assumed R, A, S

function axisMapFromAffine(aff?: number[][]): AxisMap {
  if (!aff || aff.length < 3) return IDENTITY;
  const world = [0, 1, 2].map((i) => {
    const col = [aff[0]?.[i] ?? 0, aff[1]?.[i] ?? 0, aff[2]?.[i] ?? 0];
    let w = 0;
    for (let k = 1; k < 3; k++) if (Math.abs(col[k]) > Math.abs(col[w])) w = k;
    return { w, s: Math.sign(col[w]) || 1 };
  });
  if (new Set(world.map((x) => x.w)).size !== 3) return IDENTITY;
  const perm: [number, number, number] = [0, 0, 0];
  const flip: [boolean, boolean, boolean] = [false, false, false];
  world.forEach((o, i) => {
    perm[o.w] = i;
    flip[o.w] = o.s * DESIRED_DIR[o.w] < 0;
  });
  return { perm, flip };
}

function typedView(code: number, buf: ArrayBuffer): ArrayLike<number> {
  switch (code) {
    case 2: return new Uint8Array(buf);
    case 4: return new Int16Array(buf);
    case 8: return new Int32Array(buf);
    case 16: return new Float32Array(buf);
    case 64: return new Float64Array(buf);
    case 256: return new Int8Array(buf);
    case 512: return new Uint16Array(buf);
    case 768: return new Uint32Array(buf);
    default: throw new Error(`Unsupported NIfTI data type (code ${code}).`);
  }
}

/** Parse a .nii or .nii.gz file held in memory. */
export function readNifti(input: ArrayBuffer): RawVolume {
  let buf = input;
  if (nifti.isCompressed(buf)) buf = nifti.decompress(buf) as ArrayBuffer;
  if (!nifti.isNIFTI(buf)) throw new Error("This file is not a NIfTI volume. Use a .nii or .nii.gz CT scan.");
  const h = nifti.readHeader(buf);
  if (!h) throw new Error("Could not read the NIfTI header.");
  if (!h.littleEndian) throw new Error("Big-endian NIfTI files are not supported.");
  const nz = h.dims[3] || 1;
  const img = nifti.readImage(h, buf);
  const slope = h.scl_slope && h.scl_slope !== 0 ? h.scl_slope : 1;
  return {
    dims: [h.dims[1], h.dims[2], nz],
    spacing: [Math.abs(h.pixDims[1]) || 1, Math.abs(h.pixDims[2]) || 1, Math.abs(h.pixDims[3]) || 1],
    data: typedView(h.datatypeCode, img),
    slope,
    inter: h.scl_inter || 0,
    map: axisMapFromAffine((h as { affine?: number[][] }).affine),
  };
}

function ctWindow(v: RawVolume): [number, number] {
  const n = v.dims[0] * v.dims[1] * v.dims[2];
  const step = Math.max(1, Math.floor(n / 200000));
  const s = new Float32Array(Math.ceil(n / step));
  let m = 0;
  for (let i = 0; i < n; i += step) s[m++] = v.data[i] * v.slope + v.inter;
  const sorted = s.subarray(0, m).sort();
  const q = (p: number) => sorted[Math.min(m - 1, Math.floor(p * m))];
  // Looks like Hounsfield units: use a soft-tissue window that also keeps bone.
  if (q(0.005) < -300 && q(0.995) > 100) return [-160, 320];
  const lo = q(0.01);
  return [lo, Math.max(q(0.995), lo + 1)];
}

/** Resample to a display-ordered 8-bit grid (max `maxDim` voxels per axis). */
export function toGrid(v: RawVolume, kind: "ct" | "mask", map: AxisMap = v.map, maxDim = 192): Grid {
  const stride = [1, v.dims[0], v.dims[0] * v.dims[1]];
  const size = [0, 1, 2].map((k) => Math.min(v.dims[map.perm[k]], maxDim)) as [number, number, number];
  const lut = [0, 1, 2].map((k) => {
    const n = v.dims[map.perm[k]];
    const N = size[k];
    const out = new Int32Array(N);
    for (let o = 0; o < N; o++) {
      let i = N > 1 ? Math.round((o * (n - 1)) / (N - 1)) : 0;
      if (map.flip[k]) i = n - 1 - i;
      out[o] = i * stride[map.perm[k]];
    }
    return out;
  });
  const extentsMm = [0, 1, 2].map((k) => v.dims[map.perm[k]] * v.spacing[map.perm[k]]) as [number, number, number];
  const data = new Uint8Array(size[0] * size[1] * size[2]);
  const [lo, hi] = kind === "ct" ? ctWindow(v) : [0, 1];
  const scale = 255 / (hi - lo);
  let p = 0;
  for (let c = 0; c < size[2]; c++) {
    for (let b = 0; b < size[1]; b++) {
      const base = lut[1][b] + lut[2][c];
      for (let a = 0; a < size[0]; a++) {
        const raw = v.data[lut[0][a] + base] * v.slope + v.inter;
        if (kind === "ct") {
          const t = (raw - lo) * scale;
          data[p++] = t < 0 ? 0 : t > 255 ? 255 : t;
        } else {
          const r = Math.round(raw);
          data[p++] = r >= 1 && r <= 3 ? r : 0;
        }
      }
    }
  }
  return { data, size, extentsMm };
}

/** Exact volumes in cm3, counted on the full-resolution mask. */
export function labelVolumesCm3(v: RawVolume): ClassVolumes {
  const counts = [0, 0, 0, 0];
  const n = v.dims[0] * v.dims[1] * v.dims[2];
  for (let i = 0; i < n; i++) {
    const r = Math.round(v.data[i] * v.slope + v.inter);
    if (r >= 1 && r <= 3) counts[r]++;
  }
  const voxelCm3 = (v.spacing[0] * v.spacing[1] * v.spacing[2]) / 1000;
  return { kidney: counts[1] * voxelCm3, tumor: counts[2] * voxelCm3, cyst: counts[3] * voxelCm3 };
}

/** True when two grids cover clearly different physical extents. */
export function extentsDiffer(a: Grid, b: Grid, tol = 0.05) {
  return a.extentsMm.some((e, k) => Math.abs(e - b.extentsMm[k]) / Math.max(e, 1) > tol);
}
