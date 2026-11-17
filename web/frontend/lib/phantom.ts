import type { Grid } from "./nifti";
import type { ClassVolumes } from "./types";

/**
 * DUMMY / TEST data. A synthetic torso with two kidneys, one sphere labelled
 * tumor and one labelled cyst. It exists only to test the viewer. It is not a
 * patient scan and no model is involved.
 */
export function makeDummy(): { ct: Grid; mask: Grid; volumes: ClassVolumes } {
  const nx = 128, ny = 96, nz = 112, mm = 2.5;
  const ct = new Uint8Array(nx * ny * nz);
  const mask = new Uint8Array(nx * ny * nz);
  const e = (x: number, y: number, z: number, cx: number, cy: number, cz: number, rx: number, ry: number, rz: number) =>
    ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1;
  const toByte = (hu: number) => Math.max(0, Math.min(255, ((hu + 160) / 480) * 255));
  const counts = [0, 0, 0, 0];
  let i = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        let hu = -1000;
        const body = ((x - 64) / 58) ** 2 + ((y - 48) / 40) ** 2 <= 1;
        const inner = ((x - 64) / 54) ** 2 + ((y - 48) / 36) ** 2 <= 1;
        if (body) hu = -90;
        if (inner) hu = 40;
        if (inner && (x - 64) ** 2 + (y - 16) ** 2 <= 81) hu = z % 14 > 11 ? 150 : 450; // spine
        if (inner && e(x, y, z, 64, 64, 40, 10, 7, 12)) hu = -400; // gas pocket
        let label = 0;
        if (e(x, y, z, 90, 28, 58, 9, 7, 17) || e(x, y, z, 38, 28, 58, 9, 7, 17)) { hu = 170; label = 1; }
        if (e(x, y, z, 96, 28, 64, 6.5, 6.5, 6.5)) { hu = 90; label = 2; } // tumor (left kidney)
        if (e(x, y, z, 35, 30, 50, 4.5, 4.5, 4.5)) { hu = 15; label = 3; } // cyst (right kidney)
        const noise = ((((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) & 255) / 255 - 0.5) * 12;
        ct[i] = toByte(hu + noise);
        mask[i] = label;
        counts[label]++;
      }
    }
  }
  const extentsMm: [number, number, number] = [nx * mm, ny * mm, nz * mm];
  const voxelCm3 = (mm * mm * mm) / 1000;
  return {
    ct: { data: ct, size: [nx, ny, nz], extentsMm },
    mask: { data: mask, size: [nx, ny, nz], extentsMm },
    volumes: { kidney: counts[1] * voxelCm3, tumor: counts[2] * voxelCm3, cyst: counts[3] * voxelCm3 },
  };
}
