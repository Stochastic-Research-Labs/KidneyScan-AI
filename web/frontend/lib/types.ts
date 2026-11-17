import type { LabelKey } from "./labels";

export type ClassScores = Record<LabelKey, number | null>;
export type ClassVolumes = Record<LabelKey, number>; // cm3

export type PredictResult = {
  /** The segmentation mask as a (possibly gzipped) .nii.gz file. */
  maskBuffer: ArrayBuffer;
  scores: ClassScores;
};
