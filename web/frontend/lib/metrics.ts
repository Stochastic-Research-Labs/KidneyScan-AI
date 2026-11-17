// Single source of truth for every performance number shown in the UI.
// These are the best results so far. Do not edit them to look better.
export const BEST_DICE = {
  foreground: 0.4043,
  kidney: 0.787,
  tumor: 0.378,
  cyst: 0.048,
} as const;

export const MODEL_FACTS = {
  architecture: "Light 3D U-Net",
  parameters: "1.68M",
  dataset: "KiTS23",
  cases: 599,
  trainingCases: 489,
  testCases: 110,
} as const;

export function fmtDice(v: number, digits = 3) {
  return v.toFixed(digits);
}
