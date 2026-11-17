export type LabelKey = "kidney" | "tumor" | "cyst";

export const LABELS: { id: 1 | 2 | 3; key: LabelKey; name: string; color: string }[] = [
  { id: 1, key: "kidney", name: "Kidney", color: "#34d3c0" },
  { id: 2, key: "tumor", name: "Tumor", color: "#ff5d5d" },
  { id: 3, key: "cyst", name: "Cyst", color: "#ffc83d" },
];
