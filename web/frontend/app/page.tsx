import Link from "next/link";
import { Upload, ScanEye, BarChart3 } from "lucide-react";

export default function HomePage() {
  return (
    <div className="page-shell space-y-10">
      <section className="space-y-4">
        <p className="eyebrow">KiTS23 · Kidney tumor + cyst · 3D CT</p>
        <h1 className="max-w-3xl text-4xl font-bold leading-tight sm:text-5xl">
          KidneyScan AI — see kidney, tumor, and cyst in 3D.
        </h1>
        <p className="max-w-2xl text-base leading-7 text-text-muted">
          Upload a kidney CT scan (.nii.gz). The model finds the kidney, tumor,
          and cyst, shows them in a 3D viewer you can rotate 360°, and gives
          volumes in cm³. Free for doctors and students.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link href="/upload" className="btn-primary"><Upload size={16} aria-hidden="true" /> Upload a CT</Link>
          <Link href="/viewer" className="btn-ghost"><ScanEye size={16} aria-hidden="true" /> Try the dummy demo</Link>
        </div>
        <p className="rounded-lg border border-yellow-500/40 bg-yellow-500/10 px-4 py-3 text-sm text-yellow-200">
          Assistive tool only. A doctor must confirm the result. This is not a diagnosis.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          { Icon: Upload, title: "1. Upload", text: "One CT file (.nii.gz). The upload selects the input only — the model decides the output." },
          { Icon: ScanEye, title: "2. View in 3D", text: "Two panels: original kidney on the left, prediction on the right. Rotate, zoom, fullscreen." },
          { Icon: BarChart3, title: "3. Read the result", text: "Kidney, tumor, and cyst volumes in cm³. Dice and HD95 scores when ground truth is given." },
        ].map(({ Icon, title, text }) => (
          <div key={title} className="glass-panel p-5">
            <Icon size={20} className="text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-semibold">{title}</h2>
            <p className="mt-2 text-sm leading-6 text-text-muted">{text}</p>
          </div>
        ))}
      </section>

      <section className="glass-panel p-5 text-sm leading-7 text-text-muted">
        <h2 className="font-semibold text-text">What it is — and what it is not</h2>
        <p className="mt-2">
          KidneyScan AI is a light 3D U-Net trained on KiTS23 (489 training cases,
          labels: kidney, tumor, cyst). It is built to help — for small hospitals
          with no kidney specialist, and for students learning CT segmentation.
          A normal scan shows no tumor and no cyst highlight, because the model
          found nothing — nothing is hard-coded. The demo mode shows clearly
          labelled DUMMY / TEST shapes until the real model is ready.
        </p>
      </section>
    </div>
  );
}
