import Link from "next/link";
import HeroKidney from "../components/HeroKidney";
import DiceTable from "../components/DiceTable";
import { LABELS } from "../lib/labels";
import { MODEL_FACTS } from "../lib/metrics";

const STEPS = [
  { t: "Prepare", d: "Choose one 3D CT scan as .nii.gz. It is read in your browser, so you can inspect it before anything is sent." },
  { t: "Segment", d: "The light 3D U-Net labels kidney, tumor and cyst, and returns a mask with a score for each class." },
  { t: "Explore", d: "Rotate the original CT and the overlay side by side, hide structures, and read volumes in cm³." },
];

const LABEL_NOTES: Record<string, string> = {
  kidney: "The organ itself. The strongest result so far.",
  tumor: "Solid lesions. Found partially, so review every boundary.",
  cyst: "Fluid-filled lesions. The model is still close to failing here.",
};

export default function Home() {
  return (
    <>
      <section className="ks-hero">
        <div className="ks-wrap ks-hero-grid">
          <div className="ks-hero-copy">
            <p className="ks-kicker">Kidney CT, in three dimensions</p>
            <h1>See inside the scan.</h1>
            <p className="ks-lede">
              Turn a kidney CT into an explorable 3D view of the kidney, tumor and cyst, with volumes in cm³. Built for learning and research,
              with the model&apos;s real scores shown plainly.
            </p>
            <div className="ks-cta">
              <Link href="/workspace" className="ks-btn ks-btn-primary">
                Start an analysis
              </Link>
              <Link href="/workspace?mode=dummy" className="ks-btn ks-btn-ghost">
                Try the dummy demo
              </Link>
            </div>
            <p className="ks-small">
              Built on KiTS23 ({MODEL_FACTS.cases} cases). Assistive only: a doctor must confirm every result.
            </p>
          </div>
          <div className="ks-hero-visual">
            <HeroKidney />
            <p className="ks-caption">Conceptual illustration. Not a patient scan and not a model output.</p>
          </div>
        </div>
      </section>

      <section className="ks-section">
        <div className="ks-wrap">
          <h2>From one scan to a clearer view</h2>
          <ol className="ks-steps">
            {STEPS.map((s, i) => (
              <li key={s.t}>
                <span className="ks-step-n">{i + 1}</span>
                <h3>{s.t}</h3>
                <p>{s.d}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="ks-section ks-section-alt">
        <div className="ks-wrap ks-split">
          <div>
            <h2>Three structures, in context</h2>
            <p className="ks-prose">
              Switch between the original CT and the segmentation, hide any structure, and cut away the front of the body to look inside. Both
              views rotate together, so you always compare the same angle.
            </p>
            <ul className="ks-labels">
              {LABELS.map((l) => (
                <li key={l.key}>
                  <i style={{ background: l.color }} aria-hidden />
                  <div>
                    <b>{l.name}</b> <span className="ks-muted">label {l.id}</span>
                    <p>{LABEL_NOTES[l.key]}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
          <div className="ks-card ks-results-card">
            <h3>Best results so far</h3>
            <p className="ks-muted">Dice score: 1 is a perfect match with the expert outline, 0 is no overlap.</p>
            <DiceTable />
            <p className="ks-small">
              The kidney is segmented reasonably well. The tumor is only partly found, and the cyst is close to failing. Read every prediction
              as a starting point for expert review.
            </p>
            <Link href="/insights" className="ks-link">
              How to read these numbers
            </Link>
          </div>
        </div>
      </section>

      <section className="ks-section ks-final">
        <div className="ks-wrap">
          <h2>A closer look, not a diagnosis.</h2>
          <p className="ks-prose">
            A normal scan should show no tumor and no cyst, because the model found none. Nothing is hard-coded. Dummy mode is clearly
            labelled and uses synthetic shapes.
          </p>
          <Link href="/workspace" className="ks-btn ks-btn-primary">
            Open the workspace
          </Link>
        </div>
      </section>
    </>
  );
}
