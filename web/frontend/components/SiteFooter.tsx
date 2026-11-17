import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="ks-footer">
      <div className="ks-wrap">
        <p className="ks-footer-warn">
          Research prototype. Assistive only: a doctor must confirm every result. Not a diagnosis and not validated for clinical decisions.
        </p>
        <div className="ks-footer-grid">
          <p>
            KidneyScan AI, built on{" "}
            <a href="https://kits-challenge.org/kits23/" target="_blank" rel="noreferrer">
              KiTS23
            </a>
            . A project of Stochastic Research Labs.
          </p>
          <p>Code: MIT. KiTS23 data: CC BY-NC-SA 4.0, free for research, not commercial.</p>
          <p>
            <Link href="/about">About the project</Link>
            {" · "}
            <a href="https://github.com/Stochastic-Research-Labs/KidneyScan-AI" target="_blank" rel="noreferrer">
              Source code
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
}
