import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "KidneyScan AI — Kidney Tumor + Cyst Segmentation (KiTS23)",
  description:
    "Upload a kidney CT (.nii.gz). A light 3D U-Net finds kidney, tumor, and cyst in 3D. Assistive only — a doctor must confirm.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col font-sans antialiased">
        <header className="border-b border-border bg-surface/80">
          <nav className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-4 px-5 py-4 sm:px-8">
            <Link href="/" className="text-lg font-bold text-text">
              KidneyScan <span className="text-primary">AI</span>
            </Link>
            <div className="flex gap-5 text-sm text-text-muted">
              <Link href="/" className="hover:text-text">Home</Link>
              <Link href="/upload" className="hover:text-text">Upload</Link>
              <Link href="/viewer" className="hover:text-text">Viewer</Link>
              <Link href="/results" className="hover:text-text">Results</Link>
            </div>
            <p className="ml-auto hidden text-xs text-text-muted sm:block">KiTS23 · Assistive only — a doctor must confirm</p>
          </nav>
        </header>
        <main id="main-content" className="flex-1 w-full">{children}</main>
        <footer className="border-t border-border px-5 py-6 text-center text-xs text-text-muted">
          KidneyScan AI · KiTS23 data, CC BY-NC-SA 4.0 · Free for research, not commercial · Not a diagnosis.
        </footer>
      </body>
    </html>
  );
}
