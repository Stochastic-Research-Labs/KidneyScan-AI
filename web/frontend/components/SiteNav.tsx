"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/workspace", label: "Workspace" },
  { href: "/insights", label: "Model insights" },
  { href: "/about", label: "About" },
];

export default function SiteNav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="ks-nav">
      <a className="ks-skip" href="#main">
        Skip to content
      </a>
      <div className="ks-wrap ks-nav-inner">
        <Link href="/" className="ks-brand" onClick={() => setOpen(false)}>
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
            <path d="M19 4c-5 0-9 3.600-9 9.500 0 2.800 1.200 4.500 1.200 6.500 0 2.300-1.700 3-1.700 5 0 2.200 2 3 4.500 3 5.500 0 9.500-5 9.500-11C23.500 10 22.500 4 19 4Z" fill="none" stroke="currentColor" strokeWidth="1.700" />
            <circle cx="21" cy="12" r="2.600" fill="var(--tumor)" />
          </svg>
          <span>
            KidneyScan<b>AI</b>
          </span>
        </Link>
        <button type="button" className="ks-burger" aria-expanded={open} aria-controls="ks-menu" onClick={() => setOpen(!open)}>
          {open ? "Close" : "Menu"}
        </button>
        <nav id="ks-menu" className={`ks-links${open ? " is-open" : ""}`} aria-label="Main">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} aria-current={path === l.href ? "page" : undefined} onClick={() => setOpen(false)}>
              {l.label}
            </Link>
          ))}
          <Link href="/workspace" className="ks-btn ks-btn-primary ks-btn-sm" onClick={() => setOpen(false)}>
            New analysis
          </Link>
        </nav>
      </div>
    </header>
  );
}
