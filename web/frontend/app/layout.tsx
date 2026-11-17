import type { Metadata } from "next";
import { Newsreader, Schibsted_Grotesk } from "next/font/google";
import SiteNav from "../components/SiteNav";
import SiteFooter from "../components/SiteFooter";
import "./globals.css";

const display = Newsreader({ subsets: ["latin"], weight: ["300", "400", "500"], style: ["normal", "italic"], variable: "--font-display" });
const body = Schibsted_Grotesk({ subsets: ["latin"], variable: "--font-body" });

export const metadata: Metadata = {
  title: "KidneyScan AI: kidney tumor and cyst segmentation (KiTS23)",
  description:
    "Upload a kidney CT scan (.nii.gz) and explore the predicted kidney, tumor and cyst in 3D, with volumes in cm³. Research prototype. A doctor must confirm every result.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <SiteNav />
        <main id="main">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
