import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Proxy the local FastAPI service through the Next.js origin, same pattern
  // as NeuroScan: browser requests stay same-origin, FastAPI runs on :8000.
  async rewrites() {
    return [{ source: "/backend/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};

export default nextConfig;
