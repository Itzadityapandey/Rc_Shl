import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // In production on Vercel, /api/* is served by Python serverless functions.
  // In local dev, proxy /api/* to the Python dev server on port 5000.
  async rewrites() {
    return process.env.NODE_ENV === "development"
      ? [
        {
          source: "/api/:path*",
          destination: "http://localhost:5000/api/:path*",
        },
      ]
      : [];
  },
};

export default nextConfig;
