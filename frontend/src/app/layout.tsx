import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "./components/Navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "SHL Assessment Recommender",
  description:
    "AI-powered SHL assessment recommendations based on job descriptions. Find the perfect assessment for any role in seconds.",
  keywords: "SHL, assessments, HR, recruitment, talent, AI recommendation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
