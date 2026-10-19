import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nephron-Aware Renal Digital Twin — Module 2",
  description:
    "Physiology-informed multivariate renal latent-state engine: causal dual-timeline inference with uncertainty quantification for MIMIC-IV longitudinal data.",
  keywords: ["CKD", "renal", "digital twin", "state-space model", "Kalman filter", "MIMIC-IV"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
