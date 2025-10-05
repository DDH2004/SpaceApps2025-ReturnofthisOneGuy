// app/page.tsx
"use client";

import React, { useRef } from "react";
import { motion } from "framer-motion";
import { Hero } from "@/components/Hero";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
import { Header } from "@/components/Header";
import CsvUploadPredictor from "@/components/CsvUploadPredictor";

export default function Page() {
  const heroRef = useRef<HTMLDivElement>(null);

  return (
    <main className="relative min-h-screen w-full bg-black text-white overflow-hidden">
      {/* Background dots fade in first */}
      <motion.div
        className="fixed inset-0 z-10 pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2, ease: "easeOut" }}
      >
        <ExoDotsCanvas />
      </motion.div>

      {/* Content appears 1s after background */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 1 }}
        className="relative z-20"
      >
        <Header />

        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

        <div ref={heroRef}>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 1 }}
          >
            <Hero />
              <div className="mt-10">
                <CsvUploadPredictor />
              </div>
          </motion.div>
        </div>
      </motion.div>
    </main>
  );
}
