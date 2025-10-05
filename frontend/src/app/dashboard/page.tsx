"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { Header } from "@/components/Header";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
import { motion } from "framer-motion";
import Link from "next/link";
import { v4 as uuidv4 } from 'uuid'; 

// --- MODEL INPUT SCHEMAS ---
type FeatureInput = {
  orbital_period_days: number | "";
  transit_duration_hours: number | "";
  transit_depth_ppm: number | "";
  planet_radius_re: number | "";
  equilibrium_temp_k: number | "";
  insolation_flux_earth: number | "";
  stellar_teff_k: number | "";
  stellar_radius_re: number | "";
  apparent_mag: number | "";
  ra: number | "";
  dec: number | "";
};

type PredictionOutput = { 
  Predicted_Disposition: 'CONFIRMED' | 'FALSE POSITIVE'; 
  Confidence_Confirmed: number;
  Confidence_False_Positive: number;
};

// --- Initial state for the manual input form ---
const INITIAL_INPUT: FeatureInput = {
  orbital_period_days: 10.5,
  transit_duration_hours: 2.5,
  transit_depth_ppm: 600,
  planet_radius_re: 2.2,
  equilibrium_temp_k: 750,
  insolation_flux_earth: 90,
  stellar_teff_k: 5500,
  stellar_radius_re: 0.9,
  apparent_mag: 15.0,
  ra: 285.5,
  dec: 48.2,
};

// --- Utility Functions ---
const clamp = (x: number, min: number, max: number) => Math.max(min, Math.min(max, x));
function fmtNum(n: number) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return Number.isInteger(n) ? n.toString() : n.toFixed(4);
}

// ============================
// Radial confidence meter component
// ============================
function Radial({ p }: { p: number }) {
  const pct = clamp(p, 0, 1);
  const R = 38, C = 2 * Math.PI * R, off = C * (1 - pct);
  const band = pct < 0.6 ? "red" : pct < 0.8 ? "yellow" : "emerald";
  
  return (
    <div className="relative w-24 h-24">
      <svg viewBox="0 0 100 100" className="w-24 h-24">
        <circle cx="50" cy="50" r={R} className="fill-none stroke-[10] stroke-white/10" />
        <circle
          cx="50"
          cy="50"
          r={R}
          className={`fill-none stroke-[10] -rotate-90 origin-center stroke-${band}-400 transition-all duration-1000`}
          strokeDasharray={C}
          strokeDashoffset={off}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <div className="text-center">
          <div className="text-lg font-semibold">{Math.round(pct * 100)}%</div>
          <div className="text-[10px] uppercase tracking-wide text-white/60">
            confidence
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================
// Input Field Component
// ============================
function InputField({ label, unit, value, onChange }: { 
    label: string; 
    unit: string; 
    value: number | ""; 
    onChange: (v: number | "") => void; 
}) {
  return (
    <div className="flex flex-col">
      <label className="text-white/60 text-xs uppercase tracking-wide mb-1">
        {label}
      </label>
      <div className="relative">
        <input
          type="number"
          step="any"
          value={value}
          onChange={(e) => {
            const val = e.target.value;
            onChange(val === "" ? "" : parseFloat(val));
          }}
          className="w-full bg-white/[0.05] border border-white/10 rounded-lg p-2 text-white placeholder-white/30 focus:ring-emerald-400 focus:border-emerald-400"
          placeholder="Enter value"
        />
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 text-xs">
          {unit}
        </span>
      </div>
    </div>
  );
}

function Card({ title, className, children }: { title: string; className?: string; children: React.ReactNode; }) {
  return (
    <div
      className={`rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-5 md:p-7 ${
        className || ""
      }`}
    >
      <div className="mb-3 text-sm font-medium text-white/70">{title}</div>
      {children}
    </div>
  );
}


// ============================
// Main Dashboard
// ============================
export default function DashboardPage() {
  const [clientId, setClientId] = useState<string>('');
  const [inputData, setInputData] = useState<FeatureInput>(INITIAL_INPUT);
  const [output, setOutput] = useState<PredictionOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null); // Reference to the hidden file input

  // New states for CSV upload
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvTaskId, setCsvTaskId] = useState<string | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);
  // Multimodal image upload state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [fusionProb, setFusionProb] = useState<number | null>(null);
  const [fusionLoading, setFusionLoading] = useState(false);

  // Set unique client ID on mount for WebSocket communication
  useEffect(() => {
    setClientId(uuidv4());
  }, []);

  // --- HANDLER FOR SINGLE PREDICTION ---
  const analyzeSingle = async () => {
    // ... (analyzeSingle logic remains the same)
    setError(null);
    setLoading(true);
    setOutput(null);

    const payload = Object.fromEntries(
        Object.entries(inputData)
            .map(([key, value]) => [key, (Number.isFinite(value) ? value : null)])
    );

    try {
      // Send single synchronous prediction request
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        // Try to parse JSON error body, otherwise use text
        let errText = `Server error: ${res.status}`;
        try {
          const errData = await res.json();
          errText = errData.detail ? JSON.stringify(errData.detail) : JSON.stringify(errData);
        } catch (e) {
          try {
            errText = await res.text();
          } catch (ee) {
            /* ignore */
          }
        }
        throw new Error(errText);
      }

      const data = await res.json();
      // backend returns { result: { probability: ... } }
      const prob = data?.result?.probability ?? data?.probability ?? null;
      if (prob !== null) {
        setFusionProb(prob);
        setOutput(null);
      } else {
        setError('Prediction returned no probability');
      }

    } catch (e: any) {
      // Ensure we stringify objects so they render nicely in the UI
      const msg = typeof e === 'string' ? e : (e?.message ?? JSON.stringify(e));
      setError(msg || "Failed to connect to ML API.");
    } finally {
      setLoading(false);
    }
  };

  // --- HANDLER FOR MULTIMODAL (TABULAR + IMAGE) ---
  const analyzeMultimodal = async () => {
    setFusionLoading(true);
    setFusionProb(null);
    setError(null);

    const tabularArray = Object.values(inputData).map(v => Number.isFinite(v) ? v : 0.0);

    const form = new FormData();
    form.append('tabular', JSON.stringify(tabularArray));
    if (imageFile) form.append('image', imageFile);

    try {
      const res = await fetch('http://localhost:8000/predict/multimodal', {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setFusionProb(data.probability ?? null);
    } catch (e: any) {
      setError(e?.message || 'Fusion prediction failed');
    } finally {
      setFusionLoading(false);
    }
  };
  
  // --- HANDLER FOR BATCH UPLOAD/SUBMISSION ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    setCsvFile(f);
    setCsvTaskId(null);
    setError(null);
  };
  
  const uploadBatch = async () => {
    if (!csvFile || !clientId) {
      setError("Please select a file and ensure client ID is set.");
      return;
    }
    setCsvLoading(true);
    setError(null);
    setCsvTaskId(null);

    const formData = new FormData();
    formData.append("file", csvFile); 
    formData.append("client_id", clientId); 

    try {
      // Use the synchronous CSV endpoint for now to avoid server-side batch-size rejections
      const res = await fetch("http://localhost:8000/predict/csv_tabular", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let errText = `Server rejected file: ${res.status}`;
        try {
          const errData = await res.json();
          errText = errData.detail ? JSON.stringify(errData.detail) : JSON.stringify(errData);
        } catch (e) {
          try { errText = await res.text(); } catch (_) {}
        }
        throw new Error(errText);
      }

      const data = await res.json();
      // data.predictions is an array of { probability } or { error }
      // Store a short summary (number of rows and any errors)
      const preds = data.predictions || [];
      const errors = preds.filter((p: any) => p.error).length;
      setCsvTaskId(`sync:${preds.length}:${errors}`);
      // Optionally, you could store full results in state for a results view
      
    } catch (e: any) {
      const msg = typeof e === 'string' ? e : (e?.message ?? JSON.stringify(e));
      setError(msg || "Failed to submit batch CSV.");
    } finally {
      setCsvLoading(false);
    }
  };


  // --- Output Display Logic (remains the same) ---
  const currentConfidence = useMemo(() => output 
    ? output.Predicted_Disposition === 'CONFIRMED' ? output.Confidence_Confirmed : output.Confidence_False_Positive
    : 0, [output]);
  
  const verdict = output ? (
    output.Predicted_Disposition === 'CONFIRMED' ? "Yes, likely an Exoplanet" : "No, likely a Non-planet (False Positive)"
  ) : "—";
  
  const bandColor = output ? (
    currentConfidence < 0.7 ? "text-yellow-400" : "text-emerald-400"
  ) : "text-white/70";

  return (
    <main className="relative min-h-screen bg-black text-white overflow-hidden">
      <Header />
      <div className="fixed inset-0 z-0 opacity-35 pointer-events-none">
        <ExoDotsCanvas />
      </div>

      <section className="relative z-10 px-6 pt-20 pb-28 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 flex items-center justify-between gap-4"
        >
          <h1 className="text-2xl md:text-4xl font-semibold tracking-tight">
            Exoplanet Vetting Tool
          </h1>
          <Link href="/" className="text-white/60 hover:text-white text-sm">
            ← back
          </Link>
        </motion.div>

        <div className="grid md:grid-cols-5 gap-6">
          
          {/* 1) Input Form (Col 1-3) */}
          <Card title="1) Manual Vetting (Single Candidate)" className="md:col-span-3">
            <div className="grid grid-cols-2 gap-4">
              <InputField label="Orbital Period" unit="days" value={inputData.orbital_period_days} onChange={(v) => setInputData(p => ({ ...p, orbital_period_days: v }))} />
              <InputField label="Transit Duration" unit="hours" value={inputData.transit_duration_hours} onChange={(v) => setInputData(p => ({ ...p, transit_duration_hours: v }))} />
              <InputField label="Transit Depth" unit="ppm" value={inputData.transit_depth_ppm} onChange={(v) => setInputData(p => ({ ...p, transit_depth_ppm: v }))} />
              <InputField label="Planetary Radius" unit="R_Earth" value={inputData.planet_radius_re} onChange={(v) => setInputData(p => ({ ...p, planet_radius_re: v }))} />
              <InputField label="Equilibrium Temp" unit="K" value={inputData.equilibrium_temp_k} onChange={(v) => setInputData(p => ({ ...p, equilibrium_temp_k: v }))} />
              <InputField label="Insolation Flux" unit="Earth Flux" value={inputData.insolation_flux_earth} onChange={(v) => setInputData(p => ({ ...p, insolation_flux_earth: v }))} />
              <InputField label="Stellar Temp" unit="K" value={inputData.stellar_teff_k} onChange={(v) => setInputData(p => ({ ...p, stellar_teff_k: v }))} />
              <InputField label="Stellar Radius" unit="R_Sun" value={inputData.stellar_radius_re} onChange={(v) => setInputData(p => ({ ...p, stellar_radius_re: v }))} />
              <InputField label="Apparent Mag" unit="mag" value={inputData.apparent_mag} onChange={(v) => setInputData(p => ({ ...p, apparent_mag: v }))} />
              <InputField label="RA" unit="deg" value={inputData.ra} onChange={(v) => setInputData(p => ({ ...p, ra: v }))} />
              <InputField label="Dec" unit="deg" value={inputData.dec} onChange={(v) => setInputData(p => ({ ...p, dec: v }))} />
            </div>
            <div className="mt-6">
                <button
                    onClick={analyzeSingle}
                    disabled={loading}
                    className="w-full px-6 py-3 rounded-xl bg-emerald-600 text-white font-semibold text-lg hover:bg-emerald-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? "Analyzing..." : "Analyze Single Candidate"}
                </button>
            </div>
          </Card>
          
          {/* 2) Output Display (Col 4-5) */}
          <Card title="2) Model Output" className="md:col-span-2 flex flex-col justify-between">
            <div>
              <div className="pt-6">
                <div className="flex items-center gap-5">
                  <Radial p={currentConfidence} />
                  <div>
                    <div className="text-sm uppercase tracking-wide text-white/60">
                      Verdict
                    </div>
                    <div className={`mt-1 text-2xl font-semibold ${bandColor}`}>
                      {verdict}
                    </div>
                    {output && (
                      <div className="mt-1 text-sm text-white/70">
                        {output.Predicted_Disposition === 'CONFIRMED' 
                          ? `FP Probability: ${fmtNum(output.Confidence_False_Positive)}`
                          : `Confirmed Probability: ${fmtNum(output.Confidence_Confirmed)}`
                        }
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <div className="text-sm text-white/70 mb-2">Optional multimodal image:</div>
              <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
              <div className="mt-3 flex gap-3">
                <button onClick={analyzeMultimodal} disabled={fusionLoading} className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500">
                  {fusionLoading ? 'Running...' : 'Analyze with Image (Fusion)'}
                </button>
                {fusionProb !== null && (
                  <div className="text-sm text-white/80">Fusion Probability: <b>{Math.round(fusionProb * 100)}%</b></div>
                )}
              </div>
            </div>
          </Card>
        </div>
        
        {/* 3) CSV Batch Upload Section (The Fix) */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-6"
        >
          <Card title="3) Batch Vetting (CSV Upload)" className="md:col-span-5">
            <p className="mb-4 text-sm text-white/70">
              Upload a CSV file containing multiple candidates for batch processing.
            </p>
            <div className="flex items-center space-x-4">
              
              {/* Hidden File Input */}
              <input
                type="file"
                accept=".csv"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />

              {/* Custom 'Choose file' Button */}
              <button
                onClick={() => fileInputRef.current?.click()} // Click the hidden input
                disabled={csvLoading}
                className="px-6 py-2 rounded-xl bg-white/10 text-white font-semibold hover:bg-white/20 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Choose File
              </button>

              <div className="text-white/70 text-sm flex-grow">
                {csvFile ? <b>{csvFile.name}</b> : "No file selected"}
              </div>

              {/* Submit Button */}
              <button
                onClick={uploadBatch}
                disabled={csvLoading || !csvFile}
                className="px-6 py-2 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-500 transition disabled:opacity-50"
              >
                {csvLoading ? "Uploading..." : "Start Batch Analysis"}
              </button>
            </div>

            {csvTaskId && (
              <div className="mt-4 p-3 bg-blue-900/30 rounded-lg text-sm">
                Batch Submitted! Task ID: <b>{csvTaskId}</b>
                <p className="text-xs mt-1 text-white/50">
                  Monitor task status using the backend API endpoint.
                </p>
              </div>
            )}
          </Card>
        </motion.div>
        
        {/* Error/Log display */}
        {error && <div className="mt-6 p-4 bg-red-900/30 text-red-300 rounded-xl text-sm">Error: {error}</div>}

      </section>
    </main>
  );
}