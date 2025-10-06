// app/dashboard/page.tsx
"use client";

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { Header } from "@/components/Header";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";

import { GlowCard } from "@/components/ui/GlowCard";
import { ProbabilityOrb } from "@/components/charts/ProbabilityOrb";
import { ColumnHistogram } from "@/components/charts/ColumnHistogram";
import { ScatterPlot } from "@/components/charts/ScatterPlot";
import { RadarFeatures } from "@/components/charts/RadarFeature";

import { FeatureBars } from "@/components/FeaturesBar";
import { RunHistory, addRunToHistory } from "@/components/RunHistory";
import { AnalysisSkeleton } from "@/components/AnalysisSkeleton";
import { CustomTabularMapper } from "@/components/CustomTabularMarker";

type ModelOutput = { predicted_label: 0 | 1; predicted_proba: number };

// --- utilities ---
function parseCSV(text: string) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const cols = line.split(",");
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => (obj[h] = (cols[i] ?? "").trim()));
    return obj;
  });
  return { headers, rows };
}

const REQ_TABULAR_MIN = [
  "id",
  "mission",
  "disposition",
  "label",
  "orbital_period_days",
  "transit_duration_hours",
  "transit_depth_ppm",
  "planet_radius_re",
  "equilibrium_temp_k",
  "insolation_flux_earth",
  "stellar_teff_k",
  "stellar_radius_re",
  "stellar_mass_sun",
  "stellar_logg_cgs",
  "stellar_metallicity_fe",
  "impact_param",
  "apparent_mag",
  "ra",
  "dec"
] as const;
const REQ_MULTIMODAL = [
  "kepid",
  "koi_period",
  "koi_depth",
  "koi_duration",
  "koi_steff",
  "koi_srad",
  "koi_smass",
  "koi_slogg",
  "koi_smet",
  "koi_impact",
  "residual_windows_file",
  "pixel_diffs_file",
] as const;

type Tab = "Tabular (39)" | "Multimodal";

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("Tabular (39)");

  return (
    <main className="relative min-h-screen bg-black text-white overflow-hidden">
      <Header />
      <div className="fixed inset-0 z-0 opacity-30 pointer-events-none">
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
            Exoplanet Inference Lab
          </h1>
          <Link href="/" className="text-white/60 hover:text-white text-sm">
            ← back
          </Link>
        </motion.div>

        <div className="inline-flex rounded-2xl border border-white/10 bg-white/[0.03] p-1">
          {(["Tabular (39)", "Multimodal"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 rounded-xl text-sm transition ${
                tab === t
                  ? "bg-white text-black"
                  : "text-white/70 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="mt-8">
          {tab === "Tabular (39)" ? <TabularPanel /> : <MultimodalPanel />}
        </div>
      </section>
    </main>
  );
}

/* -------------------------- Tabular (39) -------------------------- */
function TabularPanel() {
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [output, setOutput] = useState<ModelOutput | null>(null);
  const [featuresEcho, setFeaturesEcho] = useState<Record<
    string,
    number
  > | null>(null);
  const [extras, setExtras] = useState<any | null>(null);
  // inside TabularPanel
  const [useCustomMap, setUseCustomMap] = useState(false);

  {
    /* toggle */
  }
  <label className="inline-flex items-center gap-2 text-sm text-white/70 mt-4">
    <input
      type="checkbox"
      className="accent-white"
      checked={useCustomMap}
      onChange={(e) => setUseCustomMap(e.target.checked)}
    />
    Use Custom CSV (map columns to tabular schema)
  </label>;

  {
    useCustomMap && rows.length > 0 && (
      <CustomTabularMapper
        headers={headers}
        rows={rows}
        onPredict={async (mappedRow) => {
          // you can add placeholder symmetric errors here if wanted
          const row = {
            ...mappedRow,
            koi_period_err1: 1e-4 * Number(mappedRow.koi_period || 0),
            koi_period_err2: -1e-4 * Number(mappedRow.koi_period || 0),
            koi_time0bk: "",
            koi_time0bk_err1: "",
            koi_time0bk_err2: "",
            koi_depth_err1: "",
            koi_depth_err2: "",
            koi_duration_err1: "",
            koi_duration_err2: "",
            koi_impact_err1: "",
            koi_impact_err2: "",
            koi_quarters: "",
          };
          const res = await fetch("/api/predict/tabular", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ row }),
          });
          const data = await res.json();
          // reuse your existing setOutput / setExtras / setFeaturesEcho
          setOutput({
            predicted_label: data.predicted_label,
            predicted_proba: data.predicted_proba,
          });
          setFeaturesEcho(data.debug_features ?? null);
          setExtras(data.extras ?? null);
        }}
      />
    );
  }

  const haveMin = useMemo(
    () => REQ_TABULAR_MIN.every((k) => headers.includes(k)),
    [headers]
  );

  const onFile = async (file: File) => {
    setError(null);
    setOutput(null);
    setFeaturesEcho(null);
    setExtras(null);
    setSelected(null);
    const text = await file.text();
    const { headers: h, rows: r } = parseCSV(text);
    setHeaders(h);
    setRows(r);
    const missing = REQ_TABULAR_MIN.filter((k) => !h.includes(k));
    if (missing.length > 0) {
      setError(`Warning! There are missing/misnamed columns: ${missing.join(", ")}`);
    }
  };

  const analyze = async () => {
    if (selected == null) return;
    setAnalyzing(true);
    const row = rows[selected];
    try {
      const res = await fetch("/api/predict/tabular", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row }),
      });
      if (!res.ok) throw new Error("prediction failed");
      const data = await res.json();
      console.log(
        "pred:",
        data.predicted_proba,
        "top_features:",
        data.extras?.top_features
      );
      console.log(
        "debug_features keys:",
        Object.keys(data.debug_features || {})
      );

      setOutput({
        predicted_label: data.predicted_label,
        predicted_proba: data.predicted_proba,
      });
      setFeaturesEcho(data.debug_features ?? null);
      setExtras(data.extras ?? null);

      const id = row.kepid || row.KepID || `row-${selected + 1}`;
      addRunToHistory({
        id: String(id),
        label: data.predicted_label === 1 ? "Planet" : "Non-planet",
        proba: Number(data.predicted_proba ?? 0),
        ts: Date.now(),
      });
    } catch (e: any) {
      setError(e?.message || "Prediction error.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* LEFT: Upload + dataset charts + table */}
      <div className="lg:col-span-2 space-y-6">
        <GlowCard>
          <div className="mb-3 text-sm font-medium text-white/70">
            Upload KOI/Kepler CSV (Tabular)
          </div>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
            className="border border-dashed border-white/20 rounded-2xl p-8 text-center hover:border-white/40 transition"
          >
            <p className="text-white/70">
              Drag & drop a CSV (Template 1 or 2).
            </p>
            <div className="mt-4">
              <label className="inline-block px-4 py-2 rounded-xl bg-white text-black font-medium cursor-pointer hover:opacity-90">
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFile(f);
                  }}
                />
                Choose file
              </label>
            </div>
            {error && <p className="mt-3 text-red-300 text-sm">{error}</p>}
          </div>
        </GlowCard>

        {/* dataset graphs */}
        {rows.length > 0 && (
          <div className="grid md:grid-cols-3 gap-6">
            <GlowCard>
              <ColumnHistogram
                rows={rows}
                col="orbital_period_days"
                title="period"
                unit=" d"
              />
            </GlowCard>
            <GlowCard>
              <ColumnHistogram
                rows={rows}
                col="transit_depth_ppm"
                title="depth"
                unit=" ppm"
              />
            </GlowCard>
            <GlowCard>
              <ColumnHistogram
                rows={rows}
                col="transit_duration_hours"
                title="duration"
                unit=" hr"
              />
            </GlowCard>
            <GlowCard className="md:col-span-3">
              <ScatterPlot
                rows={rows}
                xKey="orbital_period_days"
                yKey="transit_duration_hours"
                selected={selected}
                title="period vs duration"
                unitX="d"
                unitY="hr"
              />
            </GlowCard>
          </div>
        )}

        {/* preview table */}
        {rows.length > 0 && (
          <GlowCard>
            <div className="text-xs text-white/70 mb-2">
              Detected columns: {headers.length} • Rows: {rows.length} •{" "}
              <span className={haveMin ? "text-emerald-300" : "text-red-300"}>
                {haveMin ? "Ready" : "Missing required"}
              </span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full text-left whitespace-nowrap text-sm">
                <thead className="bg-white/5">
                  <tr>
                    {headers.slice(0, 8).map((h) => (
                      <th key={h} className="px-3 py-2">
                        {h}
                      </th>
                    ))}
                    <th className="px-3 py-2">…</th>
                    <th className="px-3 py-2">select</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 10).map((r, i) => (
                    <tr key={i} className="border-t border-white/5">
                      {headers.slice(0, 8).map((h) => (
                        <td key={h} className="px-3 py-2">
                          {r[h]}
                        </td>
                      ))}
                      <td className="px-3 py-2">…</td>
                      <td className="px-3 py-2">
                        <button
                          onClick={() => setSelected(i)}
                          className={`px-3 py-1 rounded-xl text-xs ${
                            selected === i
                              ? "bg-white text-black"
                              : "border border-white/20 hover:bg-white/10"
                          }`}
                        >
                          {selected === i ? "Selected" : "Choose"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length > 10 && (
                <div className="px-3 py-2 text-xs text-white/50">
                  showing first 10…
                </div>
              )}
            </div>

            <button
              onClick={analyze}
              disabled={selected == null || analyzing}
              className="mt-6 px-5 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 disabled:opacity-40 transition"
            >
              {analyzing ? "Analyzing…" : "Analyze (tabular-only)"}
            </button>
          </GlowCard>
        )}
      </div>

      {/* RIGHT: Prediction visuals */}
      <GlowCard>
        <div className="mb-4 text-sm font-medium text-white/70">Prediction</div>
        {analyzing ? (
          <AnalysisSkeleton />
        ) : output ? (
          <div className="space-y-6">
            <ProbabilityOrb 
              p={output.predicted_proba} 
              prediction={output.predicted_label}
              confidenceMessage={extras?.confidence_message}
            />
            <RadarFeatures
              items={(extras?.top_features ?? []).filter((d: any) =>
                Number.isFinite(d?.z)
              )}
            />
            <FeatureBars
              zvec={
                (featuresEcho as any) ??
                Object.fromEntries(
                  (extras?.top_features ?? []).map((d: any) => [d.name, d.z])
                )
              }
            />
            <RunHistory />
          </div>
        ) : (
          <p className="text-white/60 text-sm">
            Upload a CSV, select a row, then run Analyze.
          </p>
        )}
      </GlowCard>
    </div>
  );
}

/* -------------------------- Multimodal -------------------------- */
function MultimodalPanel() {
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [residualFile, setResidualFile] = useState<File | null>(null);
  const [pixelFile, setPixelFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [output, setOutput] = useState<ModelOutput | null>(null);
  const [extras, setExtras] = useState<any | null>(null);
  const [featuresEcho, setFeaturesEcho] = useState<Record<
    string,
    number
  > | null>(null);

  const valid = useMemo(
    () => REQ_MULTIMODAL.every((k) => headers.includes(k)),
    [headers]
  );

  const onFile = async (file: File) => {
    setError(null);
    setOutput(null);
    setExtras(null);
    setFeaturesEcho(null);
    setSelected(null);
    const text = await file.text();
    const { headers: h, rows: r } = parseCSV(text);
    setHeaders(h);
    setRows(r);
    if (!REQ_MULTIMODAL.every((k) => h.includes(k))) {
      setError(`Missing required columns: ${REQ_MULTIMODAL.join(", ")}`);
    }
  };

  const analyze = async () => {
    if (selected == null) return;
    if (!residualFile || !pixelFile) {
      setError("Attach both .npy files (residual_windows & pixel_diffs).");
      return;
    }
    setAnalyzing(true);
    const row = rows[selected];
    const fd = new FormData();
    fd.append("row", JSON.stringify(row));
    fd.append("residual_windows", residualFile);
    fd.append("pixel_diffs", pixelFile);

    try {
      const res = await fetch("/api/predict/multimodal", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error("prediction failed");
      const data = await res.json();
      setOutput({
        predicted_label: data.predicted_label,
        predicted_proba: data.predicted_proba,
      });
      setExtras(data.extras ?? null);
      setFeaturesEcho(data.debug_features ?? null);
    } catch (e: any) {
      setError(e?.message || "Prediction error.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* LEFT: Upload + table */}
      <div className="lg:col-span-2 space-y-6">
        <GlowCard>
          <div className="mb-3 text-sm font-medium text-white/70">
            Upload Multimodal CSV + .npy
          </div>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
            className="border border-dashed border-white/20 rounded-2xl p-8 text-center hover:border-white/40 transition"
          >
            <p className="text-white/70">
              CSV must include{" "}
              <code className="text-white">
                residual_windows_file,pixel_diffs_file
              </code>
            </p>
            <div className="mt-4">
              <label className="inline-block px-4 py-2 rounded-xl bg-white text-black font-medium cursor-pointer hover:opacity-90">
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFile(f);
                  }}
                />
                Choose file
              </label>
            </div>
            {error && <p className="mt-3 text-red-300 text-sm">{error}</p>}
          </div>
        </GlowCard>

        {rows.length > 0 && (
          <GlowCard>
            <div className="text-xs text-white/70 mb-2">
              Detected columns: {headers.length} • Rows: {rows.length} •{" "}
              {valid ? (
                <span className="text-emerald-300">OK</span>
              ) : (
                <span className="text-red-300">Missing required</span>
              )}
            </div>

            <div className="overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full text-left whitespace-nowrap text-sm">
                <thead className="bg-white/5">
                  <tr>
                    {headers.slice(0, 8).map((h) => (
                      <th key={h} className="px-3 py-2">
                        {h}
                      </th>
                    ))}
                    <th className="px-3 py-2">…</th>
                    <th className="px-3 py-2">select</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 10).map((r, i) => (
                    <tr key={i} className="border-t border-white/5">
                      {headers.slice(0, 8).map((h) => (
                        <td key={h} className="px-3 py-2">
                          {r[h]}
                        </td>
                      ))}
                      <td className="px-3 py-2">…</td>
                      <td className="px-3 py-2">
                        <button
                          onClick={() => setSelected(i)}
                          className={`px-3 py-1 rounded-xl text-xs ${
                            selected === i
                              ? "bg-white text-black"
                              : "border border-white/20 hover:bg-white/10"
                          }`}
                        >
                          {selected === i ? "Selected" : "Choose"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length > 10 && (
                <div className="px-3 py-2 text-xs text-white/50">
                  showing first 10…
                </div>
              )}
            </div>

            {/* .npy pickers */}
            {selected != null && (
              <div className="mt-5 grid md:grid-cols-2 gap-4">
                <label className="block">
                  <div className="text-sm text-white/70 mb-1">
                    residual_windows (.npy, ~ (5,128))
                  </div>
                  <input
                    type="file"
                    accept=".npy"
                    onChange={(e) =>
                      setResidualFile(e.target.files?.[0] ?? null)
                    }
                    className="w-full text-sm file:mr-3 file:px-3 file:py-2 file:rounded-xl file:bg-white file:text-black file:cursor-pointer"
                  />
                </label>
                <label className="block">
                  <div className="text-sm text-white/70 mb-1">
                    pixel_diffs (.npy, ~ (32,24,24))
                  </div>
                  <input
                    type="file"
                    accept=".npy"
                    onChange={(e) => setPixelFile(e.target.files?.[0] ?? null)}
                    className="w-full text-sm file:mr-3 file:px-3 file:py-2 file:rounded-xl file:bg-white file:text-black file:cursor-pointer"
                  />
                </label>
              </div>
            )}

            <button
              onClick={analyze}
              disabled={selected == null || !valid || analyzing}
              className="mt-6 px-5 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 disabled:opacity-40 transition"
            >
              {analyzing ? "Analyzing…" : "Analyze (multimodal)"}
            </button>
          </GlowCard>
        )}
      </div>

      {/* RIGHT: Prediction visuals */}
      <GlowCard>
        <div className="mb-4 text-sm font-medium text-white/70">Prediction</div>
        {analyzing ? (
          <AnalysisSkeleton />
        ) : output ? (
          <div className="space-y-6">
            <ProbabilityOrb 
              p={output.predicted_proba} 
              prediction={output.predicted_label}
              confidenceMessage={extras?.confidence_message}
            />
            <div className="text-xs text-white/60 uppercase tracking-wide">
              CNN data used:{" "}
              <span
                className={
                  extras?.used_cnn ? "text-emerald-300" : "text-white/70"
                }
              >
                {extras?.used_cnn ? "yes" : "no"}
              </span>
            </div>
            <RadarFeatures
              items={(extras?.top_features ?? []).filter((d: any) =>
                Number.isFinite(d?.z)
              )}
            />
            <FeatureBars
              zvec={
                (featuresEcho as any) ??
                Object.fromEntries(
                  (extras?.top_features ?? []).map((d: any) => [d.name, d.z])
                )
              }
            />
          </div>
        ) : (
          <p className="text-white/60 text-sm">
            Upload CSV, select a row, attach .npy files, then Analyze.
          </p>
        )}
      </GlowCard>
    </div>
  );
}
