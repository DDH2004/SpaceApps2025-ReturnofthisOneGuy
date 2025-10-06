// app/api/predict/tabular/route.ts
import { NextResponse } from "next/server";

export const runtime = "nodejs";

// Default values matching backend FEATURE_DEFAULTS
const FEATURE_DEFAULTS = {
  planet_radius_re: 1.0,
  equilibrium_temp_k: 500.0,
  insolation_flux_earth: 1.0,
  stellar_teff_k: 5500.0,
  stellar_radius_re: 1.0,
  apparent_mag: 12.0,
  ra: 0.0,
  dec: 0.0
};

// Helper function to safely parse numbers with defaults
const parseNumber = (value: any, defaultValue: number = 0): number => {
  if (value === null || value === undefined || value === '') return defaultValue;
  const num = Number(value);
  return isNaN(num) ? defaultValue : num;
};

// Map CSV row to backend expected format
function mapRowToBackendFormat(row: Record<string, any>) {
  return {
    mission: row.mission || row.Mission || "KEPLER",
    orbital_period_days: parseNumber(row.orbital_period_days || row.koi_period, 365.25),
    transit_duration_hours: parseNumber(row.transit_duration_hours || row.koi_duration, 6.5),
    transit_depth_ppm: parseNumber(row.transit_depth_ppm || row.koi_depth, 1000),
    planet_radius_re: parseNumber(row.planet_radius_re || row.koi_prad, FEATURE_DEFAULTS.planet_radius_re),
    equilibrium_temp_k: parseNumber(row.equilibrium_temp_k || row.koi_teq, FEATURE_DEFAULTS.equilibrium_temp_k),
    insolation_flux_earth: parseNumber(row.insolation_flux_earth || row.koi_insol, FEATURE_DEFAULTS.insolation_flux_earth),
    stellar_teff_k: parseNumber(row.stellar_teff_k || row.koi_steff, FEATURE_DEFAULTS.stellar_teff_k),
    stellar_radius_re: parseNumber(row.stellar_radius_re || row.koi_srad, FEATURE_DEFAULTS.stellar_radius_re),
    apparent_mag: parseNumber(row.apparent_mag || row.koi_kepmag, FEATURE_DEFAULTS.apparent_mag),
    ra: parseNumber(row.ra || row.ra_str, FEATURE_DEFAULTS.ra),
    dec: parseNumber(row.dec || row.dec_str, FEATURE_DEFAULTS.dec)
  };
}

export async function POST(req: Request) {
  try {
    const { row } = await req.json();
    if (!row) {
      return NextResponse.json({ error: "Missing row" }, { status: 400 });
    }

    // Map the row to the expected backend format
    const mappedRow = mapRowToBackendFormat(row);
    
    console.log("Original row:", row);
    console.log("Mapped row for backend:", mappedRow);

    // Forward the mapped row to FastAPI backend
    const backendURL = process.env.BACKEND_URL ?? "http://localhost:8000";
    const resp = await fetch(`${backendURL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mappedRow),
    });

    if (!resp.ok) {
      const err = await resp.text();
      console.error("Backend error:", err);
      return NextResponse.json({ error: `Backend error: ${err}` }, { status: resp.status });
    }

    const data = await resp.json();
    console.log("Backend response:", data);
    
    // Transform backend response to match frontend expectations
    return NextResponse.json({
      predicted_label: data.prediction,
      predicted_proba: data.probability,
      debug_features: {}, // Add if available from backend
      extras: {
        confidence_level: data.confidence_level,
        processing_time_ms: data.processing_time_ms,
        top_features: [] // Add if available from backend
      }
    });
  } catch (e: any) {
    console.error("API error:", e);
    return NextResponse.json(
      { error: e?.message ?? "Server error" },
      { status: 500 }
    );
  }
}
