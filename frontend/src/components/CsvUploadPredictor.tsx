import React, { useState } from 'react';
import Papa from 'papaparse';

const REQUIRED_COLUMNS = [
  'mission',
  'orbital_period_days',
  'transit_duration_hours',
  'transit_depth_ppm',
  'planet_radius_re',
  'equilibrium_temp_k',
  'insolation_flux_earth',
  'stellar_teff_k',
  'stellar_radius_re',
  'apparent_mag',
  'ra',
  'dec',
];

export default function CsvUploadPredictor() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingColumns, setMissingColumns] = useState<string[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResults([]);
      setError(null);
      // Parse header to check missing columns
      Papa.parse(selectedFile, {
        preview: 1,
        header: true,
        complete: (results) => {
          const csvColumns = results.meta.fields || [];
          const missing = REQUIRED_COLUMNS.filter(col => !csvColumns.includes(col));
          setMissingColumns(missing);
        },
      });
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('http://localhost:8000/predict/csv', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResults(data.results || []);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-white rounded shadow">
      <h2 className="text-2xl font-bold mb-4">Upload CSV for Exoplanet Prediction</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} className="mb-4" />
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
      >
        {loading ? 'Uploading...' : 'Upload & Predict'}
      </button>
      {missingColumns.length > 0 && (
        <div className="mt-4 text-yellow-700 bg-yellow-100 p-2 rounded">
          <strong>Warning:</strong> Missing columns in CSV: {missingColumns.join(', ')}<br />
          These will be filled automatically using median values.
        </div>
      )}
      {error && <div className="mt-4 text-red-600">{error}</div>}
      {results.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Results</h3>
          <table className="w-full border">
            <thead>
              <tr>
                <th className="border px-2 py-1">Row</th>
                <th className="border px-2 py-1">Label</th>
                <th className="border px-2 py-1">Confidence Score</th>
                <th className="border px-2 py-1">Confidence Level</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.row}>
                  <td className="border px-2 py-1">{r.row}</td>
                  <td className="border px-2 py-1">{r.label}</td>
                  <td className="border px-2 py-1">{r.confidence_score.toFixed(3)}</td>
                  <td className="border px-2 py-1">{r.confidence_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
