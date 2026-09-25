import React, { useState } from 'react';

const DEFAULT_DATA = [
  {
    "sample_id": "REC_001_CLEAN",
    "node_name": "Receipt_Extraction_Node",
    "action": "AUTO_ACCEPT",
    "bottleneck_perplexity": 1.05,
    "mean_e2e_perplexity": 1.04,
    "max_field_entropy": 0.0,
    "fields": [
      {"field_name": "company", "extracted_value": "SUPERMARKET METRO", "perplexity": 1.02, "semantic_entropy": 0.0},
      {"field_name": "date", "extracted_value": "2026-08-14", "perplexity": 1.05, "semantic_entropy": 0.0},
      {"field_name": "address", "extracted_value": "VIA ROMA 45, MILANO", "perplexity": 1.04, "semantic_entropy": 0.0},
      {"field_name": "total", "extracted_value": 45.90, "perplexity": 1.03, "semantic_entropy": 0.0}
    ]
  },
  {
    "sample_id": "REC_002_BLURRY_DATE",
    "node_name": "Receipt_Extraction_Node",
    "action": "NEEDS_REVIEW",
    "bottleneck_perplexity": 2.85,
    "mean_e2e_perplexity": 1.48,
    "max_field_entropy": 0.65,
    "fields": [
      {"field_name": "company", "extracted_value": "CAFE CENTRAL", "perplexity": 1.08, "semantic_entropy": 0.0},
      {"field_name": "date", "extracted_value": "2026-??-12", "perplexity": 2.85, "semantic_entropy": 0.65},
      {"field_name": "address", "extracted_value": "MAIN STREET 12", "perplexity": 1.10, "semantic_entropy": 0.0},
      {"field_name": "total", "extracted_value": 12.50, "perplexity": 1.06, "semantic_entropy": 0.0}
    ]
  },
  {
    "sample_id": "REC_003_HALLUCINATED_TOTAL",
    "node_name": "Receipt_Extraction_Node",
    "action": "REJECT",
    "bottleneck_perplexity": 5.40,
    "mean_e2e_perplexity": 2.25,
    "max_field_entropy": 1.35,
    "fields": [
      {"field_name": "company", "extracted_value": "PHARMACY PLUS", "perplexity": 1.05, "semantic_entropy": 0.0},
      {"field_name": "date", "extracted_value": "2026-09-01", "perplexity": 1.08, "semantic_entropy": 0.0},
      {"field_name": "address", "extracted_value": "7TH AVENUE", "perplexity": 1.12, "semantic_entropy": 0.0},
      {"field_name": "total", "extracted_value": 9999.99, "perplexity": 5.40, "semantic_entropy": 1.35}
    ]
  }
];

export default function App() {
  const [records, setRecords] = useState(DEFAULT_DATA);
  const [activeFilter, setActiveFilter] = useState("ALL");
  const [fileName, setFileName] = useState("Default Demo Payload");

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      if (file.name.endsWith(".json") || file.name.endsWith(".jsonl")) {
        try {
          const parsed = content.trim().startsWith("[")
            ? JSON.parse(content)
            : content.trim().split("\n").map(l => JSON.parse(l));
          setRecords(parsed);
        } catch (err) {
          alert("Invalid JSON format: " + err.message);
        }
      } else if (file.name.endsWith(".csv")) {
        parseCSV(content);
      }
    };
    reader.readAsText(file);
  };

  const parseCSV = (content) => {
    const lines = content.trim().split("\n");
    if (lines.length < 2) return;
    const sampleMap = {};
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map(r => r.trim());
      if (row.length < 8) continue;
      const sampleId = row[0];
      if (!sampleMap[sampleId]) {
        sampleMap[sampleId] = {
          sample_id: sampleId,
          node_name: row[1],
          action: row[2],
          bottleneck_perplexity: parseFloat(row[3]) || 1.0,
          mean_e2e_perplexity: parseFloat(row[4]) || 1.0,
          max_field_entropy: parseFloat(row[5]) || 0.0,
          fields: []
        };
      }
      sampleMap[sampleId].fields.push({
        field_name: row[6],
        extracted_value: row[7],
        perplexity: parseFloat(row[8]) || 1.0,
        semantic_entropy: parseFloat(row[9]) || 0.0
      });
    }
    setRecords(Object.values(sampleMap));
  };

  const totalCount = records.length;
  const acceptCount = records.filter(r => r.action === "AUTO_ACCEPT").length;
  const reviewCount = records.filter(r => r.action === "NEEDS_REVIEW").length;
  const rejectCount = records.filter(r => r.action === "REJECT").length;

  const acceptPct = totalCount ? ((acceptCount / totalCount) * 100).toFixed(1) : 0;
  const reviewPct = totalCount ? ((reviewCount / totalCount) * 100).toFixed(1) : 0;
  const rejectPct = totalCount ? ((rejectCount / totalCount) * 100).toFixed(1) : 0;

  const filteredRecords = records.filter(r => {
    if (activeFilter === "ALL") return true;
    return r.action === activeFilter;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 relative z-10">
      <div className="bg-glow"></div>

      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-1 block">
            React + Vite HITL Dashboard
          </span>
          <h1 className="text-3xl font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Multimodal Extraction Uncertainty Visualizer
          </h1>
          <p className="text-slate-400 text-sm">Interactive Field-Level Extraction Inspection</p>
        </div>

        {/* File Upload Box */}
        <div className="glass-card p-3 flex items-center gap-3">
          <label className="cursor-pointer bg-gradient-to-r from-cyan-500 to-blue-600 text-black font-bold px-4 py-2 rounded-lg text-sm hover:opacity-90 transition">
            Upload CSV / JSON
            <input type="file" accept=".json,.jsonl,.csv" onChange={handleFileUpload} className="hidden" />
          </label>
          <span className="text-xs text-slate-400 font-mono max-w-[180px] truncate">{fileName}</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="glass-card p-5 border-l-4 border-l-cyan-400">
          <div className="text-xs font-medium text-slate-400 uppercase">Total Samples</div>
          <div className="text-3xl font-black text-white mt-1">{totalCount}</div>
        </div>

        <div className="glass-card p-5 border-l-4 border-l-emerald-500">
          <div className="text-xs font-medium text-slate-400 uppercase">Auto-Accept Rate</div>
          <div className="text-3xl font-black text-emerald-400 mt-1">{acceptPct}%</div>
          <div className="text-xs text-slate-500 mt-1">{acceptCount} samples</div>
        </div>

        <div className="glass-card p-5 border-l-4 border-l-amber-500">
          <div className="text-xs font-medium text-slate-400 uppercase">Needs Review</div>
          <div className="text-3xl font-black text-amber-400 mt-1">{reviewPct}%</div>
          <div className="text-xs text-slate-500 mt-1">{reviewCount} samples</div>
        </div>

        <div className="glass-card p-5 border-l-4 border-l-red-500">
          <div className="text-xs font-medium text-slate-400 uppercase">Rejection Rate</div>
          <div className="text-3xl font-black text-red-400 mt-1">{rejectPct}%</div>
          <div className="text-xs text-slate-500 mt-1">{rejectCount} samples</div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-6 border-b border-slate-800 pb-3">
        {["ALL", "AUTO_ACCEPT", "NEEDS_REVIEW", "REJECT"].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveFilter(tab)}
            className={`px-4 py-2 text-xs font-bold rounded-lg transition ${
              activeFilter === tab
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                : 'text-slate-400 hover:text-white hover:bg-slate-800'
            }`}
          >
            {tab.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Samples Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredRecords.map((sample, idx) => {
          const isAccept = sample.action === "AUTO_ACCEPT";
          const isReview = sample.action === "NEEDS_REVIEW";
          const badgeColor = isAccept
            ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
            : isReview
            ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
            : "bg-red-500/20 text-red-400 border-red-500/40";

          return (
            <div key={idx} className="glass-card p-6 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <div className="font-mono text-xs text-slate-400">{sample.sample_id}</div>
                    <div className="text-xs text-slate-500 font-medium">{sample.node_name}</div>
                  </div>
                  <span className={`text-[10px] font-black px-2.5 py-1 rounded-full border ${badgeColor}`}>
                    {sample.action}
                  </span>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-3 mb-4 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500 block text-[10px]">BOTTLENECK PPL</span>
                    <span className="font-mono font-bold text-white">{sample.bottleneck_perplexity}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">MAX ENTROPY</span>
                    <span className="font-mono font-bold text-white">{sample.max_field_entropy}</span>
                  </div>
                </div>

                {/* Fields */}
                <div className="space-y-2.5">
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                    Field Extractions ({sample.fields ? sample.fields.length : 0})
                  </div>
                  {sample.fields && sample.fields.map((f, fIdx) => {
                    const ppl = f.perplexity || 1.0;
                    const se = f.semantic_entropy || 0.0;
                    let styleClass = "field-accept";
                    let statusText = "High Confidence";

                    if (ppl > 4.4 || se > 1.2) {
                      styleClass = "field-reject";
                      statusText = "Severe Risk";
                    } else if (ppl > 2.2 || se > 0.5) {
                      styleClass = "field-review";
                      statusText = "Review Needed";
                    }

                    return (
                      <div key={fIdx} className={`p-3 rounded-lg text-xs transition ${styleClass}`}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-mono font-bold text-slate-200">{f.field_name}</span>
                          <span className="text-[9px] uppercase font-extrabold opacity-80">{statusText}</span>
                        </div>
                        <div className="font-mono text-white text-sm font-semibold mb-1.5 truncate">
                          {String(f.extracted_value)}
                        </div>
                        <div className="flex gap-3 text-[10px] text-slate-400 font-mono">
                          <span>PPL: <strong className="text-white">{ppl}</strong></span>
                          <span>SE: <strong class="text-white">{se}</strong></span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
