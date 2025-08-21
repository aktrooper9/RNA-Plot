#!/usr/bin/env python3
"""
Updated app per request:
- Volcano plot first (primary focus)
- Removed Venn; added UpSet plot for overlaps
- Dynamic list of overlapping genes with selectable sets and mode (Exact / At least these)

Notes:
- Requires: dash, pandas, numpy, matplotlib, plotly, upsetplot
- Files: expects the same 4 CSVs as before in FILES
"""

from __future__ import annotations
import base64, io
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Output, Input, State
from dash import dash_table

# Headless matplotlib for static images (UpSet)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UpSet for set visualization
from upsetplot import UpSet, from_contents  # pip install upsetplot

# Plotly for volcano
import plotly.express as px

# ---------- CONFIG ----------
FILES  = ["29081.csv", "115828.csv", "135092.csv", "99248.csv"]
LABELS = [Path(f).stem for f in FILES]
DEFAULT_PADJ = 0.05
DEFAULT_LFC  = 1.0
# ---------------------------

def load_deg_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    # Try headered
    df = pd.read_csv(p, dtype={0: str}, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    need = ["gene_id", "log2foldchange", "padj"]
    if all(k in cols for k in need):
        gid, lfc, padj = cols["gene_id"], cols["log2foldchange"], cols["padj"]
        df = df[[gid, lfc, padj]].rename(columns={gid: "gene_id", lfc: "log2FoldChange", padj: "padj"})
    else:
        # Fallback: first 3 columns (no header)
        df = pd.read_csv(p, header=None, usecols=[0, 1, 2], low_memory=False)
        df.columns = ["gene_id", "log2FoldChange", "padj"]

    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gene_id", "log2FoldChange", "padj"])

def sanitize_padj(x) -> float:
    if x is None or not np.isfinite(x):
        return DEFAULT_PADJ
    return float(min(max(x, 0.0), 1.0))

def sanitize_lfc(x) -> float:
    if x is None or not np.isfinite(x):
        return DEFAULT_LFC
    return float(abs(x))  # threshold non-negative

def filter_gene_set(df: pd.DataFrame, padj_thresh: float, lfc_thresh: float, regulation: str) -> Set[str]:
    padj_thresh = sanitize_padj(padj_thresh)
    lfc_thresh  = sanitize_lfc(lfc_thresh)
    sig = df["padj"] <= padj_thresh
    if regulation == "up":
        fc = df["log2FoldChange"] >= lfc_thresh
    elif regulation == "down":
        fc = df["log2FoldChange"] <= -lfc_thresh
    else:  # both
        fc = np.abs(df["log2FoldChange"]) >= lfc_thresh
    return set(df[sig & fc]["gene_id"])

def volcano_dataframe(df: pd.DataFrame, padj_thresh: float, lfc_thresh: float) -> pd.DataFrame:
    """Prepare a DF with -log10(padj) and significance class for volcano."""
    padj_thresh = sanitize_padj(padj_thresh)
    lfc_thresh  = sanitize_lfc(lfc_thresh)
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2FoldChange", "padj"])
    out["padj"] = out["padj"].clip(lower=np.finfo(float).tiny, upper=1.0)
    out["neglog10_padj"] = -np.log10(out["padj"])
    # classify
    up = (out["padj"] <= padj_thresh) & (out["log2FoldChange"] >= lfc_thresh)
    down = (out["padj"] <= padj_thresh) & (out["log2FoldChange"] <= -lfc_thresh)
    out["sig"] = np.where(up, "Up", np.where(down, "Down", "NS"))
    return out

def render_upset_png(named_sets: dict) -> str:
    """Render an UpSet plot and return a base64 data URI.
    Bugfix: ensure we save the SAME figure that UpSet draws into by passing
    fig=... to .plot() (otherwise we may save an empty figure). """
    total = sum(len(s) for s in named_sets.values())
    if total == 0:
        fig = plt.figure(figsize=(9, 5.5), dpi=150)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No genes pass the filters.", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        series = from_contents(named_sets)
        fig = plt.figure(figsize=(9, 5.5), dpi=150)
        up = UpSet(series, show_counts=True, show_percentages=False, sort_by="cardinality")
        # IMPORTANT: plot INTO this figure so we save the correct image
        up.plot(fig=fig)
        fig.suptitle("UpSet: Overlaps across datasets", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# Load data once
DFS = [load_deg_csv(f) for f in FILES]

# -------------- Dash UI --------------
app = dash.Dash(__name__)
app.title = "DEG Volcano + UpSet (4 sets, numeric thresholds)"
server = app.server  # for WSGI deployments

def threshold_block(i: int, label: str) -> html.Div:
    return html.Div(
        [
            html.H4(label, style={"marginBottom": "6px"}),
            html.Div([
                html.Label("Adjusted p-value (padj)", style={"marginRight": "8px"}),
                dcc.Input(
                    id=f"padj-{i}", type="number",
                    min=0, max=1, step=0.001, value=DEFAULT_PADJ,
                    style={"width": "120px"}
                ),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("Log2 Fold Change", style={"marginRight": "8px"}),
                dcc.Input(
                    id=f"lfc-{i}", type="number",
                    min=0, step=0.01, value=DEFAULT_LFC,
                    style={"width": "120px"}
                ),
            ]),
        ],
        style={"flex": "1 1 260px", "minWidth": "240px", "padding": "10px", "border": "1px solid #eee", "borderRadius": "8px"}
    )

app.layout = html.Div(
    [
        html.H2("Volcano + UpSet for DEGs (per-dataset numeric thresholds)"),

        # Hidden store to hold optional gene annotations
        dcc.Store(id="annot-store"),
        # Store enrichment results (CSV)
        dcc.Store(id="enrich-store"),

        # Thresholds
        html.Div(
            [threshold_block(0, LABELS[0]), threshold_block(1, LABELS[1]),
             threshold_block(2, LABELS[2]), threshold_block(3, LABELS[3])],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "margin": "12px 16px"},
        ),

        # Regulation (affects set membership for UpSet & overlap list)
        html.Div(
            [
                html.Label("Regulation (applies to UpSet & overlap list)"),
                dcc.Dropdown(
                    id="regulation",
                    options=[
                        {"label": "Both (|log2FC| ≥ threshold)", "value": "both"},
                        {"label": "Up (log2FC ≥ threshold)", "value": "up"},
                        {"label": "Down (log2FC ≤ -threshold)", "value": "down"},
                    ],
                    value="both", clearable=False, style={"width": "420px"},
                ),
            ],
            style={"margin": "12px 16px"},
        ),

        # --- VOLCANO FIRST ---
        html.H3("Volcano plot"),
        html.Div(
            [
                html.Label("Dataset"),
                dcc.Dropdown(
                    id="volcano-set",
                    options=[{"label": lbl, "value": i} for i, lbl in enumerate(LABELS)],
                    value=0,
                    clearable=False,
                    style={"width": "320px"},
                ),
                html.Button("Download volcano CSV", id="dl-volcano-btn", style={"marginLeft": "12px"}),
                dcc.Download(id="dl-volcano"),
            ],
            style={"margin": "12px 16px"},
        ),
        dcc.Loading(dcc.Graph(id="volcano-graph", style={"height": "620px"}), type="default"),

        html.Hr(),

        # --- UPSET PLOT (replaces Venn) ---
        html.H3("UpSet plot (overlaps across datasets)"),
        dcc.Loading(html.Img(id="upset-plot", style={"border": "1px solid #ddd", "maxWidth": "960px"}), type="default"),

        html.Hr(),

        # --- DYNAMIC OVERLAP LIST ---
        html.H3("Dynamic list of overlapping genes"),
        html.Div(
            [
                html.Label("Choose datasets to intersect"),
                dcc.Checklist(
                    id="overlap-sets",
                    options=[{"label": lbl, "value": i} for i, lbl in enumerate(LABELS)],
                    value=[0, 1],  # sensible default: first two
                    labelStyle={"display": "inline-block", "marginRight": "14px"},
                ),
                dcc.RadioItems(
                    id="overlap-mode",
                    options=[
                        {"label": "At least these (⊇)", "value": "atleast"},
                        {"label": "Exact match (≡)", "value": "exact"},
                    ],
                    value="atleast",
                    labelStyle={"display": "inline-block", "marginRight": "14px"},
                    style={"marginTop": "8px"}
                ),
                html.Div([html.Strong("Count:"), html.Span(id="overlap-count", style={"marginLeft": "6px"})], style={"marginTop": "8px"}),
                html.Button("Download overlap CSV", id="dl-overlap-btn", style={"marginTop": "8px"}),
                dcc.Download(id="dl-overlap"),
            ],
            style={"margin": "12px 16px"},
        ),
        # --- OPTIONAL ANNOTATION UPLOAD ---
        html.Details([
            html.Summary("Gene annotation (optional): upload CSV/TSV with columns gene_id, gene_symbol, gene_function"),
            html.Div([
                dcc.Upload(
                    id="annot-upload",
                    children=html.Div(["Drag and drop or ", html.A("select a CSV/TSV mapping file")]),
                    multiple=False,
                    style={
                        "width": "100%", "height": "60px", "lineHeight": "60px",
                        "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "6px",
                        "textAlign": "center", "margin": "10px 0"
                    },
                ),
                html.Div(
                    "Accepted columns (flexible names): gene_id ∼ {gene_id,gene,geneid,id,symbol}, "
                    "gene_symbol ∼ {symbol,gene_symbol}, gene_function ∼ {gene_function,description,gene_name}",
                    style={"color": "#666", "fontSize": "12px", "marginBottom": "8px"}
                ),
                html.Div(id="annot-status", style={"fontSize": "12px"}),
            ], style={"margin": "0 16px 12px"})
        ]),

        dash_table.DataTable(
            id="overlap-table",
            columns=[
                {"name": "gene_id", "id": "gene_id"},
                {"name": "gene_symbol", "id": "gene_symbol"},
                {"name": "gene_function", "id": "gene_function"},
            ],
            page_size=15,
            sort_action="native",
            filter_action="native",
            style_table={"maxHeight": "340px", "overflowY": "auto", "border": "1px solid #eee"},
            style_cell={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "fontSize": 12, "padding": "6px"},
        ),


        # Enrichment controls (GO/KEGG) for current overlap genes
        html.H3("Enrichment disabled", style={"margin": "16px 16px 6px", "color": "#666"}),
        html.Div("This build has enrichment commented out per request.", style={"margin": "0 16px 12px", "color": "#666"}),
html.Hr(),

        # Four per-dataset downloads (unchanged)
        html.Div(
            [
                html.Button(f"Download {LABELS[0]} (CSV)", id="dl-a-btn"),
                dcc.Download(id="dl-a"),
                html.Button(f"Download {LABELS[1]} (CSV)", id="dl-b-btn", style={"marginLeft": "12px"}),
                dcc.Download(id="dl-b"),
                html.Button(f"Download {LABELS[2]} (CSV)", id="dl-c-btn", style={"marginLeft": "12px"}),
                dcc.Download(id="dl-c"),
                html.Button(f"Download {LABELS[3]} (CSV)", id="dl-d-btn", style={"marginLeft": "12px"}),
                dcc.Download(id="dl-d"),
            ],
            style={"margin": "14px 16px"},
        ),

        html.Div(
            "Tip: Volcano coloring is based on the selected dataset's numeric thresholds. Regulation affects UpSet & the overlap list.",
            style={"margin": "8px 16px", "color": "#666", "fontSize": "12px"},
        ),
    ],
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "padding": "12px"},
)

# ---- Volcano figure ----
@app.callback(
    Output("volcano-graph", "figure"),
    Input("volcano-set", "value"),
    Input("padj-0", "value"), Input("lfc-0", "value"),
    Input("padj-1", "value"), Input("lfc-1", "value"),
    Input("padj-2", "value"), Input("lfc-2", "value"),
    Input("padj-3", "value"), Input("lfc-3", "value"),
)
def update_volcano(idx, p0, l0, p1, l1, p2, l2, p3, l3):
    idx = int(idx or 0)
    padjs = [sanitize_padj(p0), sanitize_padj(p1), sanitize_padj(p2), sanitize_padj(p3)]
    lfcs  = [sanitize_lfc(l0),  sanitize_lfc(l1),  sanitize_lfc(l2),  sanitize_lfc(l3)]
    df = volcano_dataframe(DFS[idx], padjs[idx], lfcs[idx])

    # Counts for title
    total_n = int(len(df))
    up_n = int((df["sig"] == "Up").sum())
    down_n = int((df["sig"] == "Down").sum())

    title = (
        f"Volcano: {LABELS[idx]} (padj ≤ {padjs[idx]:.3g}, |log2FC| ≥ {lfcs[idx]:.2f}) — "
        f"N={total_n}, Up={up_n}, Down={down_n}"
    )

    fig = px.scatter(
        df, x="log2FoldChange", y="neglog10_padj", color="sig",
        color_discrete_map={"Up": "#D55E00", "Down": "#0072B2", "NS": "#BBBBBB"},
        hover_data={"gene_id": True, "padj": True, "log2FoldChange": True, "neglog10_padj": False, "sig": True},
        labels={"log2FoldChange": "log2 Fold Change", "neglog10_padj": "-log10(padj)", "sig": "Class"},
        title=title, opacity=0.7, render_mode="webgl"
    )
    # Threshold lines
    fig.add_hline(y=-np.log10(max(padjs[idx], np.finfo(float).tiny)), line_dash="dash", line_color="gray")
    fig.add_vline(x= lfcs[idx], line_dash="dash", line_color="gray")
    fig.add_vline(x=-lfcs[idx], line_dash="dash", line_color="gray")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12), margin=dict(l=40, r=20, t=60, b=80))
    return fig

# ---- UpSet plot (replaces Venn) ----
@app.callback(
    Output("upset-plot", "src"),
    Input("padj-0", "value"), Input("lfc-0", "value"),
    Input("padj-1", "value"), Input("lfc-1", "value"),
    Input("padj-2", "value"), Input("lfc-2", "value"),
    Input("padj-3", "value"), Input("lfc-3", "value"),
    Input("regulation", "value"),
)
def update_upset(p0, l0, p1, l1, p2, l2, p3, l3, regulation):
    thresholds = [(sanitize_padj(p0), sanitize_lfc(l0)),
                  (sanitize_padj(p1), sanitize_lfc(l1)),
                  (sanitize_padj(p2), sanitize_lfc(l2)),
                  (sanitize_padj(p3), sanitize_lfc(l3))]
    gene_sets = [filter_gene_set(df, padj, lfc, regulation) for df, (padj, lfc) in zip(DFS, thresholds)]
    named = {label: s for label, s in zip(LABELS, gene_sets)}
    return render_upset_png(named)

# ---- Overlap list (dynamic) ----
@app.callback(
    Output("overlap-table", "data"),
    Output("overlap-count", "children"),
    Input("overlap-sets", "value"),
    Input("overlap-mode", "value"),
    Input("padj-0", "value"), Input("lfc-0", "value"),
    Input("padj-1", "value"), Input("lfc-1", "value"),
    Input("padj-2", "value"), Input("lfc-2", "value"),
    Input("padj-3", "value"), Input("lfc-3", "value"),
    Input("regulation", "value"),
    Input("annot-store", "data"),
)
def update_overlap_table(selected_sets, mode, p0, l0, p1, l1, p2, l2, p3, l3, regulation, annot_data):
    selected = set(int(i) for i in (selected_sets or []))
    padjs = [sanitize_padj(p0), sanitize_padj(p1), sanitize_padj(p2), sanitize_padj(p3)]
    lfcs  = [sanitize_lfc(l0),  sanitize_lfc(l1),  sanitize_lfc(l2),  sanitize_lfc(l3)]
    sets = [filter_gene_set(df, padj, lfc, regulation) for df, padj, lfc in zip(DFS, padjs, lfcs)]

    if not selected:
        return [], "Select one or more datasets"

    # Compute intersection for selected sets
    inter = set.intersection(*(sets[i] for i in selected)) if selected else set()
    if mode == "exact":
        # Remove genes that appear in any unselected set
        others = set.union(*(sets[i] for i in range(4) if i not in selected)) if len(selected) < 4 else set()
        inter = inter - others

    genes = sorted(inter)
    base_df = pd.DataFrame({"gene_id": [str(g) for g in genes]})

    # Merge in annotations if provided
    if annot_data:
        try:
            annot_df = pd.DataFrame(annot_data)
            base_df = base_df.merge(annot_df, on="gene_id", how="left")
        except Exception:
            pass

    for col in ["gene_symbol", "gene_function"]:
        if col not in base_df.columns:
            base_df[col] = ""

    data = base_df.to_dict("records")
    return data, f"{len(genes)} genes"

# ---- Overlap CSV download ----
@app.callback(
    Output("dl-overlap", "data"),
    Input("dl-overlap-btn", "n_clicks"),
    State("overlap-sets", "value"),
    State("overlap-mode", "value"),
    State("padj-0", "value"), State("lfc-0", "value"),
    State("padj-1", "value"), State("lfc-1", "value"),
    State("padj-2", "value"), State("lfc-2", "value"),
    State("padj-3", "value"), State("lfc-3", "value"),
    State("regulation", "value"),
    State("annot-store", "data"),
    prevent_initial_call=True,
)
def download_overlap(n, selected_sets, mode, p0, l0, p1, l1, p2, l2, p3, l3, regulation, annot_data):
    selected = set(int(i) for i in (selected_sets or []))
    padjs = [sanitize_padj(p0), sanitize_padj(p1), sanitize_padj(p2), sanitize_padj(p3)]
    lfcs  = [sanitize_lfc(l0),  sanitize_lfc(l1),  sanitize_lfc(l2),  sanitize_lfc(l3)]
    sets = [filter_gene_set(df, padj, lfc, regulation) for df, padj, lfc in zip(DFS, padjs, lfcs)]

    if not selected:
        return dcc.send_data_frame(pd.DataFrame({"gene_id": []}).to_csv, "overlap_genes.csv", index=False)

    inter = set.intersection(*(sets[i] for i in selected)) if selected else set()
    if mode == "exact":
        others = set.union(*(sets[i] for i in range(4) if i not in selected)) if len(selected) < 4 else set()
        inter = inter - others

    df = pd.DataFrame({"gene_id": [str(g) for g in sorted(inter)]})
    # Merge annotations if present
    if annot_data:
        try:
            annot_df = pd.DataFrame(annot_data)
            df = df.merge(annot_df, on="gene_id", how="left")
        except Exception:
            pass
    name_bits = [LABELS[i] for i in sorted(selected)]
    mode_tag = "exact" if mode == "exact" else "atleast"
    fname = f"overlap_{'__'.join(name_bits)}_{mode_tag}.csv"
    cols = [c for c in ["gene_id", "gene_symbol", "gene_function"] if c in df.columns] + [c for c in df.columns if c not in {"gene_id", "gene_symbol", "gene_function"}]
    return dcc.send_data_frame(df[cols].to_csv, fname, index=False)

# ---- Annotation upload handling ----


def _strip_ver(x):
    if isinstance(x, str) and x.upper().startswith(('ENSG','ENST','ENSMUSG','ENSMUST')) and '.' in x:
        return x.split('.',1)[0]
    return x
def _normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: gene_id, gene_symbol, gene_function (where available)."""
    lower = {c.lower(): c for c in df.columns}
    gid_opts = ["ensemblgeneid", "ensembl_gene_id", "gene_id", "gene", "geneid", "id", "symbol"]
    sym_opts = ["gene_symbol", "symbol", "hgnc_symbol"]
    fun_opts = ["gene_function", "function", "description", "gene_name", "name", "product"]

    def pick(opts):
        for k in opts:
            if k in lower:
                return lower[k]
        return None

    gid = pick(gid_opts)
    sym = pick(sym_opts)
    fun = pick(fun_opts)

    out = pd.DataFrame()
    if gid is None:
        return out
    out["gene_id"] = df[gid].astype(str).str.strip()
    if sym:
        out["gene_symbol"] = df[sym].astype(str).str.strip()
    if fun:
        out["gene_function"] = df[fun].astype(str).str.strip()
    return out.drop_duplicates(subset=["gene_id"])

@app.callback(
    Output("annot-store", "data"),
    Output("annot-status", "children"),
    Input("annot-upload", "contents"),
    State("annot-upload", "filename"),
    prevent_initial_call=True,
)

def handle_annot_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        text = decoded.decode('utf-8', errors='ignore')
        from io import StringIO
        sep = '\t' if (filename and filename.lower().endswith('.tsv')) or (text.count('    ') > text.count(',')) else ','
        df_raw = pd.read_csv(StringIO(text), sep=sep)
        df_norm = _normalize_annotation_columns(df_raw)
        if df_norm.empty:
            return dash.no_update, html.Span("Uploaded file did not contain a recognizable gene_id column.", style={"color": "#B00020"})
        return df_norm.to_dict("records"), html.Span(f"Loaded annotation for {len(df_norm)} unique gene IDs from {filename}.")
    except Exception as e:
        return dash.no_update, html.Span(f"Failed to parse {filename}: {e}", style={"color": "#B00020"})

# ---- Volcano CSV download (selected dataset) ----
@app.callback(
    Output("dl-volcano", "data"),
    Input("dl-volcano-btn", "n_clicks"),
    State("volcano-set", "value"),
    State("padj-0", "value"), State("lfc-0", "value"),
    State("padj-1", "value"), State("lfc-1", "value"),
    State("padj-2", "value"), State("lfc-2", "value"),
    State("padj-3", "value"), State("lfc-3", "value"),
    prevent_initial_call=True,
)
def download_volcano(n, idx, p0, l0, p1, l1, p2, l2, p3, l3):
    idx = int(idx or 0)
    padjs = [sanitize_padj(p0), sanitize_padj(p1), sanitize_padj(p2), sanitize_padj(p3)]
    lfcs  = [sanitize_lfc(l0),  sanitize_lfc(l1),  sanitize_lfc(l2),  sanitize_lfc(l3)]
    df = volcano_dataframe(DFS[idx], padjs[idx], lfcs[idx])
    out = df[["gene_id", "log2FoldChange", "padj", "neglog10_padj", "sig"]].copy()
    return dcc.send_data_frame(out.to_csv, f"{LABELS[idx]}_volcano.csv", index=False)

# ---- Four per-dataset filtered-set downloads (unchanged contents) ----
@app.callback(
    Output("dl-a", "data"),
    Input("dl-a-btn", "n_clicks"),
    State("padj-0", "value"), State("lfc-0", "value"),
    State("regulation", "value"),
    prevent_initial_call=True,
)
def download_a(n, padj, lfc, reg):
    s = filter_gene_set(DFS[0], padj, lfc, reg)
    df = pd.DataFrame({"gene_id": sorted(s)})
    return dcc.send_data_frame(df.to_csv, f"{LABELS[0]}_filtered.csv", index=False)

@app.callback(
    Output("dl-b", "data"),
    Input("dl-b-btn", "n_clicks"),
    State("padj-1", "value"), State("lfc-1", "value"),
    State("regulation", "value"),
    prevent_initial_call=True,
)
def download_b(n, padj, lfc, reg):
    s = filter_gene_set(DFS[1], padj, lfc, reg)
    df = pd.DataFrame({"gene_id": sorted(s)})
    return dcc.send_data_frame(df.to_csv, f"{LABELS[1]}_filtered.csv", index=False)

@app.callback(
    Output("dl-c", "data"),
    Input("dl-c-btn", "n_clicks"),
    State("padj-2", "value"), State("lfc-2", "value"),
    State("regulation", "value"),
    prevent_initial_call=True,
)
def download_c(n, padj, lfc, reg):
    s = filter_gene_set(DFS[2], padj, lfc, reg)
    df = pd.DataFrame({"gene_id": sorted(s)})
    return dcc.send_data_frame(df.to_csv, f"{LABELS[2]}_filtered.csv", index=False)

@app.callback(
    Output("dl-d", "data"),
    Input("dl-d-btn", "n_clicks"),
    State("padj-3", "value"), State("lfc-3", "value"),
    State("regulation", "value"),
    prevent_initial_call=True,
)
def download_d(n, padj, lfc, reg):
    s = filter_gene_set(DFS[3], padj, lfc, reg)
    df = pd.DataFrame({"gene_id": sorted(s)})
    return dcc.send_data_frame(df.to_csv, f"{LABELS[3]}_filtered.csv", index=False)


# ---- Enrichment (overlap genes -> Enrichr via gseapy) ----
"""
@app.callback(
    Output("enrich-table", "data"),
    Output("enrich-table", "columns"),
    Output("enrich-bar", "figure"),
    Output("enrich-status", "children"),
    Output("enrich-store", "data"),
    Input("enrich-run", "n_clicks"),
    State("overlap-sets", "value"),
    State("overlap-mode", "value"),
    State("padj-0", "value"), State("lfc-0", "value"),
    State("padj-1", "value"), State("lfc-1", "value"),
    State("padj-2", "value"), State("lfc-2", "value"),
    State("padj-3", "value"), State("lfc-3", "value"),
    State("regulation", "value"),
    State("enrich-libraries", "value"),
    State("enrich-species", "value"),
    State("enrich-topn", "value"),
    prevent_initial_call=True,
)
def run_enrichment(n_clicks, selected_sets, mode, p0, l0, p1, l1, p2, l2, p3, l3, regulation, libraries, species, topn):
    import pandas as pd, numpy as np
    from plotly import graph_objs as go
    # compute overlap genes (same logic as update_overlap_table)
    selected = set(int(i) for i in (selected_sets or []))
    padjs = [sanitize_padj(p0), sanitize_padj(p1), sanitize_padj(p2), sanitize_padj(p3)]
    lfcs  = [sanitize_lfc(l0),  sanitize_lfc(l1),  sanitize_lfc(l2),  sanitize_lfc(l3)]
    sets = [filter_gene_set(df, padj, lfc, regulation) for df, padj, lfc in zip(DFS, padjs, lfcs)]
    if not selected:
        return [], [], go.Figure(), "Select one or more datasets.", None
    inter = set.intersection(*(sets[i] for i in selected)) if selected else set()
    if mode == "exact":
        others = set.union(*(sets[i] for i in range(4) if i not in selected)) if len(selected) < 4 else set()
        inter = inter - others
    genes = sorted(inter)
    if len(genes) == 0:
        return [], [], go.Figure(), "No overlapping genes under current thresholds.", None
    if not libraries:
        return [], [], go.Figure(), "Choose at least one library.", None
    # try gseapy dynamically
    try:
        import gseapy as gp
    except Exception as e:
        return [], [], go.Figure(), f"gseapy not available: {e}", None

    # Run enrichment per library
    frames = []
    status_msgs = []
    for lib in libraries:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism=species, no_plot=True, cutoff=1.0)
            df = enr.results.copy()
            df["library"] = lib
            # standardize columns
            rename = {"Term":"term","Adjusted P-value":"padj","P-value":"pval","Overlap":"overlap","Odds Ratio":"odds_ratio","Combined Score":"combined_score","Genes":"genes"}
            for k,v in rename.items():
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            keep = [c for c in ["library","term","padj","pval","overlap","odds_ratio","combined_score","genes"] if c in df.columns]
            df = df[keep].sort_values(["library","padj","pval"], ascending=[True, True, True])
            frames.append(df)
            status_msgs.append(f"{lib}: {len(df)} terms")
        except Exception as e:
            status_msgs.append(f"{lib}: ERROR {e}")
    if not frames:
        return [], [], go.Figure(), "; ".join(status_msgs), None
    res = pd.concat(frames, ignore_index=True)

    # Build table
    cols = [{"name": c, "id": c} for c in res.columns]
    data = res.to_dict("records")

    # Build bar figure: top N per library by -log10(padj)
    if "padj" in res.columns and "term" in res.columns:
        res_plot = res.copy()
        res_plot["neglog10_padj"] = -np.log10(res_plot["padj"].replace(0, np.nextafter(0, 1)))
        res_plot = res_plot.sort_values(["library","padj"]).groupby("library", as_index=False).head(int(topn or 20))
        fig = px.bar(res_plot, x="neglog10_padj", y="term", orientation="h", facet_row="library", height=min(1200, 280 + 220*max(1, res_plot['library'].nunique())), labels={"neglog10_padj":"-log10(adj p)","term":"Term"})
        fig.update_yaxes(matches=None, automargin=True)
        fig.update_layout(barmode="group", showlegend=False, margin=dict(l=180, r=20, t=40, b=40))
    else:
        from plotly import graph_objs as go
        fig = go.Figure()

    status = f"Enrichment done on {len(genes)} overlap genes. " + " | ".join(status_msgs)
    # Store CSV for download
    csv_buf = res.to_csv(index=False)
    return data, cols, fig, status, csv_buf

# Enrichment CSV download

"""
"""
@app.callback(
    Output("enrich-dl", "data"),
    Input("enrich-dl-btn", "n_clicks"),
    State("enrich-store", "data"),
    prevent_initial_call=True,
)
def download_enrich(n, csv_buf):
    if not csv_buf:
        return dash.no_update
    return dict(content=csv_buf, filename="enrichment_results.csv", type="text/csv")

if __name__ == "__main__":
    app.run(debug=True)

"""
