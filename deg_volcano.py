
#!/usr/bin/env python3
"""
DEG Volcano + UpSet (patched)
- Volcano counts UNIQUE, non-null gene_ids (so it matches UpSet, which uses set membership).
- UpSet built from per-dataset filtered gene_id sets.
- Overlap list supports "Exact" and "At least these" modes.
- Gene annotation upload: flexible CSV/TSV with common headers (GeneID, Symbol, Description, etc.).
  Auto-normalized and merged into overlap list / download.

Requirements:
  pip install dash pandas numpy plotly matplotlib upsetplot

Files expected in working dir:
  29081.csv  115828.csv  135092.csv  99248.csv
(Each with columns: gene_id, log2FoldChange (or Log2foldchange), padj. Headered or first 3 columns.)
"""

from __future__ import annotations
import base64, io, os
from pathlib import Path
from typing import List, Set, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_contents

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State

import plotly.express as px

# ---------- CONFIG ----------
FILES  = ["29081.csv", "115828.csv", "135092.csv", "99248.csv"]
LABELS = [Path(f).stem for f in FILES]
DEFAULT_PADJ = 0.05
DEFAULT_LFC  = 1.0
# ---------------------------

def load_deg_csv(path: str | Path) -> pd.DataFrame:
    """Load a DEG CSV with columns gene_id, log2FoldChange, padj.
    Accepts case-insensitive headers; falls back to first 3 columns if needed.
    """
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
    # Normalize types
    df["gene_id"] = df["gene_id"].astype(str)
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
    return df

def sanitize_lfc(x) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return DEFAULT_LFC
    try:
        v = float(x)
        return abs(v)  # threshold non-negative
    except Exception:
        return DEFAULT_LFC

def sanitize_padj(x) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return DEFAULT_PADJ
    try:
        v = float(x)
        return float(min(max(v, 0.0), 1.0))
    except Exception:
        return DEFAULT_PADJ

def volcano_dataframe(df: pd.DataFrame, padj_thresh: float, lfc_thresh: float) -> pd.DataFrame:
    """Prepare a DF with -log10(padj) and significance class for volcano."""
    padj_thresh = sanitize_padj(padj_thresh)
    lfc_thresh  = sanitize_lfc(lfc_thresh)
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2FoldChange", "padj"]).copy()
    out["padj"] = out["padj"].clip(lower=np.finfo(float).tiny, upper=1.0)
    out["neglog10_padj"] = -np.log10(out["padj"])
    # classify
    up = (out["padj"] <= padj_thresh) & (out["log2FoldChange"] >= lfc_thresh)
    down = (out["padj"] <= padj_thresh) & (out["log2FoldChange"] <= -lfc_thresh)
    out["sig"] = np.where(up, "Up", np.where(down, "Down", "NS"))
    return out

def unique_counts_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to one row per unique non-null gene_id, preferring Up/Down over NS when duplicates occur."""
    tmp = df[df["gene_id"].notna()].copy()
    tmp["gene_id"] = tmp["gene_id"].astype(str).str.strip()
    # Order of preference for duplicates: Up > Down > NS
    cat = pd.Categorical(tmp["sig"], categories=["Up", "Down", "NS"], ordered=True)
    tmp = tmp.assign(_sig_cat=cat)
    tmp = tmp.sort_values(by=["gene_id", "_sig_cat"], ascending=[True, True])
    dedup = tmp.drop_duplicates(subset=["gene_id"], keep="first").drop(columns=["_sig_cat"])
    return dedup

def filter_gene_set(df: pd.DataFrame, padj_thresh: float, lfc_thresh: float, regulation: str) -> Set[str]:
    """Return a set of gene_ids matching the thresholds and regulation.
    - Excludes missing gene_id
    - Coerces to string (consistent keys across datasets)
    """
    padj_thresh = sanitize_padj(padj_thresh)
    lfc_thresh  = sanitize_lfc(lfc_thresh)
    sig = df["padj"] <= padj_thresh
    if regulation == "up":
        fc = df["log2FoldChange"] >= lfc_thresh
    elif regulation == "down":
        fc = df["log2FoldChange"] <= -lfc_thresh
    else:  # both
        fc = np.abs(df["log2FoldChange"]) >= lfc_thresh
    sub = df[sig & fc]
    sub = sub[sub["gene_id"].notna()].copy()
    return set(sub["gene_id"].astype(str).str.strip())

def render_upset_png(named_sets: Dict[str, Set[str]]) -> str:
    """Render an UpSet plot and return a base64 data URI."""
    # Empty guard
    total = sum(len(s) for s in named_sets.values())
    fig = plt.figure(figsize=(9, 5.5), dpi=150)
    if total == 0:
        plt.text(0.5, 0.5, "No genes at current thresholds", ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    contents = from_contents(named_sets)
    ax = fig.add_subplot(111)
    upset = UpSet(contents, subset_size="count", show_counts=True, sort_by="cardinality")
    upset.plot(fig=fig)  # IMPORTANT: plot into this figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# ---- Annotation normalizer ----
def normalize_annotation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts various common gene annotation headers and returns columns:
       gene_id, gene_symbol, gene_function (+ rich optional fields)."""
    cmap = {c.lower(): c for c in df.columns}
    def has(col): return col.lower() in cmap
    def col(col): return cmap[col.lower()]

    out = df.copy()
    # Canonical columns
    out["gene_id"] = (
        out[col("GeneID")] if has("GeneID") else
        out[col("gene_id")] if has("gene_id") else
        pd.Series([None]*len(out))
    )
    out["gene_symbol"] = (
        out[col("Symbol")] if has("Symbol") else
        out[col("gene_symbol")] if has("gene_symbol") else
        pd.Series([None]*len(out))
    )
    out["gene_function"] = (
        out[col("Description")] if has("Description") else
        out[col("gene_function")] if has("gene_function") else
        pd.Series([None]*len(out))
    )

    # Rich fields
    keep_map = {
        "description": "Description",
        "synonyms": "Synonyms",
        "gene_type": "GeneType",
        "ensembl_id": "EnsemblGeneID",
        "status": "Status",
        "chr_acc": "ChrAcc",
        "chr_start": "ChrStart",
        "chr_stop": "ChrStop",
        "strand": "Orientation",
        "length": "Length",
        "go_mf_id": "GOFunctionID",
        "go_bp_id": "GOProcessID",
        "go_cc_id": "GOComponentID",
        "go_mf": "GOFunction",
        "go_bp": "GOProcess",
        "go_cc": "GOComponent",
    }
    for new, old in keep_map.items():
        if has(old):
            out[new] = out[col(old)]

    # Normalize types
    for k in ["gene_id", "ensembl_id", "gene_symbol"]:
        if k in out.columns:
            out[k] = out[k].astype(str).str.strip()
    return out

def parse_annot_upload(contents: str, filename: str) -> pd.DataFrame:
    """Parse uploaded annotation content as CSV/TSV (delimiter auto-detected)."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.BytesIO(decoded), sep=None, engine="python", dtype=str, low_memory=False)
    return normalize_annotation_df(df)

# Load data once
DFS = [load_deg_csv(f) for f in FILES]

# -------------- Dash UI --------------
app = Dash(__name__)
app.title = "DEG Volcano + UpSet (patched)"

def threshold_block(i: int, label: str) -> html.Div:
    return html.Div(
        [
            html.H4(f"{label}", style={"margin": "0 0 6px 0"}),
            html.Label("padj ≤"),
            dcc.Input(id=f"padj-{i}", type="number", value=DEFAULT_PADJ, step=0.001, min=0, max=1, style={"width": "110px"}),
            html.Label(" |log2FC| ≥", style={"marginLeft": "10px"}),
            dcc.Input(id=f"lfc-{i}", type="number", value=DEFAULT_LFC, step=0.01, min=0, style={"width": "110px"}),
        ],
        style={"border": "1px solid #ddd", "borderRadius": "8px", "padding": "8px 10px", "minWidth": "260px"}
    )

app.layout = html.Div(
    [
        html.H2("DEG Volcano + UpSet"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Volcano dataset"),
                        dcc.Dropdown(
                            id="volcano-set",
                            options=[{"label": lbl, "value": i} for i, lbl in enumerate(LABELS)],
                            value=0, clearable=False, style={"width": "260px"}
                        ),
                    ],
                    style={"marginRight": "24px"}
                ),
                html.Div(
                    [
                        html.Label("Regulation (applies to UpSet & overlap list)"),
                        dcc.Dropdown(
                            id="regulation",
                            options=[
                                {"label": "Up only", "value": "up"},
                                {"label": "Down only", "value": "down"},
                                {"label": "Both (abs FC)", "value": "both"},
                            ],
                            value="up", clearable=False, style={"width": "260px"}
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "flex-end"}
        ),

        # Thresholds
        html.Div(
            [threshold_block(0, LABELS[0]), threshold_block(1, LABELS[1]),
             threshold_block(2, LABELS[2]), threshold_block(3, LABELS[3])],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "margin": "12px 0"}
        ),

        # Volcano
        dcc.Graph(id="volcano-graph", style={"height": "520px"}),

        html.Hr(),

        # UpSet
        html.Div([
            html.Img(id="upset-plot", style={"maxWidth": "100%"}),
        ]),

        html.Hr(),

        # Overlap selection
        html.Div([
            html.H4("Overlap genes"),
            html.Label("Select datasets to intersect"),
            dcc.Checklist(
                id="overlap-sets",
                options=[{"label": LABELS[i], "value": i} for i in range(4)],
                value=[0], inline=True
            ),
            html.Label("Mode", style={"marginLeft": "16px"}),
            dcc.RadioItems(
                id="overlap-mode",
                options=[
                    {"label": "At least these", "value": "atleast"},
                    {"label": "Exact", "value": "exact"}
                ],
                value="atleast", inline=True
            ),
            html.Div([html.Strong("Count:"), html.Span(id="overlap-count", style={"marginLeft": "6px"})], style={"marginTop": "8px"}),
        ], style={"margin": "8px 0"}),

        dash_table.DataTable(
            id="overlap-table",
            columns=[{"name": c, "id": c} for c in ["gene_id", "gene_symbol", "gene_function"]],
            page_size=12,
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": "12px"},
        ),

        html.Button("Download overlap CSV", id="dl-overlap-btn", style={"marginTop": "8px"}),
        dcc.Download(id="dl-overlap"),

        html.Hr(),

        # Annotation upload
        html.Details([
            html.Summary("Gene annotation (optional): upload CSV/TSV"),
            html.Div([
                dcc.Upload(
                    id="annot-upload",
                    children=html.Div(["Drag and drop or ", html.A("select a CSV/TSV file")]),
                    multiple=False,
                ),
                html.Div(id="annot-status", style={"marginTop": "8px", "color": "#444"}),
            ], style={"padding": "8px 0"}),
        ]),
        dcc.Store(id="annot-store"),
    ],
    style={"padding": "16px 20px", "maxWidth": "1200px", "margin": "0 auto"}
)

# ---- Volcano callback ----
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

    # Unique counts by gene_id (to match UpSet behavior)
    uniq = unique_counts_by_gene(df)
    total_n = int(len(uniq))
    up_n    = int((uniq["sig"] == "Up").sum())
    down_n  = int((uniq["sig"] == "Down").sum())

    title = (
        f"Volcano: {LABELS[idx]} (padj \u2264 {padjs[idx]:.3g}, |log2FC| \u2265 {lfcs[idx]:.2f}) — "
        f"N={total_n}, Up={up_n}, Down={down_n} (unique gene IDs)"
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

# ---- UpSet plot ----
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

# ---- Overlap list ----
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
    State("annot-store", "data"),
)
def update_overlap(selected, mode, p0, l0, p1, l1, p2, l2, p3, l3, regulation, annot_data):
    selected = selected or []
    if not selected:
        return [], "0"

    thresholds = [(sanitize_padj(p0), sanitize_lfc(l0)),
                  (sanitize_padj(p1), sanitize_lfc(l1)),
                  (sanitize_padj(p2), sanitize_lfc(l2)),
                  (sanitize_padj(p3), sanitize_lfc(l3))]
    sets = [filter_gene_set(df, padj, lfc, regulation) for df, (padj, lfc) in zip(DFS, thresholds)]

    inter = set.intersection(*(sets[i] for i in selected))
    if mode == "exact" and len(selected) < len(sets):
        # Remove any genes that appear in unselected sets
        others = set.union(*(sets[i] for i in range(len(sets)) if i not in selected))
        inter = inter - others

    genes = sorted(inter)
    base_df = pd.DataFrame({"gene_id": [str(g) for g in genes]})
    base_df["gene_id"] = base_df["gene_id"].astype(str).str.strip()
    base_df = base_df.drop_duplicates(subset=["gene_id"])

    # Merge annotations if present
    if annot_data:
        try:
            annot_df = pd.DataFrame(annot_data)
            annot_df = normalize_annotation_df(annot_df)  # ensure normalized

            # Heuristic for Ensembl-style IDs
            base_ids = base_df["gene_id"].dropna().astype(str)
            ensg_ratio = (base_ids.str.upper().str.startswith("ENSG")).mean() if len(base_ids) else 0.0

            if ensg_ratio > 0.6 and "ensembl_id" in annot_df.columns:
                annot_df["_join_key"] = annot_df["ensembl_id"]
            else:
                annot_df["_join_key"] = annot_df["gene_id"]

            joined = base_df.merge(
                annot_df.drop_duplicates("_join_key"),
                left_on="gene_id", right_on="_join_key", how="left"
            ).drop(columns=["_join_key"])

            # Nice column order
            preferred = [c for c in [
                "gene_id", "gene_symbol", "gene_function", "description", "synonyms",
                "gene_type", "ensembl_id", "status", "chr_acc", "chr_start", "chr_stop",
                "strand", "length", "go_mf", "go_bp", "go_cc", "go_mf_id", "go_bp_id", "go_cc_id"
            ] if c in joined.columns]
            other = [c for c in joined.columns if c not in preferred]
            base_df = joined[preferred + other]
        except Exception as e:
            print("Annotation merge failed:", e)

    return base_df.to_dict("records"), str(len(base_df))

# ---- Download overlap CSV ----
@app.callback(
    Output("dl-overlap", "data"),
    Input("dl-overlap-btn", "n_clicks"),
    State("overlap-table", "data"),
    prevent_initial_call=True,
)
def download_overlap(n, data):
    df = pd.DataFrame(data or [])
    return dcc.send_data_frame(df.to_csv, "overlap_genes.csv", index=False)

# ---- Annotation upload ----
@app.callback(
    Output("annot-store", "data"),
    Output("annot-status", "children"),
    Input("annot-upload", "contents"),
    State("annot-upload", "filename"),
    prevent_initial_call=True,
)
def upload_annot(contents, filename):
    if not contents:
        return dash.no_update, "No file."
    try:
        df = parse_annot_upload(contents, filename or "annotation")
        return df.to_dict("records"), f"Loaded {len(df)} annotation rows."
    except Exception as e:
        return dash.no_update, f"Failed to parse annotation: {e}"

if __name__ == "__main__":
    app.run(debug=True)
