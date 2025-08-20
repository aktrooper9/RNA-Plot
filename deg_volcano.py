#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

# Optional: for UpSet chart as an image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from upsetplot import UpSet

# --------------
# Config
# --------------
# Default DEG table filenames (change via env vars if needed)
DEG_FILES = [
    os.getenv("DEG0", "135092.csv"),
    os.getenv("DEG1", "29081.csv"),
    os.getenv("DEG2", "115828.csv"),
    os.getenv("DEG3", "99248.csv"),
]

DEG_LABELS = [
    os.getenv("DEG0_LABEL", Path(DEG_FILES[0]).stem),
    os.getenv("DEG1_LABEL", Path(DEG_FILES[1]).stem),
    os.getenv("DEG2_LABEL", Path(DEG_FILES[2]).stem),
    os.getenv("DEG3_LABEL", Path(DEG_FILES[3]).stem),
]

# Annotation file (case-sensitive on Linux). You currently have 'Human.GRCh38.p13.csv' on Render.
ANNOT_PATH = os.getenv("ANNOT_PATH", "Human.GRCh38.p13.csv")  # supports .csv/.tsv/.gz

# --------------
# Helpers: IO & normalization
# --------------
def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".csv", ".tsv", ".txt", ".gz"):
        # Let pandas infer delimiter + compression, works for csv/tsv and gz
        return pd.read_csv(path, sep=None, engine="python", dtype=str, low_memory=False, compression="infer")
    # Fallback to CSV
    return pd.read_csv(path, dtype=str, low_memory=False)

def _resolve_path(pth: str) -> Optional[Path]:
    p = Path(pth)
    here = Path(__file__).parent
    for cand in (p, here/p, here/"data"/p, Path.cwd()/p):
        if cand.is_file():
            return cand
    # case-insensitive fallback in app dir
    target = p.name.lower()
    try:
        for cand in here.iterdir():
            if cand.is_file() and cand.name.lower() == target:
                return cand
    except Exception:
        pass
    return None

# Column name candidates
_GENE_KEYS = ("gene_id", "geneid", "ensembl_gene_id", "ensemblgeneid", "ensembl", "id", "gene")
_LFC_KEYS  = ("log2foldchange", "logfc", "log2fc", "log2_fc", "log2 fold change")
_PADJ_KEYS = ("padj", "p_adj", "fdr", "qval", "adj.p.val", "q_value", "qvalue")

def _std_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """Return a copy with standardized columns: gene_id, logfc, padj. Also return original names used."""
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(keys, default=None):
        for k in keys:
            if k in cols:
                return cols[k]
        return default

    gcol = pick(_GENE_KEYS, next(iter(df.columns)))
    lcol = pick(_LFC_KEYS)
    pcol = pick(_PADJ_KEYS)

    out = df.copy()
    if gcol not in out:
        gcol = next(iter(out.columns))  # fallback
    # Convert numeric cols
    if lcol is not None:
        out["logfc"] = pd.to_numeric(out[lcol], errors="coerce")
    else:
        out["logfc"] = np.nan
        lcol = "logfc"
    if pcol is not None:
        out["padj_std"] = pd.to_numeric(out[pcol], errors="coerce")
    else:
        out["padj_std"] = np.nan
        pcol = "padj_std"

    # gene_id normalized (strip version)
    out["gene_id_std"] = out[gcol].astype(str).str.strip().str.split(".").str[0]

    # derived for volcano
    out["_neglog10_padj"] = -np.log10(out["padj_std"].replace(0, np.nextafter(0, 1)))
    out["_is_up"] = out["logfc"] > 0
    out["_is_down"] = out["logfc"] < 0

    return out, "gene_id_std", lcol, "padj_std"

def _load_deg(path_str: str) -> Optional[pd.DataFrame]:
    p = _resolve_path(path_str)
    if p is None:
        return None
    try:
        df = _read_table(p)
        df, gcol, lcol, pcol = _std_columns(df)
        return df
    except Exception as e:
        print(f"[deg] Failed to load {path_str}: {e}")
        return None

def _normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    # Prefer EnsemblGeneID for joining with DEG gene_id
    gid = None
    for k in ("ensemblgeneid", "ensembl_gene_id", "gene_id", "geneid", "ensembl", "id", "gene"):
        if k in cols:
            gid = cols[k]; break
    if gid is None:
        gid = df.columns[0]
    sym = None
    for k in ("gene_symbol", "symbol", "hgnc_symbol", "genesymbol"):
        if k in cols:
            sym = cols[k]; break
    fun = None
    for k in ("gene_function", "gene_name", "name", "description", "gene_description"):
        if k in cols:
            fun = cols[k]; break

    out = df[[gid]].copy().rename(columns={gid: "gene_id"})
    out["gene_id"] = out["gene_id"].astype(str).str.strip().str.split(".").str[0]
    if sym:
        out["gene_symbol"] = df[sym].astype(str).str.strip()
    if fun:
        out["gene_function"] = df[fun].astype(str).str.strip()
    return out.drop_duplicates(subset=["gene_id"])

def _load_annotations() -> tuple[Optional[List[dict]], str]:
    p = _resolve_path(ANNOT_PATH)
    if p is None:
        msg = f"Annotation file not found: {ANNOT_PATH} (set ANNOT_PATH or upload via UI)."
        print("[annot]", msg)
        return None, msg
    try:
        raw = _read_table(p)
        norm = _normalize_annotation_columns(raw)
        if norm.empty:
            msg = f"Loaded {p.name}, but no recognizable gene ID column."
            print("[annot]", msg)
            return None, msg
        data = norm.to_dict("records")
        msg = f"Loaded {len(data)} annotations from {p.name}."
        print("[annot]", msg)
        return data, msg
    except Exception as e:
        msg = f"Failed to load annotations from {p}: {e}"
        print("[annot]", msg)
        return None, msg

# --------------
# Data: load up to 4 DEG sets eagerly (optional; users can still upload later)
# --------------
DFS: List[Optional[pd.DataFrame]] = []
for f in DEG_FILES:
    DFS.append(_load_deg(f))

# --------------
# Dash app
# --------------
external_scripts = []
external_stylesheets = []

app: Dash = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
server = app.server  # for gunicorn

# Preload annotations
_INITIAL_ANNOT_DATA, _INITIAL_ANNOT_STATUS = _load_annotations()

def dataset_options():
    opts = []
    for i, (label, df) in enumerate(zip(DEG_LABELS, DFS)):
        lab = f"{label} ({'loaded' if df is not None else 'missing'})"
        opts.append({"label": lab, "value": i})
    return opts

def sanitize_padj(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if v <= 0:
            return None
        return v
    except Exception:
        return None

def sanitize_lfc(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return abs(float(x))
    except Exception:
        return None

def filter_gene_set(df: Optional[pd.DataFrame], padj: Optional[float], abs_lfc: Optional[float], regulation: str) -> set:
    if df is None or df.empty:
        return set()
    mask = pd.Series(True, index=df.index)
    if padj is not None:
        mask &= df["padj_std"] <= padj
    if abs_lfc is not None:
        mask &= df["logfc"].abs() >= abs_lfc
    if regulation == "up":
        mask &= df["_is_up"]
    elif regulation == "down":
        mask &= df["_is_down"]
    return set(df.loc[mask, "gene_id_std"].dropna().astype(str))

def build_volcano(df: Optional[pd.DataFrame], padj: Optional[float], abs_lfc: Optional[float]) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    # Significant flags
    sig = pd.Series(True, index=df.index)
    if padj is not None:
        sig &= df["padj_std"] <= padj
    if abs_lfc is not None:
        sig &= df["logfc"].abs() >= abs_lfc
    color = np.where(~sig, "ns", np.where(df["_is_up"], "up", "down"))
    plot_df = df.copy()
    plot_df["sig_state"] = color
    fig = px.scatter(
        plot_df,
        x="logfc",
        y="_neglog10_padj",
        color="sig_state",
        category_orders={"sig_state": ["ns", "up", "down"]},
        hover_data={"logfc":":.3f", "_neglog10_padj":":.3f", "gene_id_std": True, "sig_state": False},
        labels={"logfc":"log2FoldChange", "_neglog10_padj":"-log10(adj p)"},
        opacity=0.85,
    )
    fig.update_layout(margin=dict(l=30,r=10,t=30,b=40), legend_title_text="")
    return fig

def render_upset(sets: List[set], labels: List[str]) -> str:
    """Return a base64 PNG of an UpSet plot (membership across sets)."""
    try:
        # Build membership dict for UpSet from union of all genes
        all_genes = set().union(*sets) if sets else set()
        if not all_genes:
            return ""
        membership = {lab: [g in s for g in all_genes] for lab, s in zip(labels, sets)}
        df_bool = pd.DataFrame(membership, index=list(all_genes))
        plt.figure(figsize=(8, 3 + 0.8*len(sets)))
        UpSet(df_bool, subset_size="count", show_counts=True).plot()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=144, bbox_inches="tight")
        plt.close()
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/png;base64," + data
    except Exception as e:
        print("[upset] error:", e)
        return ""

app.layout = html.Div([
    html.H2("DEG Volcano + Overlap + Enrichment (Dash)"),
    html.Div("This app auto-loads up to four DEG tables and a human annotation file at startup. Uploads can override."),
    html.Div([
        html.Div([
            html.Label("Choose dataset for volcano"),
            dcc.Dropdown(id="volcano-dataset", options=dataset_options(), value=0, clearable=False, style={"width":"320px"}),
        ], style={"marginRight":"16px"}),
        html.Div([
            html.Label("Padj cutoff"),
            dcc.Input(id="padj", type="number", value=0.05, step=0.001, style={"width":"120px"}),
        ], style={"marginRight":"16px"}),
        html.Div([
            html.Label("|log2FC| cutoff"),
            dcc.Input(id="lfc", type="number", value=1.0, step=0.1, style={"width":"120px"}),
        ]),
    ], style={"display":"flex","alignItems":"flex-end","gap":"8px","margin":"10px 0 20px"}),

    html.Div([
        html.Div([
            dcc.Graph(id="volcano-fig", style={"height":"520px"}),
        ], style={"flex":"1"}),
        html.Div([
            html.Label("Regulation filter for sets"),
            dcc.RadioItems(
                id="regulation", options=[
                    {"label":"Both","value":"both"},
                    {"label":"Up","value":"up"},
                    {"label":"Down","value":"down"},
                ], value="both", inline=True
            ),
            html.Br(),
            html.Label("Select sets to overlap"),
            dcc.Checklist(id="overlap-sets", options=[{"label": lab, "value": i} for i,lab in enumerate(DEG_LABELS)], value=[0,1], inline=False),
            html.Br(),
            html.Label("Overlap mode"),
            dcc.Dropdown(
                id="overlap-mode",
                options=[
                    {"label":"Intersection (inclusive)","value":"inclusive"},
                    {"label":"Exact (intersection minus any other set)","value":"exact"},
                ], value="inclusive", clearable=False, style={"width":"320px"}
            ),
            html.Br(),
            html.Img(id="upset-img", style={"maxWidth":"420px", "border":"1px solid #ddd"}),
        ], style={"width":"460px","paddingLeft":"10px"}),
    ], style={"display":"flex"}),

    html.Hr(),
    html.H3("Overlapping genes (with annotations if available)"),
    html.Div(id="annot-status", children=_INITIAL_ANNOT_STATUS or "No annotations loaded yet.", style={"margin":"0 0 10px","color":"#555"}),
    dcc.Store(id="annot-store", data=_INITIAL_ANNOT_DATA),
    html.Div([
        html.Button("Download overlap (CSV)", id="dl-overlap-btn"),
        dcc.Download(id="dl-overlap"),
        html.Button("Download annotated overlap (CSV)", id="dl-overlap-ann-btn", style={"marginLeft":"12px"}),
        dcc.Download(id="dl-overlap-ann"),
    ], style={"marginBottom":"6px"}),
    dash_table.DataTable(
        id="overlap-table",
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"maxHeight":"420px","overflowY":"auto","border":"1px solid #eee"},
        style_cell={"fontFamily":"Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                    "fontSize":12,"padding":"6px"},
    ),

    html.Hr(),
    html.H3("Enrichment (Enrichr via gseapy) on the current overlap genes"),
    html.Div([
        html.Label("Libraries"),
        dcc.Dropdown(
            id="enrich-libraries", multi=True,
            options=[
                {"label":"KEGG_2021_Human", "value":"KEGG_2021_Human"},
                {"label":"GO_Biological_Process_2023", "value":"GO_Biological_Process_2023"},
                {"label":"GO_Molecular_Function_2023", "value":"GO_Molecular_Function_2023"},
                {"label":"GO_Cellular_Component_2023", "value":"GO_Cellular_Component_2023"},
            ],
            value=["KEGG_2021_Human", "GO_Biological_Process_2023"],
            style={"width":"360px"}
        ),
        html.Label("Species", style={"marginLeft":"16px"}),
        dcc.Dropdown(id="enrich-species", options=[{"label":"human","value":"human"},{"label":"mouse","value":"mouse"}], value="human", clearable=False, style={"width":"160px"}),
        html.Label("Top N", style={"marginLeft":"16px"}),
        dcc.Input(id="enrich-topn", type="number", value=20, min=1, max=100, style={"width":"100px"}),
        html.Button("Run enrichment", id="enrich-run", n_clicks=0, style={"marginLeft":"12px"}),
        html.Button("Download enrichment CSV", id="enrich-dl-btn", n_clicks=0, style={"marginLeft":"12px"}),
        dcc.Download(id="enrich-dl"),
    ], style={"display":"flex","alignItems":"center","gap":"8px","margin":"6px 0 12px"}),
    html.Div(id="enrich-status", style={"color":"#555","marginBottom":"6px"}),
    dcc.Graph(id="enrich-bar", style={"height":"580px"}),
    dash_table.DataTable(
        id="enrich-table",
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={"maxHeight":"400px","overflowY":"auto","border":"1px solid #eee"},
        style_cell={"fontFamily":"Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                    "fontSize":12,"padding":"6px"},
    ),

    html.Hr(),
    html.Div("Upload new annotation file (CSV/TSV/.gz) to override the current one:"),
    dcc.Upload(
        id="annot-upload",
        children=html.Div(["Drag & Drop or ", html.A("Select Annotation File")]),
        style={
            "width":"100%","height":"60px","lineHeight":"60px",
            "borderWidth":"1px","borderStyle":"dashed","borderRadius":"8px",
            "textAlign":"center","margin":"6px 0 10px"
        },
        multiple=False,
    ),
])

# --------------
# Callbacks
# --------------
@app.callback(
    Output("volcano-fig", "figure"),
    Input("volcano-dataset", "value"),
    Input("padj", "value"),
    Input("lfc", "value"),
)
def update_volcano(dataset_idx, padj, lfc):
    padj = sanitize_padj(padj)
    lfc = sanitize_lfc(lfc)
    try:
        df = DFS[dataset_idx] if dataset_idx is not None else None
    except Exception:
        df = None
    return build_volcano(df, padj, lfc)

@app.callback(
    Output("upset-img", "src"),
    Output("overlap-table", "data"),
    Output("overlap-table", "columns"),
    Output("annot-status", "children"),
    Input("overlap-sets", "value"),
    Input("overlap-mode", "value"),
    Input("regulation", "value"),
    Input("padj", "value"),
    Input("lfc", "value"),
    State("annot-store", "data"),
)
def update_overlap(selected_sets, mode, regulation, padj, lfc, annot_data):
    padj = sanitize_padj(padj)
    lfc = sanitize_lfc(lfc)
    selected = set(int(i) for i in (selected_sets or []))
    if not selected:
        return "", [], [], "Select one or more sets."
    # Build filtered sets
    sets = [filter_gene_set(df, padj, lfc, regulation) for df in DFS]
    chosen_sets = [sets[i] for i in selected]
    labels = [DEG_LABELS[i] for i in selected]

    # Compute overlap
    inter = set.intersection(*chosen_sets) if chosen_sets else set()
    if mode == "exact":
        others = set.union(*(sets[i] for i in range(len(sets)) if i not in selected)) if len(selected) < len(sets) else set()
        inter = inter - others

    genes = sorted(inter)
    base_df = pd.DataFrame({"gene_id": genes})
    base_df["gene_id"] = base_df["gene_id"].astype(str).str.split(".").str[0]

    status_msg = _INITIAL_ANNOT_STATUS or "No annotations loaded yet."
    if annot_data:
        try:
            ann = pd.DataFrame(annot_data)
            ann["gene_id"] = ann["gene_id"].astype(str).str.split(".").str[0]
            out = base_df.merge(ann, on="gene_id", how="left")
            cols = [{"name": c, "id": c} for c in out.columns]
            upset_src = render_upset(chosen_sets, labels)
            return upset_src, out.to_dict("records"), cols, f"{len(out)} genes (annotations merged). " + status_msg
        except Exception as e:
            cols = [{"name":"gene_id","id":"gene_id"}]
            upset_src = render_upset(chosen_sets, labels)
            return upset_src, base_df.to_dict("records"), cols, f"{len(base_df)} genes (annotation merge failed: {e}). " + status_msg
    else:
        cols = [{"name":"gene_id","id":"gene_id"}]
        upset_src = render_upset(chosen_sets, labels)
        return upset_src, base_df.to_dict("records"), cols, f"{len(base_df)} genes (no annotations loaded)."

# Downloads
@app.callback(
    Output("dl-overlap", "data"),
    Input("dl-overlap-btn", "n_clicks"),
    State("overlap-table", "data"),
    prevent_initial_call=True,
)
def dl_overlap(n, rows):
    if not rows:
        return dash.no_update
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "overlap_genes.csv", index=False)

@app.callback(
    Output("dl-overlap-ann", "data"),
    Input("dl-overlap-ann-btn", "n_clicks"),
    State("overlap-table", "data"),
    prevent_initial_call=True,
)
def dl_overlap_ann(n, rows):
    if not rows:
        return dash.no_update
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "overlap_genes_annotated.csv", index=False)

# Annotation upload (overrides store)
@app.callback(
    Output("annot-store", "data"),
    Output("annot-status", "children"),
    Input("annot-upload", "contents"),
    State("annot-upload", "filename"),
    prevent_initial_call=True,
)
def handle_annot_upload(contents, filename):
    if not contents:
        return dash.no_update, dash.no_update
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df_raw = pd.read_csv(io.BytesIO(decoded), sep=None, engine="python", dtype=str, low_memory=False, compression="infer")
        df_norm = _normalize_annotation_columns(df_raw)
        if df_norm.empty:
            return dash.no_update, "Uploaded file did not contain a recognizable gene ID column."
        return df_norm.to_dict("records"), f"Loaded {len(df_norm)} annotations from {filename}."
    except Exception as e:
        return dash.no_update, f"Failed to parse {filename}: {e}"

# --------------
# Enrichment callbacks
# --------------
@app.callback(
    Output("enrich-table", "data"),
    Output("enrich-table", "columns"),
    Output("enrich-bar", "figure"),
    Output("enrich-status", "children"),
    Input("enrich-run", "n_clicks"),
    State("overlap-table", "data"),
    State("enrich-libraries", "value"),
    State("enrich-species", "value"),
    State("enrich-topn", "value"),
    prevent_initial_call=True,
)
def run_enrichment(n_clicks, overlap_rows, libraries, species, topn):
    import numpy as np
    if not overlap_rows:
        return [], [], go.Figure(), "No overlap genes to run enrichment on."
    genes = pd.DataFrame(overlap_rows)["gene_id"].dropna().astype(str).str.strip().str.upper().tolist()
    if not genes:
        return [], [], go.Figure(), "No valid gene IDs found in overlap."
    if not libraries:
        return [], [], go.Figure(), "Choose at least one library."

    # Try gseapy
    try:
        import gseapy as gp
    except Exception as e:
        return [], [], go.Figure(), f"gseapy not available: {e}"

    frames = []
    msgs = []
    for lib in libraries:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism=species, no_plot=True, cutoff=1.0)
            df = enr.results.copy()
            # standardize
            rename = {
                "Term": "term",
                "Adjusted P-value": "padj",
                "P-value": "pval",
                "Overlap": "overlap",
                "Odds Ratio": "odds_ratio",
                "Combined Score": "combined_score",
                "Genes": "genes",
            }
            for k, v in rename.items():
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            keep = [c for c in ["term","padj","pval","overlap","odds_ratio","combined_score","genes"] if c in df.columns]
            df = df[keep].sort_values(["padj","pval"], ascending=[True, True])
            df["library"] = lib
            frames.append(df)
            msgs.append(f"{lib}: {len(df)} terms")
        except Exception as e:
            msgs.append(f"{lib}: ERROR {e}")
    if not frames:
        return [], [], go.Figure(), "; ".join(msgs)
    res = pd.concat(frames, ignore_index=True)

    cols = [{"name": c, "id": c} for c in res.columns]
    data = res.to_dict("records")

    # plot
    if "padj" in res.columns and "term" in res.columns:
        plot_df = res.copy()
        plot_df["neglog10_padj"] = -np.log10(plot_df["padj"].replace(0, np.nextafter(0, 1)))
        # topN per library
        plot_df = plot_df.sort_values(["library","padj"]).groupby("library", as_index=False).head(int(topn or 20))
        fig = px.bar(plot_df, x="neglog10_padj", y="term", orientation="h", facet_row="library",
                     labels={"neglog10_padj":"-log10(adj p)","term":"Term"})
        fig.update_yaxes(matches=None, automargin=True)
        fig.update_layout(showlegend=False, margin=dict(l=220, r=20, t=40, b=40))
    else:
        fig = go.Figure()

    status = f"Enrichment on {len(genes)} overlap genes. " + " | ".join(msgs)
    return data, cols, fig, status

# --------------
# Main
# --------------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "8050")))
