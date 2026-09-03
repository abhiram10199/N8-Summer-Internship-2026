import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import ast
import json
import glob
import os
import sys

# Page Configuration
st.set_page_config(
    page_title="ALFD Lyapunov Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Scientific Dark Theme CSS (No neon gradients, no emoji spam)
st.markdown("""
<style>
    /* Base typography & containers */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .dashboard-title {
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #f0f6fc;
        margin-bottom: 0.25rem;
    }
    .dashboard-subtitle {
        font-size: 0.88rem;
        color: #8b949e;
        margin-bottom: 1.25rem;
    }
    
    /* Metadata banner */
    .meta-panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 1.25rem;
    }
    .meta-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .badge-success {
        display: inline-block;
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-warning {
        display: inline-block;
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .meta-tags {
        font-size: 0.85rem;
        color: #8b949e;
    }
    .meta-tags strong {
        color: #c9d1d9;
    }
    .equation-display {
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        font-size: 1.05rem;
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 10px 14px;
        color: #7ee787;
        overflow-x: auto;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.92rem;
        font-weight: 500;
        color: #8b949e;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES & STRING SANITIZATION ---

def sanitize_expression(expr_str: str) -> str:
    """Cleans raw string from log into valid SymPy parseable syntax."""
    if not expr_str:
        return ""
    s = str(expr_str).strip()
    s = re.sub(r'^\s*V\s*\([^)]*\)\s*=\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\s*V\s*=\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', s)
    s = s.replace('−', '-').replace('–', '-')
    
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(subscript_map)
    s = s.replace('²', '**2').replace('³', '**3').replace('⁴', '**4')
    s = re.sub(r'\bx\s+(\d+)\s+(\d+)\b', r'x\1**\2', s)
    s = re.sub(r'\bx\s+(\d+)\b', r'x\1', s)
    s = s.replace('^', '**')
    s = s.replace('[', '').replace(']', '').strip()
    return s

def parse_prefix_traversal(traversal_str: str) -> str:
    """Converts Polish/prefix notation tokens into standard infix algebraic syntax."""
    if not traversal_str:
        return None
    tokens = [t.strip() for t in traversal_str.split(',') if t.strip()]
    stack = []
    for token in reversed(tokens):
        if token in ['add', 'sub', 'mul', 'div']:
            if len(stack) < 2: return None
            left = stack.pop()
            right = stack.pop()
            if token == 'add': stack.append(f"({left} + {right})")
            elif token == 'sub': stack.append(f"({left} - {right})")
            elif token == 'mul': stack.append(f"({left} * {right})")
            elif token == 'div': stack.append(f"({left} / {right})")
        elif token in ['sin', 'cos', 'exp', 'log']:
            if len(stack) < 1: return None
            arg = stack.pop()
            stack.append(f"{token}({arg})")
        elif token == 'n2':  
            if len(stack) < 1: return None
            arg = stack.pop()
            stack.append(f"({arg}**2)")
        else:
            stack.append(token) 
    return stack[0] if stack else None

def parse_float_safe(val) -> float:
    try:
        f = float(val)
        return np.nan if np.isnan(f) else f
    except (ValueError, TypeError):
        return np.nan

# --- LOG PARSER ---

def parse_log_file(log_text: str):
    """
    Parses ALFD execution log:
    - Header (lines 4-6): dataset, dimensionality, state variables, domain bounds
    - Training telemetry: epoch metrics, loss, rewards, timing
    - Optimization archive: PQT buffer, Hall of Fame, Pareto frontier
    - Result dictionary: termination status, NMSE, execution walltime
    """
    lines = log_text.splitlines()
    
    metadata = {
        "dataset": "Unknown",
        "dimensions": 2,
        "variables": ["x1", "x2"],
        "domain_bounds": {},
        "parameters": "Unknown",
        "early_stopping": False,
        "success": False,
        "discovered_expression": None,
        "test_nmse": None,
        "walltime": None,
        "total_samples": None,
        "cached_samples": None,
        "cache_hit_rate": None,
        "seed": None,
        "error_square": 0,
        "error_underflow": 0,
        "error_overflow": 0,
        "config": {}
    }

    # 1. Parse dataset header specification (lines 4-6)
    for line in lines[:35]:
        if "0*x1" in line or ('{"x1":' in line and ("'U'" in line or '"U"' in line or "'E'" in line or '"E"' in line)):
            m = re.search(r'([a-zA-Z0-9_-]+)\s+(\d+)\s+([^\s\t]+)\s+(\{.*\})', line)
            if m:
                metadata["dataset"] = m.group(1).strip()
                metadata["dimensions"] = int(m.group(2).strip())
                raw_json = m.group(4).strip()
                last_brace = raw_json.rfind('}')
                if last_brace != -1:
                    raw_json = raw_json[:last_brace + 1]
                try:
                    spec_dict = json.loads(raw_json)
                    bounds = {}
                    var_list = []
                    for k, v in spec_dict.items():
                        var_list.append(k)
                        if "U" in v: bounds[k] = (float(v["U"][0]), float(v["U"][1]))
                        elif "E" in v: bounds[k] = (float(v["E"][0]), float(v["E"][1]))
                    if bounds:
                        metadata["domain_bounds"] = bounds
                    if var_list:
                        metadata["variables"] = var_list
                except Exception:
                    pass
            break
            
    if not metadata["domain_bounds"] or not metadata["variables"]:
        num_v = metadata["dimensions"]
        metadata["variables"] = [f"x{i}" for i in range(1, num_v + 1)]
        metadata["domain_bounds"] = {v: (-2.0, 2.0) for v in metadata["variables"]}

    # 2. Parse [TEST RESULT] summary dictionary
    test_result_match = re.search(r"\[TEST RESULT\]\s*(\{.*?\})", log_text)
    if test_result_match:
        try:
            dict_str = test_result_match.group(1)
            clean_dict_str = re.sub(r"'program':\s*([^,']+?(?:,[^,']+?)*),\s*'([a-zA-Z0-9_]+)':", r"'program': '\1', '\2':", dict_str)
            res_dict = ast.literal_eval(clean_dict_str)
            metadata["success"] = bool(res_dict.get("success", False))
            raw_expr = res_dict.get("expression", None)
            metadata["discovered_expression"] = sanitize_expression(raw_expr) if raw_expr else None
            metadata["test_nmse"] = res_dict.get("nmse_test", None)
            metadata["walltime"] = res_dict.get("t", None)
            metadata["total_samples"] = res_dict.get("n_samples", None)
            metadata["cached_samples"] = res_dict.get("n_cached", None)
            if metadata["total_samples"] and metadata["cached_samples"]:
                metadata["cache_hit_rate"] = (metadata["cached_samples"] / metadata["total_samples"]) * 100.0
            metadata["seed"] = res_dict.get("seed", None)
            metadata["dataset"] = res_dict.get("dataset", metadata["dataset"])
            metadata["error_square"] = res_dict.get("error_square", 0)
            metadata["error_underflow"] = res_dict.get("error_node_underflow", 0)
            metadata["error_overflow"] = res_dict.get("error_node_overflow", 0)
        except Exception:
            pass

    # 3. Parse Configuration block
    config_match = re.search(r"config:\s*(\{.*\})", log_text)
    if config_match:
        try:
            metadata["config"] = ast.literal_eval(config_match.group(1))
        except Exception:
            pass

    for line in lines:
        if "Early stopping criteria met" in line:
            metadata["early_stopping"] = True
        if "TransformerTreeEncoderController parameters:" in line:
            m_param = re.search(r"parameters:\s*(\d+)", line)
            if m_param: metadata["parameters"] = f"{int(m_param.group(1)):,}"

    # 4. Parse Epoch Telemetry
    metric_regex = re.compile(
        r"\[Test epoch=(\d+)\].*?nevals=(\d+).*?train_loss=([^\s\t|]+).*?eqs_invalid %=([^\s\t|]+).*?r_best=([^\s\t|]+)"
        r"(?:.*?quantile=([^\s\t|]+))?"
        r"(?:.*?r_quantile_mean=([^\s\t|]+))?"
        r"(?:.*?r_raw_sum=([^\s\t|]+))?"
        r"(?:.*?r_raw_mean=([^\s\t|]+))?"
        r"(?:.*?r2=([^\s\t|]+))?"
        r"(?:.*?acc_iid=([^\s\t|]+))?"
        r"(?:.*?acc_ood=([^\s\t|]+))?"
        r"(?:.*?nmse_test=([^\s\t|]+))?"
        r"(?:.*?true_equation_set_count=(\d+))?"
        r".*?s/it=([^\s\t|]+)"
    )
    
    epochs, nevals, losses, invalids, best_rewards = [], [], [], [], []
    quantiles, r_quantile_means, raw_means = [], [], []
    r2_scores, nmse_tests, eq_counts, compute_times = [], [], [], []
    gp_runtimes_map, gp_means_map, gp_success_map = {}, {}, {}
    current_epoch = 0
    
    for line in lines:
        if "Genetic Porgramming Runtime" in line:
            m_gpr = re.search(r"Genetic Porgramming Runtime\s*([\d.]+)", line)
            if m_gpr: gp_runtimes_map[current_epoch + 1] = float(m_gpr.group(1))
            
        if "GP output mean:" in line:
            m_gpm = re.search(r"GP output mean:\s*([\d.eE+-]+|nan)", line)
            if m_gpm: gp_means_map[current_epoch + 1] = parse_float_safe(m_gpm.group(1))
            
        if "GP output success rate:" in line:
            m_gps = re.search(r"GP output success rate:\s*([\d.eE+-]+|nan)", line)
            if m_gps: gp_success_map[current_epoch + 1] = parse_float_safe(m_gps.group(1))

        m = metric_regex.search(line)
        if m:
            ep = int(m.group(1))
            current_epoch = ep
            epochs.append(ep)
            nevals.append(int(m.group(2)))
            losses.append(parse_float_safe(m.group(3)))
            invalids.append(parse_float_safe(m.group(4)))
            best_rewards.append(parse_float_safe(m.group(5)))
            quantiles.append(parse_float_safe(m.group(6)) if m.group(6) else np.nan)
            r_quantile_means.append(parse_float_safe(m.group(7)) if m.group(7) else np.nan)
            raw_means.append(parse_float_safe(m.group(9)) if m.group(9) else np.nan)
            r2_scores.append(parse_float_safe(m.group(10)) if m.group(10) else np.nan)
            nmse_tests.append(parse_float_safe(m.group(13)) if m.group(13) else np.nan)
            eq_counts.append(int(m.group(14)) if m.group(14) else 0)
            compute_times.append(parse_float_safe(m.group(15)) if m.group(15) else np.nan)

    gp_runtimes = [gp_runtimes_map.get(ep, np.nan) for ep in epochs]
    gp_means = [gp_means_map.get(ep, np.nan) for ep in epochs]
    gp_success = [gp_success_map.get(ep, np.nan) for ep in epochs]

    metrics_df = pd.DataFrame({
        "Epoch": epochs,
        "Evaluations": nevals,
        "Policy Loss": losses,
        "Invalid (%)": invalids,
        "Best Reward": best_rewards,
        "Quantile Cutoff": quantiles,
        "Elite Mean": r_quantile_means,
        "Batch Raw Mean": raw_means,
        "NMSE Test": nmse_tests,
        "R2 Score": r2_scores,
        "Unique Equations": eq_counts,
        "GP Mean Reward": gp_means,
        "GP Success Rate": gp_success,
        "GP Time (s)": gp_runtimes,
        "Step Time (s)": compute_times
    }) if epochs else None

    # 5. Parse Priority Queue Training (PQT) Buffer
    pqt_entries = []
    pqt_blocks = re.findall(r"Priority queue entry (\d+):(.*?)(?=(?:Priority queue entry|\-\- EVALUATION END))", log_text, re.DOTALL)
    for entry_idx, block in pqt_blocks:
        r_m = re.search(r"Reward:\s*([^\n\r]+)", block)
        off_m = re.search(r"Count Off-policy:\s*(\d+)", block)
        on_m = re.search(r"Count On-policy:\s*(\d+)", block)
        orig_m = re.search(r"Originally on Policy:\s*(True|False)", block)
        trav_m = re.search(r"Traversal:\s*([^\n\r]+)", block)
        
        parsed_eq = parse_prefix_traversal(trav_m.group(1).strip()) if trav_m else None
        pqt_entries.append({
            "Rank": int(entry_idx),
            "Reward": parse_float_safe(r_m.group(1).strip()) if r_m else 0.0,
            "On-Policy Hits": int(on_m.group(1)) if on_m else 0,
            "Off-Policy Hits": int(off_m.group(1)) if off_m else 0,
            "Originally On-Policy": orig_m.group(1) if orig_m else "Unknown",
            "Expression": sanitize_expression(parsed_eq) if parsed_eq else "N/A"
        })
    pqt_df = pd.DataFrame(pqt_entries) if pqt_entries else None

    # 6. Parse Hall of Fame & Pareto Front
    hof_list, pareto_list = [], []
    in_hof, in_pareto = False, False
    for line in lines:
        if "Hall of Fame" in line:
            in_hof, in_pareto = True, False
            continue
        elif "Pareto Front" in line:
            in_hof, in_pareto = False, True
            continue
        elif "ANALYZING LOG END" in line or ("--" in line and not line.startswith("MainProcess")):
            in_hof, in_pareto = False, False
            
        if in_hof and "<--" in line:
            m = re.search(r"(\d+):\s*S=\d+\s*R=([\d.]+)\s*<--\s*\[(.*?)\]", line)
            if m:
                hof_list.append({
                    "Rank": int(m.group(1)),
                    "Reward": float(m.group(2)),
                    "Expression": sanitize_expression(m.group(3))
                })
        if in_pareto and "<--" in line:
            m = re.search(r"(\d+):\s*S=\d+\s*R=([\d.]+)\s*C=([\d.]+)\s*<--\s*\[(.*?)\]", line)
            if m:
                pareto_list.append({
                    "Rank": int(m.group(1)),
                    "Reward": float(m.group(2)),
                    "Complexity": float(m.group(3)),
                    "Expression": sanitize_expression(m.group(4))
                })
                
    hof_df = pd.DataFrame(hof_list) if hof_list else None
    pareto_df = pd.DataFrame(pareto_list) if pareto_list else None

    return metadata, metrics_df, pqt_df, hof_df, pareto_df

# --- SCIENTIFIC PLOT THEME HELPER ---

def apply_scientific_layout(fig, title="", height=440):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f0f6fc")),
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="system-ui, -apple-system, sans-serif", color="#c9d1d9", size=12),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", showgrid=True),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", showgrid=True),
        legend=dict(bgcolor="rgba(22, 27, 34, 0.8)", bordercolor="#30363d", borderwidth=1)
    )
    return fig

# --- SIDEBAR LOG SELECTION ---

workspace_root = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
logs_dir = os.path.join(workspace_root, "logs")
existing_log_files = sorted(glob.glob(os.path.join(logs_dir, "*_log.txt")), reverse=True)

with st.sidebar:
    st.markdown("### Experiment Data Source")
    
    source_mode = st.radio("Log Source", ["Select Existing Run", "Upload Log File"], horizontal=True, label_visibility="collapsed")
    log_content = None
    selected_filename = None
    
    if source_mode == "Select Existing Run" and existing_log_files:
        log_options = {os.path.basename(p): p for p in existing_log_files}
        chosen_log_name = st.selectbox("Available Runs", list(log_options.keys()), index=0)
        selected_filename = chosen_log_name
        with open(log_options[chosen_log_name], "r") as fp:
            log_content = fp.read()
    else:
        uploaded_file = st.file_uploader("Upload ALFD Log File (.txt)", type=["txt", "log", "out"])
        if uploaded_file:
            selected_filename = uploaded_file.name
            log_content = uploaded_file.read().decode("utf-8")

# --- MAIN APP LOGIC ---

st.markdown('<div class="dashboard-title">Analytical Lyapunov Function Discovery (ALFD)</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Automated symbolic discovery, verified energy topographies, and reinforcement learning telemetry.</div>', unsafe_allow_html=True)

if not log_content:
    st.info("Select an existing run from the sidebar or upload a new log file to begin analysis.")
    st.stop()

metadata, metrics_df, pqt_df, hof_df, pareto_df = parse_log_file(log_content)

# Resolve default expression
primary_candidate = metadata["discovered_expression"]
if not primary_candidate and hof_df is not None and not hof_df.empty:
    primary_candidate = hof_df.iloc[0]["Expression"]

# Resolve active equation (defaults directly to the discovered result from the run)
active_eq_str = primary_candidate if primary_candidate else ""

with st.sidebar:
    st.markdown("---")
    custom_override = st.text_input("Formula Override (Optional)", placeholder="Leave blank to use discovered V(x)")
    if custom_override.strip():
        active_eq_str = custom_override.strip()

# --- RUN METADATA PANEL ---

is_success = bool(metadata["success"]) or (metrics_df is not None and not metrics_df.empty and metrics_df["Best Reward"].max() > 0.99)
status_badge = '<span class="badge-success">Verified Solution</span>' if is_success else '<span class="badge-warning">Unconverged Search</span>'

st.markdown(f"""
<div class="meta-panel">
    <div class="meta-header-row">
        {status_badge}
        <div class="meta-tags">
            Dataset: <strong>{metadata['dataset']}</strong> &nbsp;|&nbsp;
            Dimension: <strong>{metadata['dimensions']}D</strong> &nbsp;|&nbsp;
            Seed: <strong>{metadata['seed'] if metadata['seed'] is not None else 'N/A'}</strong> &nbsp;|&nbsp;
            Evaluations: <strong>{metadata['total_samples']:,}</strong> &nbsp;|&nbsp;
            Walltime: <strong>{metadata['walltime']:.1f}s</strong>
        </div>
    </div>
    <div class="equation-display">
        V(x) = {active_eq_str if active_eq_str else 'N/A'}
    </div>
</div>
""", unsafe_allow_html=True)

# Performance Indicators
best_reward_val = metrics_df['Best Reward'].max() if metrics_df is not None and not metrics_df.empty else (hof_df.iloc[0]['Reward'] if hof_df is not None and not hof_df.empty else 0.0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Maximum Reward (r_best)", f"{best_reward_val:.5f}")
c2.metric("Test NMSE", f"{metadata['test_nmse']:.3f}" if metadata['test_nmse'] is not None else "N/A")
c3.metric("Cache Hit Rate", f"{metadata['cache_hit_rate']:.1f}%" if metadata['cache_hit_rate'] is not None else "N/A")
c4.metric("Early Stopping", "Triggered" if metadata["early_stopping"] else "Completed All Iterations")

# --- DASHBOARD TABS ---

tab_surface, tab_telemetry, tab_archive, tab_config = st.tabs([
    "State Space Slicing & Topography",
    "Optimization Telemetry",
    "Candidate Archive & Pareto Frontier",
    "Run Configuration & Diagnostics"
])

# ==============================================================================
# TAB 1: STATE SPACE SLICING & TOPOGRAPHY
# ==============================================================================
with tab_surface:
    if not active_eq_str:
        st.warning("No candidate equation available for surface visualization.")
    else:
        clean_expr_str = sanitize_expression(active_eq_str)
        try:
            expr_v = sp.sympify(clean_expr_str)
            
            # State variable resolution
            system_vars = list(metadata["variables"])
            for s in expr_v.free_symbols:
                s_name = str(s)
                if s_name.startswith("x") and s_name not in system_vars:
                    system_vars.append(s_name)
            system_vars = sorted(system_vars, key=lambda it: int(re.sub(r'\D', '', it)) if re.sub(r'\D', '', it) else 0)
            
            # Slicing axis defaults (pick variables active in the candidate)
            active_vars = [v for v in system_vars if v in [str(s) for s in expr_v.free_symbols]]
            x_default = system_vars.index(active_vars[0]) if active_vars else 0
            y_default = system_vars.index(active_vars[1]) if len(active_vars) > 1 else (1 if len(system_vars) > 1 else 0)
            
            st.markdown("##### 2D Slicing Plane Selection")
            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
            with ctrl1:
                x_axis = st.selectbox("X-Axis State", system_vars, index=x_default)
            with ctrl2:
                y_axis = st.selectbox("Y-Axis State", system_vars, index=y_default)
                
            if x_axis == y_axis:
                st.error("X and Y axes must be distinct state variables.")
                st.stop()
                
            bounds = metadata.get("domain_bounds", {})
            x_b = bounds.get(x_axis, (-5.0, 5.0))
            y_b = bounds.get(y_axis, (-5.0, 5.0))
            
            with ctrl3:
                x_min = st.number_input(f"{x_axis} Min", value=float(x_b[0]), step=1.0)
                x_max = st.number_input(f"{x_axis} Max", value=float(x_b[1]), step=1.0)
            with ctrl4:
                y_min = st.number_input(f"{y_axis} Min", value=float(y_b[0]), step=1.0)
                y_max = st.number_input(f"{y_axis} Max", value=float(y_b[1]), step=1.0)

            # Slicing plane fixed values for background variables
            bg_vars = [v for v in system_vars if v not in [x_axis, y_axis]]
            bg_subs = {}
            if bg_vars:
                with st.expander(f"Equilibrium Coordinate Slicing ({len(bg_vars)} background states)", expanded=False):
                    cols = st.columns(min(len(bg_vars), 4))
                    for i, bv in enumerate(bg_vars):
                        b_min, b_max = bounds.get(bv, (-5.0, 5.0))
                        with cols[i % 4]:
                            bg_subs[bv] = st.slider(f"Fixed {bv}", float(b_min), float(b_max), 0.0, 0.1, key=f"sl_{bv}")

            # Substitute background states into SymPy expression
            sub_dict = {sp.Symbol(k): v for k, v in bg_subs.items()}
            for s in expr_v.free_symbols:
                s_name = str(s)
                if s_name not in [x_axis, y_axis] and s_name not in sub_dict:
                    sub_dict[s] = 0.0
                    
            expr_2d = expr_v.subs(sub_dict)
            func_2d = sp.lambdify((sp.Symbol(x_axis), sp.Symbol(y_axis)), expr_2d, modules=['numpy', 'sympy'])

            # Grid computation
            grid_res = 120
            X_lin = np.linspace(x_min, x_max, grid_res)
            Y_lin = np.linspace(y_min, y_max, grid_res)
            X_mesh, Y_mesh = np.meshgrid(X_lin, Y_lin)

            try:
                Z_mesh = func_2d(X_mesh, Y_mesh)
            except Exception:
                Z_mesh = np.zeros_like(X_mesh)
                
            if isinstance(Z_mesh, (int, float)):
                Z_mesh = np.full_like(X_mesh, float(Z_mesh))
            Z_mesh = np.nan_to_num(Z_mesh, nan=0.0, posinf=1e6, neginf=-1e6)

            # Render Slices
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**3D Energy Potential Surface: $V({x_axis}, {y_axis})$**")
                fig_3d = go.Figure(data=[
                    go.Surface(
                        x=X_lin,
                        y=Y_lin,
                        z=Z_mesh,
                        colorscale="Viridis",
                        colorbar=dict(title="V(x)", len=0.75, thickness=16)
                    )
                ])
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        zaxis_title="V(x)",
                        camera=dict(eye=dict(x=1.65, y=-1.65, z=1.2)),
                        bgcolor="#0d1117"
                    ),
                    height=520,
                    margin=dict(l=0, r=0, b=0, t=10),
                    template="plotly_dark",
                    paper_bgcolor="#0d1117"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with p_col2:
                st.markdown(f"**2D Isocontour Map: $V({x_axis}, {y_axis})$**")
                fig_2d = go.Figure(data=[
                    go.Contour(
                        x=X_lin,
                        y=Y_lin,
                        z=Z_mesh,
                        colorscale="Viridis",
                        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                        colorbar=dict(title="V(x)", len=0.75, thickness=16)
                    )
                ])
                fig_2d.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode="markers",
                    marker=dict(symbol="x", color="#f85149", size=12, line=dict(width=2)),
                    name="Equilibrium (Origin)"
                ))
                fig_2d = apply_scientific_layout(fig_2d, "", height=520)
                fig_2d.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig_2d, use_container_width=True)

            # Mathematical Definiteness Summary
            v_origin = float(func_2d(0.0, 0.0)) if hasattr(func_2d, "__call__") else 0.0
            v_min_grid = float(np.min(Z_mesh))
            v_max_grid = float(np.max(Z_mesh))
            
            q1, q2, q3 = st.columns(3)
            q1.metric("Equilibrium Evaluation V(0)", f"{v_origin:.4e}", "Satisfied: V(0)=0" if abs(v_origin) < 1e-4 else "Non-zero Offset")
            q2.metric("Domain Minimum (V_min)", f"{v_min_grid:.4f}", "Positive Definite" if v_min_grid >= -1e-6 else "Violated (Negative Region)")
            q3.metric("Domain Maximum (V_max)", f"{v_max_grid:.4f}")

        except Exception as err:
            st.error(f"Symbolic evaluation error: {err}")

# ==============================================================================
# TAB 2: OPTIMIZATION TELEMETRY
# ==============================================================================
with tab_telemetry:
    if metrics_df is None or metrics_df.empty:
        st.info("No epoch telemetry records available in this log.")
    else:
        st.markdown("##### Reinforcement Learning & Symbolic Search Dynamics")
        
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Best Reward'], mode='lines', name='Best Reward (r_best)', line=dict(color='#3fb950', width=2.5)))
            fig_reward.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Quantile Cutoff'], mode='lines', name='Elite Quantile Threshold', line=dict(color='#d29922', dash='dash')))
            fig_reward.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Batch Raw Mean'], mode='lines', name='Batch Mean Reward', line=dict(color='#58a6ff', dash='dot')))
            fig_reward = apply_scientific_layout(fig_reward, "Reward Progression", height=360)
            fig_reward.update_layout(xaxis_title="Training Epoch", yaxis_title="Reward Score")
            st.plotly_chart(fig_reward, use_container_width=True)

        with row1_c2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Policy Loss'], mode='lines', name='Policy Gradient Loss', line=dict(color='#f85149', width=2)))
            fig_loss.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Invalid (%)'], mode='lines', name='Grammar Invalid (%)', yaxis='y2', line=dict(color='#bc8cff', dash='dot')))
            fig_loss = apply_scientific_layout(fig_loss, "Loss & Grammar Syntax Filter", height=360)
            fig_loss.update_layout(
                xaxis_title="Training Epoch",
                yaxis_title="Policy Loss",
                yaxis2=dict(title="Invalid Expressions (%)", overlaying="y", side="right", range=[0, 100], gridcolor="#161b22")
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            fig_nmse = go.Figure()
            fig_nmse.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['NMSE Test'], mode='lines', name='Normalized MSE (Test)', line=dict(color='#f0883e', width=2)))
            fig_nmse.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['R2 Score'], mode='lines', name='R2 Fit', yaxis='y2', line=dict(color='#3fb950', dash='dash')))
            fig_nmse = apply_scientific_layout(fig_nmse, "Verification Error & Fit Score", height=360)
            fig_nmse.update_layout(
                xaxis_title="Training Epoch",
                yaxis_title="NMSE (log scale)",
                yaxis=dict(type="log", gridcolor="#21262d"),
                yaxis2=dict(title="R2 Coefficient", overlaying="y", side="right", gridcolor="#161b22")
            )
            st.plotly_chart(fig_nmse, use_container_width=True)

        with row2_c2:
            fig_div = go.Figure()
            fig_div.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Unique Equations'], mode='lines', name='Unique Programs Explored', line=dict(color='#58a6ff', width=2)))
            fig_div.add_trace(go.Scatter(x=metrics_df['Epoch'], y=metrics_df['Step Time (s)'], mode='lines', name='Step Latency (s)', yaxis='y2', line=dict(color='#8b949e', dash='dot')))
            fig_div = apply_scientific_layout(fig_div, "Search Diversity & Latency", height=360)
            fig_div.update_layout(
                xaxis_title="Training Epoch",
                yaxis_title="Unique Program Count",
                yaxis2=dict(title="Latency (s/iter)", overlaying="y", side="right", gridcolor="#161b22")
            )
            st.plotly_chart(fig_div, use_container_width=True)

        with st.expander("Tabular Epoch Log"):
            st.dataframe(metrics_df, use_container_width=True)

# ==============================================================================
# TAB 3: CANDIDATE ARCHIVE & PARETO FRONTIER
# ==============================================================================
with tab_archive:
    st.markdown("##### Multi-Objective Complexity vs. Reward Frontier")
    
    col_par, col_pqt = st.columns([1.1, 0.9])
    with col_par:
        if pareto_df is not None and not pareto_df.empty:
            fig_pareto = px.scatter(
                pareto_df,
                x="Complexity",
                y="Reward",
                hover_data=["Expression"],
                text="Rank",
                template="plotly_dark"
            )
            fig_pareto.update_traces(marker=dict(size=12, color="#58a6ff", line=dict(width=1, color="#ffffff")))
            fig_pareto = apply_scientific_layout(fig_pareto, "Pareto Frontier (Expression Complexity vs. Reward)", height=360)
            fig_pareto.update_layout(xaxis_title="Symbolic Complexity (Token Count)", yaxis_title="Reward")
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("No Pareto frontier entries identified in run log.")
            
    with col_pqt:
        if pqt_df is not None and not pqt_df.empty:
            fig_pqt = px.bar(
                pqt_df,
                x="Rank",
                y=["On-Policy Hits", "Off-Policy Hits"],
                barmode="stack",
                color_discrete_sequence=["#3fb950", "#58a6ff"],
                template="plotly_dark"
            )
            fig_pqt = apply_scientific_layout(fig_pqt, "Priority Queue Training (PQT) Buffer Reuse", height=360)
            fig_pqt.update_layout(xaxis_title="Buffer Rank", yaxis_title="Sample Reuse Count")
            st.plotly_chart(fig_pqt, use_container_width=True)
        else:
            st.info("No PQT replay buffer entries recorded.")

    st.markdown("##### Hall of Fame: Elite Discovered Equations")
    if hof_df is not None and not hof_df.empty:
        st.dataframe(hof_df, use_container_width=True)
    else:
        st.info("No Hall of Fame records found.")

# ==============================================================================
# TAB 4: RUN CONFIGURATION & DIAGNOSTICS
# ==============================================================================
with tab_config:
    st.markdown("##### Benchmark Specification & Hyperparameters")
    
    cfg_c1, cfg_c2 = st.columns(2)
    with cfg_c1:
        st.markdown("**State Space Domain Bounds**")
        if metadata.get("domain_bounds"):
            b_data = [{"State": k, "Lower Bound": v[0], "Upper Bound": v[1], "Interval Range": v[1] - v[0]} for k, v in metadata["domain_bounds"].items()]
            st.dataframe(pd.DataFrame(b_data), use_container_width=True)

        st.markdown("**Training Hyperparameter Configuration**")
        if metadata.get("config"):
            st.json(metadata["config"])
        else:
            st.info("No configuration dictionary captured in log header.")

    with cfg_c2:
        st.markdown("**Numerical Stability & Computational Statistics**")
        diag_records = [
            {"Parameter / Metric": "Total Evaluated Programs (n_samples)", "Value": f"{metadata.get('total_samples', 0):,}" if metadata.get('total_samples') else "N/A"},
            {"Parameter / Metric": "Cached Expression Hits (n_cached)", "Value": f"{metadata.get('cached_samples', 0):,}" if metadata.get('cached_samples') else "N/A"},
            {"Parameter / Metric": "Cache Utilization Rate", "Value": f"{metadata.get('cache_hit_rate', 0.0):.2f}%" if metadata.get('cache_hit_rate') is not None else "N/A"},
            {"Parameter / Metric": "Controller Trainable Parameters", "Value": str(metadata.get("parameters", "N/A"))},
            {"Parameter / Metric": "Numerical Overflows (error_node_overflow)", "Value": f"{metadata.get('error_overflow', 0):,}"},
            {"Parameter / Metric": "Numerical Underflows (error_node_underflow)", "Value": f"{metadata.get('error_underflow', 0):,}"},
            {"Parameter / Metric": "Domain Exceptions (error_square)", "Value": f"{metadata.get('error_square', 0):,}"},
            {"Parameter / Metric": "Random Seed", "Value": str(metadata.get("seed", "N/A"))}
        ]
        st.dataframe(pd.DataFrame(diag_records), use_container_width=True)
