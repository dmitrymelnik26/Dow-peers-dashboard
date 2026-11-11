

# ================================================================
# DOW & PEERS — PANEL 03
# STATISTICAL DISTRIBUTIONS AND TAIL BEHAVIOR
#
# INPUT:
# 03_Results/dow_financial_data_wide.csv
#
# DATA STRUCTURE:
# Rows    = company_metric identifiers
# Columns = 31 quarter-end dates (2019Q1–2026Q3)
#
# RESEARCH OBJECTIVE:
# Study the historical distributions of financial
# performance and evaluate distributional assumptions.
#
# IMPORTANT:
# - Uses original source ratios.
# - No financial observations are simulated.
# - No financial ratios are recalculated.
# - No outliers are automatically removed.
# - Source files remain unchanged.
# ================================================================

# ================================================================
# SECTION 0 — LIBRARIES
# ================================================================

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import display

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)

# ================================================================
# SECTION 1 — READ VALIDATED DATA
# ================================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT
    / "data"
    / "dow_financial_data_wide.csv"
)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Source file not found: {DATA_FILE}"
    )

raw = pd.read_csv(DATA_FILE)

print("SOURCE:", DATA_FILE.name)
print("SOURCE DIMENSIONS:", raw.shape)

if "metric" not in raw.columns:
    raise ValueError(
        "Expected a 'metric' identifier column."
    )

# Preserve the original source values.
# No overwriting or ratio calculations occur.

raw["metric"] = (
    raw["metric"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if raw["metric"].duplicated().any():
    raise ValueError(
        "Duplicate metric identifiers in source CSV."
    )

# Identify actual quarterly date columns.

quarter_columns = []

for column in raw.columns:

    if column == "metric":
        continue

    parsed = pd.to_datetime(
        column,
        errors="coerce"
    )

    if pd.notna(parsed):
        quarter_columns.append(column)

quarter_columns = sorted(
    quarter_columns,
    key=lambda c: pd.Timestamp(c)
)

if not quarter_columns:
    raise ValueError(
        "No quarterly date columns detected."
    )

print(
    "QUARTERS:",
    len(quarter_columns)
)

print(
    "PERIOD:",
    quarter_columns[0],
    "to",
    quarter_columns[-1]
)

# ================================================================
# SECTION 2 — TRANSFORM WIDE DATA TO LONG DATA
# ================================================================
#
# Example:
#
# dow_ev_sales_ratio | 0.945 | 0.933 | ...
#
# becomes:
#
# company = Dow
# metric  = ev_sales_ratio
# quarter = 2019Q1
# value   = 0.945
#
# This is a structural transformation only.
# It does not change the financial values.
# ================================================================

long = raw.melt(
    id_vars="metric",
    value_vars=quarter_columns,
    var_name="quarter_end",
    value_name="value"
)

COMPANY_PREFIXES = {
    "dow": "Dow",
    "lyb": "LyondellBasell",
    "dd": "DuPont",
    "basf": "BASF"
}

# Allow alternate prefixes if present.

COMPANY_PREFIXES.update({
    "dupont": "DuPont",
    "lyondellbasell": "LyondellBasell"
})

def split_identifier(identifier):

    for prefix in sorted(
        COMPANY_PREFIXES,
        key=len,
        reverse=True
    ):

        if identifier.startswith(prefix + "_"):

            return (
                COMPANY_PREFIXES[prefix],
                identifier[len(prefix) + 1:]
            )

    return (None, identifier)

parsed = long["metric"].apply(
    split_identifier
)

long["company"] = parsed.str[0]
long["metric_code"] = parsed.str[1]

long["quarter_end"] = pd.to_datetime(
    long["quarter_end"]
)

long["quarter"] = (
    long["quarter_end"]
    .dt.to_period("Q")
    .astype(str)
)

long["value"] = pd.to_numeric(
    long["value"],
    errors="coerce"
)

# Do not silently drop unknown companies.

unknown = long.loc[
    long["company"].isna(),
    "metric"
].unique()

if len(unknown) > 0:
    print(
        "\nUnrecognized identifiers:",
        unknown.tolist()
    )

long = long.dropna(
    subset=["company"]
).copy()

# ================================================================
# SECTION 3 — METRIC CONFIGURATION
# ================================================================
#
# These are SOURCE metric codes, not calculated ratios.
#
# The aliases below allow the program to recognize
# common naming variations.
#
# Unknown codes are printed in the audit.
# ================================================================

METRICS = {

    "Net income margin": [
        "ni_margin",
        "net_income_margin",
        "net_margin",
        "profit_margin"
    ],

    "Free cash flow margin": [
        "fcf_margin",
        "free_cash_flow_margin"
    ],

    "Return on equity": [
        "roe",
        "return_on_equity",
        "quarterly_roe"
    ],

    "Asset turnover": [
        "asset_turnover",
        "asset_turnover_ratio"
    ],

    "Financial leverage": [
        "financial_leverage",
        "equity_multiplier",
        "leverage",
        "assets_to_equity"
    ],

    "Revenue QoQ growth": [
        "rev_q2q",
        "revenue_qoq_growth",
        "revenue_growth_qoq",
        "rev_qoq",
        "revenue_growth"
    ],

    "EV / Sales": [
        "ev_sales_ratio",
        "ev_sales",
        "ev_to_sales",
        "ev_revenue"
    ]
}

PERCENT_METRICS = {
    "Net income margin",
    "Free cash flow margin",
    "Return on equity",
    "Revenue QoQ growth"
}

# IMPORTANT:
# Original percentage ratios are assumed stored
# as decimals (0.05 = 5%), consistent with the
# displayed rev_q2q source observations.
#
# Set to False only if source values are
# already expressed as percentage points.

PERCENTAGES_ARE_DECIMALS = True

# Match each requested metric to a source code.

available_codes = set(
    long["metric_code"].unique()
)

metric_map = {}

for label, aliases in METRICS.items():

    matches = [
        code for code in aliases
        if code in available_codes
    ]

    if matches:
        metric_map[label] = matches[0]

print("\nRECOGNIZED METRICS:")

for label, code in metric_map.items():
    print(f"  {label:25s} -> {code}")

print("\nALL SOURCE METRIC CODES:")

for code in sorted(available_codes):
    print(" ", code)

missing_metrics = [
    label for label in METRICS
    if label not in metric_map
]

if missing_metrics:
    print("\nUNMATCHED REQUESTED METRICS:")
    for label in missing_metrics:
        print(" ", label)

if not metric_map:
    raise ValueError(
        "No requested financial metrics matched. "
        "Inspect the source codes printed above "
        "and update METRICS aliases."
    )

# ================================================================
# SECTION 4 — EXTRACT FINANCIAL OBSERVATIONS
# ================================================================

def get_observations(company, metric):

    code = metric_map[metric]

    subset = long.loc[
        (long["company"] == company)
        & (long["metric_code"] == code),
        [
            "quarter",
            "quarter_end",
            "value"
        ]
    ].copy()

    subset = subset.sort_values(
        "quarter_end"
    )

    # Preserve missing values in source.
    # Exclude nonfinite values only from
    # statistical estimation.

    subset = subset.loc[
        np.isfinite(subset["value"])
    ].copy()

    if (
        metric in PERCENT_METRICS
        and PERCENTAGES_ARE_DECIMALS
    ):
        subset["display_value"] = (
            subset["value"] * 100.0
        )
    else:
        subset["display_value"] = (
            subset["value"]
        )

    return subset.reset_index(drop=True)

# ================================================================
# SECTION 5 — DESCRIPTIVE STATISTICS
# ================================================================

def describe_distribution(x):

    x = np.asarray(x, dtype=float)

    n = len(x)

    if n == 0:
        return {}

    return {
        "Observations": n,
        "Mean": np.mean(x),
        "Median": np.median(x),
        "Sample SD": (
            np.std(x, ddof=1)
            if n > 1 else np.nan
        ),
        "Skewness": (
            stats.skew(x, bias=False)
            if n > 2 and np.std(x) > 0
            else np.nan
        ),
        "Excess kurtosis": (
            stats.kurtosis(
                x,
                fisher=True,
                bias=False
            )
            if n > 3 and np.std(x) > 0
            else np.nan
        ),
        "Minimum": np.min(x),
        "5th percentile": np.percentile(x, 5),
        "25th percentile": np.percentile(x, 25),
        "75th percentile": np.percentile(x, 75),
        "95th percentile": np.percentile(x, 95),
        "Maximum": np.max(x)
    }

# ================================================================
# SECTION 6 — FIT PROBABILITY DISTRIBUTIONS
# ================================================================
#
# KDE:
# Smooth, nonparametric density estimate.
#
# NORMAL:
# One Gaussian distribution fitted to all data.
#
# STUDENT'S t:
# Location-scale t distribution with fitted
# degrees of freedom, location, and scale.
#
# All models use observed quarterly values.
# ================================================================

def fit_models(x, bandwidth=1.0):

    x = np.asarray(x, dtype=float)

    if len(x) < 5 or np.std(x) == 0:
        return None

    mu, sigma = stats.norm.fit(x)

    result = {
        "normal": (mu, sigma)
    }

    # KDE using Scott's bandwidth multiplied
    # by a user-selectable sensitivity factor.

    try:

        kde = stats.gaussian_kde(
            x,
            bw_method=lambda obj:
                obj.scotts_factor() * bandwidth
        )

        result["kde"] = kde

    except Exception:
        result["kde"] = None

    # Fit Student's t distribution.

    try:

        df_t, loc_t, scale_t = stats.t.fit(x)

        if (
            df_t > 0
            and scale_t > 0
            and np.all(
                np.isfinite(
                    [df_t, loc_t, scale_t]
                )
            )
        ):
            result["t"] = (
                df_t,
                loc_t,
                scale_t
            )

    except Exception:
        pass

    return result

# ================================================================
# SECTION 7 — VISUAL DESIGN
# ================================================================

BG = "#071827"
PANEL = "#10283B"
GRID = "#30465A"
TEXT = "#EAF1F8"
MUTED = "#A8B9C9"

COMPANY_COLORS = {
    "Dow": "#50A7E8",
    "LyondellBasell": "#F0AA52",
    "DuPont": "#35C2A0",
    "BASF": "#B58BE8"
}

KDE_COLOR = "#38D9B0"
NORMAL_COLOR = "#FF737D"
T_COLOR = "#C79AFF"

# ================================================================
# SECTION 8 — BUILD THE DASHBOARD
# ================================================================

def build_panel03(
    company,
    metric,
    show_kde=True,
    show_normal=True,
    show_t=True,
    bandwidth=1.0,
    qq_reference="Normal"
):

    data = get_observations(
        company,
        metric
    )

    x = data["display_value"].to_numpy()

    if len(x) < 5:
        raise ValueError(
            "At least five usable observations "
            "are required for this view."
        )

    desc = describe_distribution(x)

    fitted = fit_models(
        x,
        bandwidth=bandwidth
    )

    if fitted is None:
        raise ValueError(
            "Unable to fit distributions: "
            "insufficient variation."
        )

    unit = (
        "percentage points"
        if metric in PERCENT_METRICS
        else "ratio"
    )

    color = COMPANY_COLORS.get(
        company,
        "#50A7E8"
    )

    # --------------------------------------------
    # DENSITY EVALUATION GRID
    # --------------------------------------------

    spread = np.std(x, ddof=1)

    lower = np.min(x) - 0.4 * spread
    upper = np.max(x) + 0.4 * spread

    grid = np.linspace(
        lower,
        upper,
        700
    )

    # --------------------------------------------
    # CREATE FOUR-PANEL FIGURE
    # --------------------------------------------

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [
                {"type": "xy"},
                {"type": "table"}
            ],
            [
                {"type": "xy"},
                {"type": "xy"}
            ]
        ],
        subplot_titles=[
            "03A | OBSERVED DISTRIBUTION",
            "03B | DESCRIPTIVE STATISTICS",
            "03C | EMPIRICAL CDF",
            "03D | Q-Q DIAGNOSTIC"
        ],
        column_widths=[0.60, 0.40],
        row_heights=[0.54, 0.46],
        horizontal_spacing=0.11,
        vertical_spacing=0.19
    )

    # ============================================
    # 03A — HISTOGRAM
    # ============================================

    fig.add_trace(
        go.Histogram(
            x=x,
            histnorm="probability density",
            nbinsx=max(
                5,
                int(np.ceil(np.sqrt(len(x))))
            ),
            name="Observed quarters",
            marker=dict(
                color=color,
                line=dict(
                    color=BG,
                    width=1
                )
            ),
            opacity=0.65
        ),
        row=1,
        col=1
    )

    # KDE curve

    if show_kde and fitted.get("kde") is not None:

        fig.add_trace(
            go.Scatter(
                x=grid,
                y=fitted["kde"](grid),
                mode="lines",
                name="KDE",
                line=dict(
                    color=KDE_COLOR,
                    width=3
                )
            ),
            row=1,
            col=1
        )

    # Normal curve

    mu, sigma = fitted["normal"]

    if show_normal:

        fig.add_trace(
            go.Scatter(
                x=grid,
                y=stats.norm.pdf(
                    grid,
                    mu,
                    sigma
                ),
                mode="lines",
                name="Fitted normal",
                line=dict(
                    color=NORMAL_COLOR,
                    width=2.5,
                    dash="dash"
                )
            ),
            row=1,
            col=1
        )

    # Student's t curve

    if show_t and "t" in fitted:

        df_t, loc_t, scale_t = fitted["t"]

        fig.add_trace(
            go.Scatter(
                x=grid,
                y=stats.t.pdf(
                    grid,
                    df_t,
                    loc_t,
                    scale_t
                ),
                mode="lines",
                name="Fitted Student t",
                line=dict(
                    color=T_COLOR,
                    width=2.5,
                    dash="dot"
                )
            ),
            row=1,
            col=1
        )

    # --------------------------------------------
    # Mark zero when economically meaningful.
    # --------------------------------------------

    if metric in PERCENT_METRICS:

        fig.add_vline(
            x=0,
            line_color="#B5C1CC",
            line_dash="dot",
            opacity=0.55,
            row=1,
            col=1
        )

    # ============================================
    # 03B — DESCRIPTIVE STATISTICS
    # ============================================

    stat_names = []
    stat_values = []

    for key, value in desc.items():

        stat_names.append(key)

        if key == "Observations":
            stat_values.append(str(int(value)))

        elif pd.isna(value):
            stat_values.append("N/A")

        else:
            stat_values.append(
                f"{value:,.3f}"
            )

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "STATISTIC",
                    "VALUE"
                ],
                fill_color="#1C3B52",
                align="left",
                font=dict(
                    color=TEXT,
                    size=12
                ),
                height=28
            ),
            cells=dict(
                values=[
                    stat_names,
                    stat_values
                ],
                fill_color=PANEL,
                align="left",
                font=dict(
                    color=TEXT,
                    size=11
                ),
                height=24
            )
        ),
        row=1,
        col=2
    )

    # ============================================
    # 03C — EMPIRICAL CDF
    # ============================================
    #
    # The ECDF is constructed directly from
    # observations, without fitting a distribution.
    #
    # At value v, it reports the fraction of
    # observed quarters at or below v.
    # ============================================

    ordered = np.sort(x)

    cumulative = (
        np.arange(1, len(x) + 1)
        / len(x)
    )

    fig.add_trace(
        go.Scatter(
            x=ordered,
            y=cumulative,
            mode="lines+markers",
            line=dict(
                color=KDE_COLOR,
                width=2.5,
                shape="hv"
            ),
            marker=dict(size=5),
            name="Empirical CDF",
            showlegend=False,
            hovertemplate=(
                "Value: %{x:.3f}<br>"
                "Observed fraction: %{y:.1%}"
                "<extra></extra>"
            )
        ),
        row=2,
        col=1
    )

    # ============================================
    # 03D — Q-Q DIAGNOSTIC
    # ============================================
    #
    # Compare observed quantiles against quantiles
    # implied by the selected fitted distribution.
    #
    # A straight-line pattern supports compatibility.
    # Systematic curvature indicates mismatch.
    #
    # This is descriptive, not a formal hypothesis
    # test of distributional assumptions.
    # ============================================

    n = len(x)

    probabilities = (
        np.arange(1, n + 1) - 0.5
    ) / n

    observed = np.sort(x)

    if (
        qq_reference == "Student t"
        and "t" in fitted
    ):

        df_t, loc_t, scale_t = fitted["t"]

        theoretical = stats.t.ppf(
            probabilities,
            df_t,
            loc=loc_t,
            scale=scale_t
        )

        reference_label = "Student t"

    else:

        theoretical = stats.norm.ppf(
            probabilities,
            loc=mu,
            scale=sigma
        )

        reference_label = "Normal"

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=observed,
            mode="markers",
            name="Observed quantiles",
            showlegend=False,
            marker=dict(
                color=color,
                size=8
            ),
            hovertemplate=(
                "Model quantile: %{x:.3f}<br>"
                "Observed quantile: %{y:.3f}"
                "<extra></extra>"
            )
        ),
        row=2,
        col=2
    )

    qmin = min(
        np.min(theoretical),
        np.min(observed)
    )

    qmax = max(
        np.max(theoretical),
        np.max(observed)
    )

    fig.add_trace(
        go.Scatter(
            x=[qmin, qmax],
            y=[qmin, qmax],
            mode="lines",
            name="Equality reference",
            showlegend=False,
            line=dict(
                color=NORMAL_COLOR,
                width=2,
                dash="dash"
            )
        ),
        row=2,
        col=2
    )

    # ============================================
    # FORMATTING
    # ============================================

    fig.update_layout(
        title=dict(
            text=(
                "PANEL 03 | STATISTICAL DISTRIBUTIONS"
                f"<br><sup>{company} · {metric} · "
                f"{len(x)} historical quarters</sup>"
            ),
            x=0.5,
            font=dict(
                color=TEXT,
                size=23
            )
        ),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(
            family="Arial",
            color=TEXT,
            size=12
        ),
        width=1550,
        height=1000,
        margin=dict(
            l=85,
            r=65,
            t=160,
            b=110
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=1.11
        ),
        bargap=0.08,
        hovermode="closest"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    fig.update_xaxes(
        title_text=f"{metric} ({unit})",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Probability density",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text=f"{metric} ({unit})",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Observed cumulative fraction",
        tickformat=".0%",
        range=[0, 1.05],
        row=2,
        col=1
    )

    fig.update_xaxes(
        title_text=(
            f"Fitted {reference_label} quantiles"
        ),
        row=2,
        col=2
    )

    fig.update_yaxes(
        title_text="Observed quantiles",
        row=2,
        col=2
    )

    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "Historical descriptive analysis only. "
            "Fitted densities are not calibrated forecasts. "
            "Tail estimates are sensitive to sample size, "
            "bandwidth, structural changes, and unusual quarters."
        ),
        font=dict(
            color=MUTED,
            size=11
        )
    )

    return fig

# ================================================================
# SECTION 9 — SINGLE INTERACTIVE PLOTLY DASHBOARD
# ================================================================
# Colab does not reliably display Plotly charts inside an
# ipywidgets.Output callback. Instead, build each company/metric
# view ONCE, then use Plotly's own dropdown to show only that view.
# No Python callback is needed after the figure is displayed.
#
# IMPORTANT: Plotly's native dropdowns do not maintain shared state
# across TWO independent menus. For reliability, one combined menu
# offers every valid company/metric pair (up to 28 choices).
# The selection updates all four panels and the statistics table.
# ================================================================

import copy
import plotly.io as pio
# Streamlit selects the renderer automatically.

COMPANY_ORDER = ["Dow", "LyondellBasell", "DuPont", "BASF"]
companies = [c for c in COMPANY_ORDER if c in set(long["company"].dropna())]

