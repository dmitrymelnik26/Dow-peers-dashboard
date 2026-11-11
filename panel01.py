
# ============================================================
# DOW AND PEERS RESEARCH
# NOTEBOOK 02 — EXPLORATORY DATA ANALYSIS
# PANEL 01 — HISTORICAL FINANCIAL TIME SERIES
#
# QUESTION:
# How has each company's financial performance evolved
# over time, and what companion metric helps explain it?
#
# METHOD:
# Display actual financial history with paired metrics.
#
# DISCOVERY:
# Identify trends, reversals, volatility and unusual
# quarters BEFORE conducting statistical inference.
#
# SOURCE:
# Validated CSV files produced by Notebook 01.
#
# No simulated data.
# No changes to original financial observations.
# Existing financial ratios are preserved.
# Revenue growth and EV/Sales are companion lines, not separate views.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================



PROJECT = Path(__file__).resolve().parent

RESULTS = PROJECT / "data"

DATA_FILE = RESULTS / "dow_financial_data_wide.csv"
EVENTS_FILE = RESULTS / "dow_structural_events.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

# ============================================================
# 2. LOAD VALIDATED FINANCIAL DATA
# ============================================================

data = pd.read_csv(DATA_FILE)

if "quarter" in data.columns:
    financial = data.set_index("quarter")

elif "metric" in data.columns:
    financial = data.set_index("metric").T

elif "Unnamed: 0" in data.columns:
    first = data.set_index("Unnamed: 0")

    if "dow_revenue" in first.index:
        financial = first.T
    else:
        financial = first

else:
    date_col = next(
        (
            c for c in data.columns
            if str(c).lower()
            in {"date", "period", "quarter_end"}
        ),
        None
    )

    if date_col is None:
        raise ValueError(
            "Cannot identify the quarter column."
        )

    financial = data.set_index(date_col)

financial.index = pd.to_datetime(
    financial.index,
    errors="raise"
)

financial = financial.sort_index()
financial = financial.apply(
    pd.to_numeric,
    errors="coerce"
)

if financial.index.has_duplicates:
    raise ValueError("Duplicate quarters detected.")

if "dow_revenue" not in financial.columns:
    raise ValueError(
        "Expected dow_revenue column is missing."
    )

print("VALIDATED FINANCIAL DATA")
print("Quarters:", len(financial))
print("Metrics:", len(financial.columns))
print("First quarter:", financial.index.min().date())
print("Last quarter:", financial.index.max().date())

# ============================================================
# 3. COMPANY DEFINITIONS
# ============================================================

COMPANIES = {
    "dow": "Dow",
    "lyb": "LyondellBasell",
    "dd": "DuPont",
    "basf": "BASF"
}

# Source currencies are not converted.
# Verify any cross-company currency assumptions separately.

CURRENCY = {
    "dow": "USD",
    "lyb": "USD",
    "dd": "USD",
    "basf": "EUR"
}

# ============================================================
# 4. METRIC DEFINITIONS
# ============================================================

METRIC_LABELS = {
    "revenue": "Revenue",
    "ni": "Net Income",
    "ni_margin": "Net Income Margin",
    "fcf": "Free Cash Flow",
    "fcf_margin": "Free Cash Flow Margin",
    "mkt_cap": "Market Capitalization",
    "ev": "Enterprise Value",
    "total_assets": "Total Assets",
    "equity": "Shareholders' Equity",
    "leverage": "Leverage (Assets / Equity)",
    "roe": "Return on Equity",
    "asset_turnover": "Asset Turnover",
    "debt": "Debt",
    "current_ratio": "Current Ratio",
    "wc": "Working Capital",
    "dio": "Days Inventory Outstanding",
    "dso": "Days Sales Outstanding",
    "dpo": "Days Payables Outstanding",
    "ccc": "Cash Conversion Cycle"
}

# Companion labels are used for orange lines only, not dropdown choices.
COMPANION_LABELS = {
    "rev_q2q": "Revenue QoQ Growth",
    "ni_margin": "Net Income Margin",
    "fcf_margin": "Free Cash Flow Margin",
    "ev_sales_ratio": "EV / Sales",
}

# Deliberately exclude DP_roe.
# The DuPont decomposition belongs in a separate panel.

AMOUNT_METRICS = {
    "revenue", "ni", "fcf", "mkt_cap", "ev",
    "total_assets", "equity", "debt", "wc"
}

PERCENT_METRICS = {
    "rev_q2q", "ni_margin", "fcf_margin", "roe"
}

DAY_METRICS = {"dio", "dso", "dpo", "ccc"}

# ============================================================
# 5. PRESENTATION RULES
# ============================================================

# Explicit companions for financially meaningful pairings.
#
# Revenue -> Revenue QoQ Growth
# Net Income -> Net Income Margin
# FCF -> FCF Margin
# EV -> EV/Sales
#
# All companion values come directly from the validated CSV.

COMPANIONS = {
    "revenue": "rev_q2q",
    "ni": "ni_margin",
    "fcf": "fcf_margin",
    "ev": "ev_sales_ratio"
}

# ROE is displayed alone.
# Its components will be examined in a separate panel.

NO_CHANGE_LINE = {"roe"}

# ============================================================
# 6. OPTIONAL STRUCTURAL EVENTS
# ============================================================

events = pd.DataFrame()

if EVENTS_FILE.exists():
    events = pd.read_csv(EVENTS_FILE)

    if "quarter" in events.columns:
        events["quarter"] = pd.to_datetime(
            events["quarter"],
            errors="coerce"
        )

print("Structural events loaded:", len(events))

# Keep events available for later annotated analysis.
# Do not place unverified event labels on Panel 01.

# ============================================================
# 7. DASHBOARD COLORS
# ============================================================

BACKGROUND = "#071626"
PLOT_BACKGROUND = "#0C2033"

BAR_COLOR = "#4A9BDB"
LINE_COLOR = "#FFAE42"

TEXT_COLOR = "#E9F1F8"
GRID_COLOR = "rgba(185,210,230,0.14)"

# ============================================================
# 8. FINANCIAL FORMATTING
# ============================================================

def display_metric(series, metric, company):
    """
    Convert values for chart display only.

    Amounts:
        Original currency units -> billions

    Percentages:
        Stored fractions -> displayed percentages

    Ratios:
        Original ratio values, unchanged

    Days:
        Original day values, unchanged
    """

    if metric in AMOUNT_METRICS:
        currency = CURRENCY[company]

        return (
            series / 1e9,
            f"{currency} billions",
            "Billions"
        )

    if metric in PERCENT_METRICS:
        return series * 100, "%", "%"

    if metric in DAY_METRICS:
        return series, "days", "Days"

    return series, "ratio units", "Ratio"


# ============================================================
# 9. QUARTERLY CHANGE HELPER
# ============================================================

def quarterly_change(series, metric):
    """
    Calculate changes only for visualization.

    Percentage metrics:
        Percentage-point changes

    Ratios:
        Changes in ratio units

    Days:
        Changes in days

    Monetary amounts:
        Percentage changes, but only when both
        consecutive observations are positive.

    We deliberately avoid interpreting sign-crossing
    financial values as ordinary growth rates.
    """

    if metric in PERCENT_METRICS:
        return (
            series.diff() * 100,
            "Change (percentage points)",
            "Quarterly change"
        )

    if metric in DAY_METRICS:
        return (
            series.diff(),
            "Change (days)",
            "Quarterly change"
        )

    if metric not in AMOUNT_METRICS:
        return (
            series.diff(),
            "Change (ratio units)",
            "Quarterly change"
        )

    previous = series.shift(1)

    change = (
        (series - previous) / previous * 100
    )

    # Growth rates are meaningful here only when
    # both observations are positive.
    valid = (
        (series > 0)
        & (previous > 0)
    )

    change = change.where(valid)
    change.iloc[0] = np.nan

    return (
        change,
        "QoQ change (%)",
        "QoQ change"
    )


# ============================================================
# 10. SELECT THE COMPANION SERIES
# ============================================================

def get_companion(company, metric, source):
    """
    Return:
        Companion values
        Right-axis title
        Line legend label

    Use original Excel-derived series for
    revenue growth, NI margin, FCF margin,
    and EV/Sales.

    Do not recompute these ratios.
    """

    if metric in NO_CHANGE_LINE:
        return None, "", ""

    if metric in COMPANIONS:

        companion_metric = COMPANIONS[metric]
        companion_col = (
            f"{company}_{companion_metric}"
        )

        if companion_col not in financial.columns:
            return None, "", ""

        companion = financial[
            companion_col
        ].copy()

        # Revenue growth and margins are stored
        # as fractional values in the validated data.
        if companion_metric in PERCENT_METRICS:
            companion = companion * 100
            axis_title = (
                COMPANION_LABELS[companion_metric] + " (%)"
            )

        else:
            axis_title = (
                COMPANION_LABELS[companion_metric]
                + " (ratio)"
            )

        # First revenue growth observation is excluded
        # because its comparison quarter is outside
        # the validated historical window.
        if metric == "revenue":
            companion.iloc[0] = np.nan

        return (
            companion,
            axis_title,
            COMPANION_LABELS[companion_metric]
        )

    return quarterly_change(source, metric)


# ============================================================
# 11. BUILD COMPANY / METRIC SELECTIONS
# ============================================================

combinations = []

for company, company_name in COMPANIES.items():

    for metric, label in METRIC_LABELS.items():

        column = f"{company}_{metric}"

        if column not in financial.columns:
            continue

        if financial[column].notna().any():
            combinations.append(
                (company, metric, column)
            )

if not combinations:
    raise ValueError("No plottable data found.")

print(
    "Available company/metric combinations:",
    len(combinations)
)

# ============================================================
# 12. BUILD INTERACTIVE PLOTLY CHART
# ============================================================

fig = go.Figure()
buttons = []

for i, (company, metric, column) in enumerate(
    combinations
):

    company_name = COMPANIES[company]
    metric_name = METRIC_LABELS[metric]

    source = financial[column].copy()

    level, unit, short_unit = display_metric(
        source, metric, company
    )

    companion, companion_axis, companion_name = (
        get_companion(company, metric, source)
    )

    initially_visible = (i == 0)

    # --------------------------------------------------------
    # BLUE BARS — HISTORICAL FINANCIAL LEVEL
    # --------------------------------------------------------

    fig.add_trace(
        go.Bar(
            x=financial.index,
            y=level,
            name=metric_name,
            marker_color=BAR_COLOR,
            opacity=0.94,
            visible=initially_visible,
            hovertemplate=(
                "Quarter ending: %{x|%Y-%m-%d}"
                "<br>Value: %{y:,.3f}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # ORANGE LINE — COMPANION METRIC
    # --------------------------------------------------------

    has_companion = companion is not None

    if not has_companion:
        companion = pd.Series(
            np.nan,
            index=financial.index
        )

    fig.add_trace(
        go.Scatter(
            x=financial.index,
            y=companion,
            mode="lines+markers",
            name=companion_name or "Not applicable",
            line=dict(
                color=LINE_COLOR,
                width=2.5
            ),
            marker=dict(size=5),
            yaxis="y2",
            visible=(
                initially_visible and has_companion
            ),
            hovertemplate=(
                "Quarter ending: %{x|%Y-%m-%d}"
                "<br>Value: %{y:,.3f}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # DROPDOWN VISIBILITY
    # --------------------------------------------------------

    visibility = [False] * (
        2 * len(combinations)
    )

    visibility[2 * i] = True

    visibility[2 * i + 1] = has_companion

    # --------------------------------------------------------
    # RIGHT AXIS
    # --------------------------------------------------------

    # Hide the second axis when no companion exists.
    right_axis_visible = has_companion

    # --------------------------------------------------------
    # DROPDOWN BUTTON
    # --------------------------------------------------------

    buttons.append(
        dict(
            label=(
                f"{company_name} — {metric_name}"
            ),
            method="update",
            args=[
                {
                    "visible": visibility
                },
                {
                    "title.text": (
                        f"{company_name} — {metric_name}"
                    ),
                    "yaxis.title.text": (
                        f"{metric_name} ({unit})"
                    ),
                    "yaxis2.title.text": (
                        companion_axis
                    ),
                    "yaxis2.visible": (
                        right_axis_visible
                    )
                }
            ]
        )
    )

# ============================================================
# 13. INITIAL AXIS SETTINGS
# ============================================================

first_company, first_metric, first_column = (
    combinations[0]
)

first_level, first_unit, _ = display_metric(
    financial[first_column],
    first_metric,
    first_company
)

first_companion, first_right_axis, _ = (
    get_companion(
        first_company,
        first_metric,
        financial[first_column]
    )
)

# ============================================================
# 14. DASHBOARD STYLING
# ============================================================

fig.update_layout(

    title=dict(
        text=(
            f"{COMPANIES[first_company]} — "
            f"{METRIC_LABELS[first_metric]}"
        ),
        x=0.52,
        xanchor="center",
        y=0.97,
        font=dict(
            size=23,
            color=TEXT_COLOR
        )
    ),

    paper_bgcolor=BACKGROUND,
    plot_bgcolor=PLOT_BACKGROUND,

    font=dict(
        family="Arial",
        color=TEXT_COLOR,
        size=12
    ),

    height=650,

    margin=dict(
        l=85,
        r=65,
        t=160,
        b=190
    ),

    barmode="group",

    legend=dict(
        orientation="h",
        x=0.02,
        xanchor="left",
        y=1.08,
        yanchor="bottom",
        bgcolor="rgba(0,0,0,0)"
    ),

    xaxis=dict(
        title="Quarter",
        showgrid=False,
        tickformat="%Y",
        tickfont=dict(color=TEXT_COLOR),
        linecolor=GRID_COLOR
    ),

    yaxis=dict(
        title=(
            f"{METRIC_LABELS[first_metric]} "
            f"({first_unit})"
        ),
        gridcolor=GRID_COLOR,
        zeroline=True,
        zerolinecolor=(
            "rgba(230,240,250,0.45)"
        ),
        tickfont=dict(color=TEXT_COLOR)
    ),

    yaxis2=dict(
        title=first_right_axis,
        overlaying="y",
        side="right",
        visible=first_companion is not None,
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=TEXT_COLOR)
    ),

    hovermode="x unified",

    # --------------------------------------------------------
    # DROPDOWN — LIGHT BACKGROUND / DARK TEXT
    # --------------------------------------------------------

    updatemenus=[
        dict(
            type="dropdown",
            buttons=buttons,
            direction="down",
            showactive=True,
            active=0,

            x=0,
            y=1.48,
            xanchor="left",
            yanchor="top",

            bgcolor="#E8EEF5",
            bordercolor="#8FA4B8",
            borderwidth=1,

            font=dict(
                color="#152638",
                size=12
            ),

            pad=dict(
                l=8,
                r=8,
                t=6,
                b=6
            )
        )
    ],

    annotations=[
        dict(
            text=(
                "SELECT COMPANY AND FINANCIAL METRIC"
            ),
            x=0,
            y=1.39,
            xref="paper",
            yref="paper",
            xanchor="left",
            showarrow=False,
            font=dict(
                size=11,
                color="#AFC6DA"
            )
        )
    ]
)

# ============================================================
# 15. RENDER PANEL 01 IN GOOGLE COLAB
# ============================================================

# Chart rendered by Streamlit.

# ============================================================
# 16. VERIFICATION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EDA PANEL 01 — UPDATED HISTORICAL TIME SERIES")
print("=" * 65)

print("Historical quarters:", len(financial))
print("Available selections:", len(combinations))
print("Revenue QoQ Growth and EV/Sales appear as companion lines only.")

print("\nSPECIAL FINANCIAL PAIRINGS:")
print("Revenue       -> Revenue QoQ Growth")
print("Net Income    -> Net Income Margin")
print("Free Cash Flow-> FCF Margin")
print("Enterprise Val-> EV / Sales")
print("Market Cap    -> QoQ Change")
print("Leverage      -> Change in ratio units")
print("ROE           -> ROE alone")

print("\nDATA INTEGRITY:")
print("Original financial values preserved.")
print("Original margin and valuation ratios preserved.")
print("No simulated financial observations.")
print("No original source values overwritten.")
print("DuPont decomposition reserved for separate panel.")

print("=" * 65)



