"""Dow and Peers — Streamlit demonstration of existing Colab research.

All financial observations originate in the validated CSV files exported
from the original workbook. No source data are changed by this application.
"""
import streamlit as st

st.set_page_config(page_title="Dow & Peers | Financial Research", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp {background-color:#071626;color:#E9F1F8;}
section[data-testid="stSidebar"] {background-color:#0C2033;}
.block-container {padding-top:1.8rem;max-width:1600px;}
[data-testid="stMetricValue"] {color:#4A9BDB;}
</style>""", unsafe_allow_html=True)

st.sidebar.title("DOW & PEERS")
st.sidebar.caption("Interactive financial research · 2019 Q1–2026 Q3")
page = st.sidebar.radio("Navigate", ["Overview", "01 · Historical performance", "02 · ROE & DuPont", "03 · Statistical distributions"])
st.sidebar.divider()
st.sidebar.caption("Research source: validated quarterly financial CSV. Historical analysis only; not investment advice.")

st.title("Dow & Peers | Financial Research")
st.caption("Dow · LyondellBasell · DuPont · BASF  |  31 historical quarters  |  Prototype deployed from Google Colab research")

if page == "Overview":
    st.subheader("From research notebook to public dashboard")
    st.write("This prototype carries three completed exploratory research panels from Google Colab into Streamlit. Select a panel in the left sidebar to explore the original interactive Plotly visualizations.")
    a,b,c=st.columns(3)
    a.metric("Companies", "4")
    b.metric("Historical quarters", "31")
    c.metric("Analysis panels", "3")
    st.info("Original ratios and observations are preserved. Structural events and small-sample limitations matter when interpreting historical distributions.")
    st.markdown("**Panel 01:** Financial time series and companion measures.  \n**Panel 02:** DuPont ROE decomposition, exact Shapley bridges, and sensitivity analysis.  \n**Panel 03:** Historical distributions, KDE, fitted normal/Student t, empirical CDF and Q–Q diagnostics.")

elif page == "01 · Historical performance":
    import panel01 as p
    st.subheader("01 | What actually happened?")
    st.write("Select the company and financial metric using the dropdown within the Plotly chart. The companion metric updates with the primary series.")
    st.plotly_chart(p.fig, use_container_width=True, config={"displaylogo":False})
    with st.expander("Structural event annotations"):
        st.dataframe(p.events if hasattr(p,'events') else __import__('pandas').read_csv(p.EVENTS_FILE),use_container_width=True)

elif page == "02 · ROE & DuPont":
    import panel02 as p
    st.subheader("02 | What drives return on equity?")
    options={"02A · Peer comparison":p.fig_a,"02B · ROE factor evolution":p.fig_b,"02C · ROE change bridge":p.fig_c,"02D · Mathematical sensitivity":p.fig_d,"02E · Peer gap bridge":p.fig_e}
    selection=st.selectbox("Choose a DuPont analysis",list(options))
    st.plotly_chart(options[selection],use_container_width=True,config={"displaylogo":False})
    st.caption("DuPont factor decompositions and Shapley bridges describe accounting identities and mathematical contributions; they do not establish causal effects or forecasts.")

elif page == "03 · Statistical distributions":
    import panel03 as p
    st.subheader("03 | What do the historical observations look like?")
    left,right=st.columns([1,2])
    with left: company=st.selectbox("Company",p.companies)
    with right: metric=st.selectbox("Financial metric",list(p.metric_map))
    with st.expander("Distribution settings",expanded=False):
        c1,c2,c3=st.columns(3)
        with c1: show_kde=st.checkbox("KDE",value=True)
        with c2: show_normal=st.checkbox("Fitted normal",value=True)
        with c3: show_t=st.checkbox("Fitted Student t",value=True)
        bandwidth=st.slider("KDE bandwidth multiplier",0.4,2.5,1.0,0.1)
        qq_reference=st.selectbox("Q–Q reference",["Normal","Student t"])
    fig=p.build_panel03(company=company,metric=metric,show_kde=show_kde,show_normal=show_normal,show_t=show_t,bandwidth=bandwidth,qq_reference=qq_reference)
    fig.update_layout(title=dict(text=f"PANEL 03 | STATISTICAL DISTRIBUTIONS<br><sup>{company} · {metric} · {len(p.get_observations(company,metric))} historical quarters</sup>"),updatemenus=[],margin=dict(l=85,r=65,t=180,b=115))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.caption("Descriptive historical distributions are not calibrated probabilities of future financial outcomes. Extreme observations are retained; 31 quarters is a limited sample.")
