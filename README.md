# Dow & Peers — Streamlit prototype

Port of three interactive Plotly research panels from `Dow_02_EDA.ipynb` by Dmitry Melnik

Independent quantitative research examining financial performance, profitability, return on equity, statistical distributions, and uncertainty across Dow, LyondellBasell, DuPont, and BASF.

Live interactive dashboard:
https://dow-peers-dashboard-qwf2gum2xgpkfpsusxbvnk.streamlit.app/

Research methodology and source code:
https://github.com/dmitrymelnik26/Dow-peers-dashboard

Historical financial analysis only. Not investment advice.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, e.g. `dow-peers-dashboard`.
2. Upload **all files and the `data/` directory** from this package to the repository root.
3. At https://share.streamlit.io select **Create app**, select your GitHub repository, branch `main`, and entry point `app.py`.
4. Deploy. The app reads bundled CSVs and requires no Google Drive access or secrets.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

- `app.py`: Streamlit sidebar and navigation.
- `panel01.py`: historical financial performance, preserving original Plotly controls.
- `panel02.py`: five DuPont/ROE views, preserving original Plotly controls.
- `panel03.py`: distribution analysis, controlled through Streamlit selectors.
- `data/dow_financial_data_wide.csv`: validated source data.
- `data/dow_structural_events.csv`: structural event annotations.

The original data are not modified. Historical descriptions and fitted distributions are not forecasts. Source CSVs are bundled in the public repository; do not include confidential information.
