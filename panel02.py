

# ================================================================
# DOW AND PEERS | NOTEBOOK 02 | PANEL 02 — DUPONT ROE ANALYSIS
# Replace the ENTIRE existing Panel 02 code cell with this script.
# Source: validated CSV from Notebook 01; no simulated observations.
# ================================================================
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1 — SOURCE AND CONFIGURATION

PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / 'data' / 'dow_financial_data_wide.csv'
if not DATA_FILE.is_file():
    raise FileNotFoundError(f'Validated source not found: {DATA_FILE}')
raw = pd.read_csv(DATA_FILE)
if 'quarter' in raw.columns:
    fin = raw.set_index('quarter')
elif 'metric' in raw.columns:
    fin = raw.set_index('metric').T
elif 'Unnamed: 0' in raw.columns:
    t = raw.set_index('Unnamed: 0')
    fin = t.T if 'dow_revenue' in t.index else t
else:
    raise ValueError(f'Unrecognized CSV layout. First columns: {raw.columns[:8].tolist()}')
fin.index = pd.to_datetime(fin.index, errors='raise')
fin = fin.sort_index().apply(pd.to_numeric, errors='coerce')
if fin.index.has_duplicates:
    raise ValueError('Duplicate quarter dates detected')

COMPANIES = {'dow':'Dow', 'lyb':'LyondellBasell', 'dd':'DuPont', 'basf':'BASF'}
COLORS = {'dow':'#4A9BDB', 'lyb':'#F2A541', 'dd':'#37B58A', 'basf':'#AE83E8'}
FACTOR_COLORS = {'margin':'#F16C71','turnover':'#4A9BDB','leverage':'#37B58A'}
BG, PANEL, FG, GRID = '#071626','#0C2033','#E9F1F8','rgba(185,210,230,0.16)'
FACTORS = ('margin','turnover','leverage')
REQUIRED = ('revenue','ni','total_assets','equity','ni_margin','asset_turnover','leverage','roe')
missing = [f'{c}_{m}' for c in COMPANIES for m in REQUIRED if f'{c}_{m}' not in fin.columns]
if missing:
    raise ValueError('Missing required validated fields: '+', '.join(missing))
if len(fin) < 9:
    raise ValueError('At least nine quarters required for comparisons')
print(f'Validated quarters: {len(fin)} | {fin.index.min().date()} to {fin.index.max().date()}')
print(f'Companies: {len(COMPANIES)} | All original quarterly observations retained')

# 2 — FINANCIAL CALCULATIONS AND DEFINITIONS
# For quarterly and historical-average views, original Excel ratios are preserved.
# For LTM and annualized views, new aggregate ratios are computed from original
# flow and stock data, without changing the source or stored Excel ratios.

def div(a,b):
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a/b)

def source(c, m):
    return fin[f'{c}_{m}']

def aggregate_window(c, start, end, annualize=False):
    """Exact DuPont identity for a contiguous window of quarters.

    Annualize flows only if requested. Stocks use the average of the
    quarter immediately preceding the window and the window's last quarter.
    """
    if start < 1 or end >= len(fin) or end < start:
        raise ValueError('Window requires preceding balance-sheet quarter')
    win = fin.iloc[start:end+1]
    count = len(win)
    rev = source(c,'revenue').iloc[start:end+1].sum(min_count=count)
    ni = source(c,'ni').iloc[start:end+1].sum(min_count=count)
    a0, a1 = source(c,'total_assets').iloc[start-1],source(c,'total_assets').iloc[end]
    e0, e1 = source(c,'equity').iloc[start-1],source(c,'equity').iloc[end]
    avg_a, avg_e = (a0+a1)/2, (e0+e1)/2
    scale = 4/count if annualize else 1
    rev, ni = rev*scale, ni*scale
    result = {'margin':div(ni,rev),'turnover':div(rev,avg_a),
              'leverage':div(avg_a,avg_e),'roe':div(ni,avg_e)}
    if not np.all(np.isfinite(list(result.values()))):
        raise ValueError(f'Non-finite DuPont aggregate for {c}, rows {start}:{end}')
    return result

def latest(c):
    return {'margin':float(source(c,'ni_margin').iloc[-1]),
            'turnover':float(source(c,'asset_turnover').iloc[-1]),
            'leverage':float(source(c,'leverage').iloc[-1]),
            'roe':float(source(c,'roe').iloc[-1])}

def quarterly(c, i):
    return {'margin':float(source(c,'ni_margin').iloc[i]),
            'turnover':float(source(c,'asset_turnover').iloc[i]),
            'leverage':float(source(c,'leverage').iloc[i]),
            'roe':float(source(c,'roe').iloc[i])}

def historical_average(c):
    # Descriptive means: NOT a multiplicative DuPont identity.
    return {'margin':float(source(c,'ni_margin').mean()),
            'turnover':float(source(c,'asset_turnover').mean()),
            'leverage':float(source(c,'leverage').mean()),
            'roe':float(source(c,'roe').mean())}

def annualized(c):
    # Retain the original full-period definition from the user's prior code:
    # 4 x average quarterly flows / mean historical balance-sheet stocks.
    rev = source(c,'revenue').mean()*4
    ni = source(c,'ni').mean()*4
    assets = source(c,'total_assets').mean()
    equity = source(c,'equity').mean()
    return {'margin':div(ni,rev),'turnover':div(rev,assets),
            'leverage':div(assets,equity),'roe':div(ni,equity)}

PERIODS = {
    'Latest quarter':lambda c:latest(c),
    'Latest four quarters (LTM)':lambda c:aggregate_window(c,len(fin)-4,len(fin)-1),
    'Historical quarterly average':lambda c:historical_average(c),
    'Full-period annualized':lambda c:annualized(c),
}

# Identity checks: use a strict test only for internally aggregated views.
for c in COMPANIES:
    for name in ('Latest four quarters (LTM)','Full-period annualized'):
        d = PERIODS[name](c)
        assert np.isclose(d['margin']*d['turnover']*d['leverage'],d['roe'],rtol=1e-9,atol=1e-9), (c,name)

# 3 — SHARED CHART DESIGN

def style(fig, title, height=820, top=160):
    fig.update_layout(
        title=dict(text=title,x=.5,xanchor='center',y=.985,font=dict(size=23,color=FG)),
        paper_bgcolor=BG,plot_bgcolor=PANEL,font=dict(family='Arial',size=12,color=FG),
        height=height,margin=dict(l=80,r=65,t=top,b=85),
        showlegend=False,hovermode='closest',bargap=.28)
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor=GRID,automargin=True)
    fig.update_yaxes(gridcolor=GRID,zeroline=True,zerolinecolor='rgba(235,245,255,.42)',automargin=True)
    return fig

def add_menu(fig, buttons, caption, x=.02, y=1.14, active=0):
    menus=list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    menus.append(dict(type='dropdown',buttons=buttons,active=active,x=x,y=y,
                      xanchor='left',yanchor='top',bgcolor='#E8EEF5',
                      bordercolor='#8FA4B8',font=dict(color='#152638',size=12)))
    fig.update_layout(updatemenus=menus)
    fig.add_annotation(x=x,y=y+.08,xref='paper',yref='paper',text=caption,
                       showarrow=False,xanchor='left',font=dict(size=11,color='#AFC6DA'))

def title_axes(fig, labels):
    # Explicitly place subplot labels ABOVE axes, not over the tallest bars.
    for i, label in enumerate(labels,1):
        xa = fig.layout[f'xaxis{i}' if i>1 else 'xaxis'].domain
        ya = fig.layout[f'yaxis{i}' if i>1 else 'yaxis'].domain
        fig.add_annotation(x=(xa[0]+xa[1])/2,y=ya[1]+.035,xref='paper',yref='paper',
                           text=f'<b>{label}</b>',showarrow=False,
                           font=dict(size=12,color=FG))

def bars(fig, x, values, colors, row, col, unit, visible=True):
    values=np.asarray(values,dtype=float)
    texts=[f'{v:.2f}{unit}' if np.isfinite(v) else 'N/A' for v in values]
    fig.add_trace(go.Bar(x=x,y=values,marker_color=colors,text=texts,
                         textposition='outside',cliponaxis=False,
                         hovertemplate='%{x}<br>%{y:.3f}'+unit+'<extra></extra>',
                         visible=visible),row=row,col=col)

# 4 — PANEL 02A: FOUR-COMPANY DUPONT COMPARISON
# QUESTION: Why are ROEs different across companies?

metrics=[('margin','NET INCOME MARGIN — PROFITABILITY','%'),
         ('turnover','ASSET TURNOVER — EFFICIENCY','×'),
         ('leverage','FINANCIAL LEVERAGE — MULTIPLIER','×'),
         ('roe','RETURN ON EQUITY — RESULT','%')]
period_labels=list(PERIODS)
fig_a=make_subplots(rows=2,cols=2,horizontal_spacing=.13,vertical_spacing=.25)
for pi,p in enumerate(period_labels):
    data=[PERIODS[p](c) for c in COMPANIES]
    for mi,(m,label,unit) in enumerate(metrics):
        vals=[d[m]*(100 if m in ('margin','roe') else 1) for d in data]
        bars(fig_a,list(COMPANIES.values()),vals,list(COLORS.values()),
             mi//2+1,mi%2+1,unit,visible=(pi==1))
style(fig_a,'02A — DuPont Peer Comparison | Latest four quarters (LTM)',height=950,top=195)
title_axes(fig_a,[x[1] for x in metrics])
for i,(_,_,unit) in enumerate(metrics):
    fig_a.update_yaxes(title_text=['Margin (%)','Turnover (×)','Leverage (×)','ROE (%)'][i],row=i//2+1,col=i%2+1)
buttons=[]
for pi,p in enumerate(period_labels):
    visible=[pi==j//4 for j in range(4*len(period_labels))]
    buttons.append(dict(label=p,method='update',args=[{'visible':visible},
                   {'title.text':f'02A — DuPont Peer Comparison | {p}'}]))
add_menu(fig_a,buttons,'SELECT COMPARISON PERIOD',active=1)

# Chart rendered by Streamlit.

# 5 — PANEL 02B: FOUR HISTORICAL FACTOR SERIES, ONE COMPANY
# QUESTION: How did profitability, efficiency and leverage evolve?
# Raw quarterly observations; no smoothing or fabricated regime breaks.

fig_b=make_subplots(rows=2,cols=2,horizontal_spacing=.13,vertical_spacing=.25)
for ci,c in enumerate(COMPANIES):
    for mi,(m,label,unit) in enumerate(metrics):
        source_metric={'margin':'ni_margin','turnover':'asset_turnover',
                       'leverage':'leverage','roe':'roe'}[m]
        y=source(c,source_metric)*(100 if m in ('margin','roe') else 1)
        fig_b.add_trace(go.Scatter(x=fin.index,y=y,mode='lines+markers',
                         line=dict(color=FACTOR_COLORS.get(m,COLORS[c]),width=2.5),
                         marker=dict(size=5),visible=(ci==0),
                         hovertemplate='%{x|%Y Q%q}<br>%{y:.3f}'+unit+'<extra></extra>'),
                         row=mi//2+1,col=mi%2+1)
style(fig_b,'02B — ROE Evolution | Dow',height=950,top=195)
title_axes(fig_b,[x[1] for x in metrics])
for mi,(m,label,unit) in enumerate(metrics):
    fig_b.update_yaxes(title_text=label.split(' — ')[0]+' ('+unit+')',row=mi//2+1,col=mi%2+1)
buttons=[]
for ci,(c,name) in enumerate(COMPANIES.items()):
    buttons.append(dict(label=name,method='update',args=[
        {'visible':[ci==j//4 for j in range(4*len(COMPANIES))]},
        {'title.text':f'02B — ROE Evolution | {name}'}]))
add_menu(fig_b,buttons,'SELECT COMPANY')

# Chart rendered by Streamlit.

# 6 — EXACT THREE-FACTOR SHAPLEY BRIDGE
# For f(m,t,l)=m*t*l, average the marginal impact of each factor
# over all 3! = 6 possible orders of replacement. This assigns all
# interactions and reconciles exactly to the ROE change.

def product(d):
    return float(np.prod([d[k] for k in FACTORS]))

def shapley(a,b):
    if not all(np.isfinite([a[k], b[k]]).all() for k in FACTORS):
        raise ValueError('Bridge requires finite factor values')
    out={k:0.0 for k in FACTORS}
    for order in permutations(FACTORS):
        cur=dict(a)
        for k in order:
            before=product(cur)
            cur[k]=b[k]
            out[k]+=(product(cur)-before)/6
    if not np.isclose(sum(out.values()),product(b)-product(a),atol=1e-10):
        raise AssertionError('Shapley bridge failed reconciliation')
    return out

def bridge_chart(a,b,title,subtitle):
    parts=shapley(a,b)
    x=['Starting ROE','Net margin','Asset turnover','Leverage','Ending ROE']
    y=[product(a)*100,parts['margin']*100,parts['turnover']*100,
       parts['leverage']*100,product(b)*100]
    fig=go.Figure(go.Waterfall(x=x,y=y,
        measure=['absolute','relative','relative','relative','total'],
        text=[f'{v:+.2f} pp' if i in (1,2,3) else f'{v:.2f}%' for i,v in enumerate(y)],
        textposition='outside',cliponaxis=False,
        increasing=dict(marker=dict(color='#37B58A')),
        decreasing=dict(marker=dict(color='#F16C71')),
        totals=dict(marker=dict(color='#4A9BDB')),
        connector=dict(line=dict(color='rgba(220,235,250,.35)')),
        hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'))
    style(fig,title,height=630,top=165)
    fig.update_layout(margin=dict(l=95,r=65,t=165,b=95))
    fig.update_yaxes(title_text='ROE / ROE change (percentage points)')
    fig.add_annotation(x=.5,y=1.12,xref='paper',yref='paper',
        text=subtitle,showarrow=False,font=dict(size=12,color='#B7CADB'))
    return fig

# 7 — PANEL 02C: WHAT CHANGED ROE?
# A bridge MUST start/end with internally consistent factor products.
# Compare (a) preceding vs latest quarter, or (b) first four vs last four
# quarters. The latter uses nonoverlapping four-quarter windows.

def bridge_period(c, mode):
    if mode=='qoq':
        # Original rounded Excel factors may not exactly equal Excel ROE.
        # For exact attribution, endpoints are their FACTOR PRODUCTS.
        return quarterly(c,-2),quarterly(c,-1), 'Previous vs latest quarter'
    if mode=='four':
        return (aggregate_window(c,1,4),
                aggregate_window(c,len(fin)-4,len(fin)-1),
                'First four eligible vs latest four quarters')
    raise ValueError(mode)

# Build one figure per company/mode; display with dropdown on combined figure.
# Each waterfall has a single trace; dropdown selects which is visible.
fig_c=go.Figure()
c_specs=[]
for c,name in COMPANIES.items():
    for mode in ('four','qoq'):
        a,b,desc=bridge_period(c,mode)
        p=shapley(a,b)
        vals=[product(a)*100,p['margin']*100,p['turnover']*100,
              p['leverage']*100,product(b)*100]
        fig_c.add_trace(go.Waterfall(
            x=['Starting ROE','Net margin','Asset turnover','Leverage','Ending ROE'],
            y=vals,measure=['absolute','relative','relative','relative','total'],
            text=[f'{v:+.2f} pp' if i in (1,2,3) else f'{v:.2f}%' for i,v in enumerate(vals)],
            textposition='outside',cliponaxis=False,
            increasing=dict(marker=dict(color='#37B58A')),
            decreasing=dict(marker=dict(color='#F16C71')),
            totals=dict(marker=dict(color='#4A9BDB')),
            connector=dict(line=dict(color='rgba(220,235,250,.35)')),
            visible=(c=='dow' and mode=='four')))
        c_specs.append((name,desc))
style(fig_c,'02C — ROE Change Bridge | Dow | First four eligible vs latest four quarters',height=650,top=180)
fig_c.update_yaxes(title_text='ROE / change (percentage points)')
fig_c.update_layout(margin=dict(l=95,r=70,t=180,b=95))
c_buttons=[]
for i,(name,desc) in enumerate(c_specs):
    c_buttons.append(dict(label=f'{name} | {desc}',method='update',args=[
        {'visible':[j==i for j in range(len(c_specs))]},
        {'title.text':f'02C — ROE Change Bridge | {name} | {desc}'}]))
add_menu(fig_c,c_buttons,'SELECT COMPANY AND COMPARISON',y=1.20)

# Chart rendered by Streamlit.

# 8 — PANEL 02D: WHICH LEVER MATTERS MATHEMATICALLY?
# Exact one-at-a-time finite differences (NOT forecasts or achievable targets).
# +1 percentage point margin; +0.10x turnover; +0.10x leverage.
# The three changes differ in units and are NOT comparable in feasibility.

def sensitivity(d):
    baseline=product(d)
    scenarios=[('Margin +1 pp',dict(d,margin=d['margin']+.01)),
               ('Turnover +0.10×',dict(d,turnover=d['turnover']+.10)),
               ('Leverage +0.10×',dict(d,leverage=d['leverage']+.10))]
    return [(label,(product(changed)-baseline)*100) for label,changed in scenarios]

fig_d=go.Figure()
d_specs=[]
for c,name in COMPANIES.items():
    for p in ('Latest quarter','Latest four quarters (LTM)','Full-period annualized'):
        d=PERIODS[p](c)
        ss=sensitivity(d)
        fig_d.add_trace(go.Bar(y=[s[0] for s in ss],x=[s[1] for s in ss],
            orientation='h',marker_color=['#F16C71','#4A9BDB','#37B58A'],
            text=[f'{s[1]:+.2f} pp' for s in ss],textposition='outside',
            cliponaxis=False,visible=(c=='dow' and p=='Latest four quarters (LTM)'),
            hovertemplate='%{y}<br>ROE impact: %{x:+.3f} pp<extra></extra>'))
        d_specs.append((name,p))
style(fig_d,'02D — ROE Sensitivity | Dow | Latest four quarters (LTM)',height=630,top=175)
fig_d.update_xaxes(title_text='Change in ROE (percentage points)',zeroline=True)
fig_d.update_layout(margin=dict(l=180,r=100,t=175,b=105))
fig_d.add_annotation(x=.5,y=-.22,xref='paper',yref='paper',showarrow=False,
    text='Illustrative one-factor changes; mathematical sensitivities, not operational forecasts.',
    font=dict(size=11,color='#B7CADB'))
d_buttons=[]
for i,(name,p) in enumerate(d_specs):
    d_buttons.append(dict(label=f'{name} | {p}',method='update',args=[
        {'visible':[j==i for j in range(len(d_specs))]},
        {'title.text':f'02D — ROE Sensitivity | {name} | {p}'}]))
add_menu(fig_d,d_buttons,'SELECT COMPANY AND BASELINE',y=1.20,active=1)

# Chart rendered by Streamlit.

# 9 — PANEL 02E: WHY DOES A PEER HAVE DIFFERENT ROE FROM DOW?
# Exact factor-product bridge, using Shapley interaction allocation.
# Direction: Dow -> selected peer. Difference = peer ROE - Dow ROE.
# Historical quarterly average intentionally omitted: mean ratios do not
# satisfy a multiplicative identity.

fig_e=go.Figure()
e_specs=[]
for peer in ('lyb','dd','basf'):
    for p in ('Latest quarter','Latest four quarters (LTM)','Full-period annualized'):
        a,b=PERIODS[p]('dow'),PERIODS[p](peer)
        parts=shapley(a,b)
        vals=[product(a)*100,parts['margin']*100,parts['turnover']*100,
              parts['leverage']*100,product(b)*100]
        fig_e.add_trace(go.Waterfall(
            x=['Dow ROE','Margin gap','Turnover gap','Leverage gap',COMPANIES[peer]+' ROE'],
            y=vals,measure=['absolute','relative','relative','relative','total'],
            text=[f'{v:+.2f} pp' if i in (1,2,3) else f'{v:.2f}%' for i,v in enumerate(vals)],
            textposition='outside',cliponaxis=False,
            increasing=dict(marker=dict(color='#37B58A')),
            decreasing=dict(marker=dict(color='#F16C71')),
            totals=dict(marker=dict(color='#4A9BDB')),
            connector=dict(line=dict(color='rgba(220,235,250,.35)')),
            visible=(peer=='lyb' and p=='Latest four quarters (LTM)')))
        e_specs.append((peer,p))
style(fig_e,'02E — ROE Peer Gap | Dow → LyondellBasell | Latest four quarters (LTM)',height=650,top=180)
fig_e.update_yaxes(title_text='ROE / peer gap (percentage points)')
fig_e.update_layout(margin=dict(l=95,r=70,t=180,b=100))
e_buttons=[]
for i,(peer,p) in enumerate(e_specs):
    e_buttons.append(dict(label=f'Dow vs {COMPANIES[peer]} | {p}',method='update',args=[
        {'visible':[j==i for j in range(len(e_specs))]},
        {'title.text':f'02E — ROE Peer Gap | Dow → {COMPANIES[peer]} | {p}'}]))
add_menu(fig_e,e_buttons,'SELECT PEER AND COMPARISON PERIOD',y=1.20,active=1)

# Chart rendered by Streamlit.

# 10 — AUDIT OUTPUTS: NO SOURCE FILE IS OVERWRITTEN
rows=[]
for p in PERIODS:
    for c,name in COMPANIES.items():
        d=PERIODS[p](c)
        rows.append(dict(company=name,period=p,**d,
                         product=product(d),identity_gap=d['roe']-product(d)))
audit=pd.DataFrame(rows)
print('\nPANEL 02 — DUPONT DATA AUDIT')

# Notebook-only display suppressed.

print('\nBRIDGE RECONCILIATION CHECKS (percentage points):')
for c,name in COMPANIES.items():
    for mode in ('four','qoq'):
        a,b,desc=bridge_period(c,mode)
        err=(sum(shapley(a,b).values())-(product(b)-product(a)))*100
        print(f'{name:18s} | {desc:44s} | error {err:+.10f} pp')
print('\nNOTES:')
print('• Latest-quarter and historical-average ratios are the original Excel-derived values.')
print('• LTM sums flows; uses beginning/ending balance-sheet averages.')
print('• Historical average is descriptive: the average ROE need not equal the product of average factors.')
print('• Bridge endpoints use factor products; source ROE may differ due to rounding.')
print('• Bridges allocate interaction effects exactly; they do not establish causality.')
print('• Sensitivity scenarios are mathematical, not forecasts or management targets.')
print('• No original CSV or workbook values were overwritten.')

