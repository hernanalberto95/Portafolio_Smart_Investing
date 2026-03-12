import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sco
from datetime import datetime, timedelta
from scipy.stats import norm

# Configuración Global
st.set_page_config(page_title="Dashboard Pro", layout="wide")
st.markdown("""<style>.stApp { background-color: #050505; color: #E0E0E0; } 
    h1, h2, h3 { color: #BA68C8; } .stMetric { background-color: #121212; padding: 15px; border-radius: 10px; border: 1px solid #4A148C; }</style>""", unsafe_allow_html=True)

@st.cache_data
def get_data(tickers, benchmark):
    end = datetime.today(); start = end - timedelta(days=5*365)
    df = yf.download(tickers + [benchmark], start=start, end=end)
    return (df['Adj Close'] if 'Adj Close' in df.columns else df['Close'] if 'Close' in df.columns else df.xs('Close', level=0, axis=1)).dropna()

st.title("📈 Dashboard Financiero Pro")
tickers = [t.strip().upper() for t in st.sidebar.text_input("Tickers (separados por coma):", "AAPL, MSFT, GOOGL, AMZN").split(",")]
benchmark = "^GSPC"
data = get_data(tickers, benchmark)

# 1. Evolución de Precios
st.header("1. Evolución de Precios")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Todos los Activos (Base 100)")
    fig_all = px.line((data[tickers] / data[tickers].iloc[0]) * 100, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Purples_r)
    fig_all.update_layout(showlegend=False)
    st.plotly_chart(fig_all, use_container_width=True)
with c2:
    sel = st.selectbox("Seleccionar Activo:", tickers)
    st.subheader(f"Precio: {sel}")
    fig_ind = px.line(data[sel], template="plotly_dark", color_discrete_sequence=['#BA68C8'])
    fig_ind.update_layout(showlegend=False)
    st.plotly_chart(fig_ind, use_container_width=True)

# 2. Métricas y Matrices
returns = np.log(data[tickers] / data[tickers].shift(1)).dropna()
st.header("2. Métricas y Relaciones")
st.dataframe(pd.DataFrame({'Retorno Anual': returns.mean()*252, 'Volatilidad': returns.std()*np.sqrt(252), 'Máximo': data[tickers].max(), 'Mínimo': data[tickers].min()}), use_container_width=True)

c3, c4 = st.columns(2)
c3.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="Purples").update_layout(showlegend=False), use_container_width=True)
c4.plotly_chart(px.imshow(returns.cov()*252, text_auto=".4f", color_continuous_scale="Purples").update_layout(showlegend=False), use_container_width=True)

# 3. Portafolio Optimizado
st.header("3. Portafolio Optimizado (Max Sharpe)")
def get_perf(w): return np.sum(returns.mean()*w)*252, np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
opt = sco.minimize(lambda w: -(get_perf(w)[0]-0.04)/get_perf(w)[1], len(tickers)*[1./len(tickers)], bounds=[(0, 0.4) for _ in tickers], constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))

c_pie1, c_pie2 = st.columns([2, 1])
with c_pie1:
    fig_pie = px.pie(names=tickers, values=opt.x, hole=0.4, color_discrete_sequence=['#4A148C', '#7B1FA2', '#9C27B0', '#CE93D8'])
    fig_pie.update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)
with c_pie2:
    st.write("### Composición")
    df_weights = pd.DataFrame({'Activo': tickers, 'Peso': opt.x})
    st.dataframe(df_weights.style.format({'Peso': '{:.2%}'}), use_container_width=True, hide_index=True)

ret_p, std_p = get_perf(opt.x)
c_m1, c_m2, c_m3 = st.columns(3)
c_m1.metric("Rendimiento Anual", f"{ret_p:.2%}")
c_m2.metric("Desviación Estándar", f"{std_p:.2%}")
c_m3.metric("Sharpe Ratio", f"{(ret_p-0.04)/std_p:.2f}")

# 4. Backtesting
st.header("4. Backtesting")
c5, c6 = st.columns(2)
port_rets = (returns * opt.x).sum(axis=1)
df_back = pd.DataFrame({'Portafolio': (1 + port_rets).cumprod(), 'Benchmark': (1 + np.log(data[benchmark] / data[benchmark].shift(1)).dropna()).cumprod()})
fig_back = px.line(df_back, color_discrete_sequence=['#BA68C8', '#4A148C'], template="plotly_dark")
fig_back.update_layout(title="Portafolio Optimizado vs Benchmark", showlegend=True)
c5.plotly_chart(fig_back, use_container_width=True)
fig_roll = px.line(((port_rets.rolling(126).mean()*252) - 0.04) / (port_rets.rolling(126).std()*np.sqrt(252)), color_discrete_sequence=['#CE93D8'], template="plotly_dark")
fig_roll.update_layout(title="Rolling Sharpe (6 meses)", showlegend=False)
c6.plotly_chart(fig_roll, use_container_width=True)

# 5. Análisis de Riesgo (Dinámico)
st.header("5. Análisis de Riesgo")
conf = st.select_slider("Nivel de Confianza:", options=[90, 95, 99], value=95)
alpha = (100 - conf) / 100

port_rets = (returns * opt.x).sum(axis=1)
mu_anual = port_rets.mean() * 252
sigma_anual = port_rets.std() * np.sqrt(252)

# Cálculos
var_hist = abs(np.percentile(port_rets, alpha * 100) * np.sqrt(252))
var_param = abs(mu_anual - norm.ppf(alpha, 0, 1) * sigma_anual)
# Monte Carlo
Z = np.random.normal(0, 1, (252, 10000))
price_paths = np.cumprod(np.exp((mu_anual/252 - 0.5 * (sigma_anual/np.sqrt(252))**2) + (sigma_anual/np.sqrt(252)) * Z), axis=0)
var_mc = abs(np.percentile(price_paths[-1] - 1, alpha * 100))

c7, c8, c9 = st.columns(3)
c7.metric(f"VaR Histórico {conf}%", f"{var_hist:.2%}")
c8.metric(f"VaR Paramétrico {conf}%", f"{var_param:.2%}")
c9.metric(f"VaR Monte Carlo {conf}%", f"{var_mc:.2%}")

c10, c11 = st.columns(2)
fig_hist = px.histogram(price_paths[-1] - 1, title=f"Distribución (Confianza {conf}%)", color_discrete_sequence=["#9C27B0"], template="plotly_dark")
fig_hist.update_layout(showlegend=False)
c10.plotly_chart(fig_hist, use_container_width=True)
fig_paths = px.line(price_paths[:, ::100], title="Trayectorias Monte Carlo", color_discrete_sequence=["#BA68C8"], template="plotly_dark")
fig_paths.update_layout(showlegend=False)
c11.plotly_chart(fig_paths, use_container_width=True)
