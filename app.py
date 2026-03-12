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
    all_symbols = list(set(tickers + [benchmark]))
    # Descargamos
    df = yf.download(all_symbols, period="5y")
    
    # Aplanamos el DataFrame:
    # Si yfinance devuelve MultiIndex, tomamos Adj Close y lo convertimos a tabla simple
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    
    return df.dropna(how='all')

st.title("📈 Dashboard Financiero Pro")
ticker_input = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in ticker_input.split(",")]
benchmark = "^GSPC"

data = get_data(tickers, benchmark)

# Validación simple
if data.empty:
    st.error("Error: No se obtuvieron datos. Revisa los tickers o tu conexión.")
    st.stop()

# 1. Evolución de Precios
st.header("1. Evolución de Precios")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Todos los Activos (Base 100)")
    fig_all = px.line((data[tickers] / data[tickers].iloc[0]) * 100, template="plotly_dark")
    st.plotly_chart(fig_all, use_container_width=True)
with c2:
    sel = st.selectbox("Seleccionar Activo:", tickers)
    st.subheader(f"Precio: {sel}")
    fig_ind = px.line(data[sel], template="plotly_dark")
    st.plotly_chart(fig_ind, use_container_width=True)

# 2. Métricas y Matrices
returns = np.log(data[tickers] / data[tickers].shift(1)).dropna()
st.header("2. Métricas y Relaciones")
st.dataframe(pd.DataFrame({'Retorno Anual': returns.mean()*252, 'Volatilidad': returns.std()*np.sqrt(252), 'Máximo': data[tickers].max(), 'Mínimo': data[tickers].min()}), use_container_width=True)

c3, c4 = st.columns(2)
c3.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="Purples"), use_container_width=True)
c4.plotly_chart(px.imshow(returns.cov()*252, text_auto=".4f", color_continuous_scale="Purples"), use_container_width=True)

# 3. Portafolio Optimizado
st.header("3. Portafolio Optimizado (Max Sharpe)")
def get_perf(w): return np.sum(returns.mean()*w)*252, np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
opt = sco.minimize(lambda w: -(get_perf(w)[0]-0.04)/get_perf(w)[1], len(tickers)*[1./len(tickers)], bounds=[(0, 0.4) for _ in tickers], constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))

c_pie1, c_pie2 = st.columns([2, 1])
with c_pie1:
    fig_pie = px.pie(names=tickers, values=opt.x, hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)
with c_pie2:
    st.write("### Composición")
    st.dataframe(pd.DataFrame({'Activo': tickers, 'Peso': opt.x}), use_container_width=True, hide_index=True)

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
fig_back = px.line(df_back, template="plotly_dark")
c5.plotly_chart(fig_back, use_container_width=True)
fig_roll = px.line(((port_rets.rolling(126).mean()*252) - 0.04) / (port_rets.rolling(126).std()*np.sqrt(252)), template="plotly_dark")
c6.plotly_chart(fig_roll, use_container_width=True)

# 5. Análisis de Riesgo
st.header("5. Análisis de Riesgo")
conf = st.select_slider("Nivel de Confianza:", options=[90, 95, 99], value=95)
alpha = (100 - conf) / 100
port_rets = (returns * opt.x).sum(axis=1)
mu_anual = port_rets.mean() * 252
sigma_anual = port_rets.std() * np.sqrt(252)
Z = np.random.normal(0, 1, (252, 10000))
price_paths = np.cumprod(np.exp((mu_anual/252 - 0.5 * (sigma_anual/np.sqrt(252))**2) + (sigma_anual/np.sqrt(252)) * Z), axis=0)
var_hist = abs(np.percentile(port_rets, alpha * 100) * np.sqrt(252))
var_param = abs(mu_anual - norm.ppf(alpha, 0, 1) * sigma_anual)
var_mc = abs(np.percentile(price_paths[-1] - 1, alpha * 100))

c7, c8, c9 = st.columns(3)
c7.metric(f"VaR Histórico {conf}%", f"{var_hist:.2%}")
c8.metric(f"VaR Paramétrico {conf}%", f"{var_param:.2%}")
c9.metric(f"VaR Monte Carlo {conf}%", f"{var_mc:.2%}")

c10, c11 = st.columns(2)
c10.plotly_chart(px.histogram(price_paths[-1] - 1, template="plotly_dark"), use_container_width=True)
c11.plotly_chart(px.line(price_paths[:, ::100], template="plotly_dark"), use_container_width=True)
