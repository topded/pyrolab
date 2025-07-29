import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from time import sleep
from math import exp
from io import BytesIO
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# ====== Data Dasar ======
PLASTIC_YIELDS = {
    "PP": {"oil": 0.70, "gas": 0.20, "residue": 0.10},
    "PET": {"oil": 0.45, "gas": 0.35, "residue": 0.20},
    "HDPE": {"oil": 0.60, "gas": 0.25, "residue": 0.15},
    "PVC": {"oil": 0.30, "gas": 0.30, "residue": 0.40, "hcl": 0.25}
}
GAS_YIELDS = {
    "PP": {"CH4": 0.10, "H2": 0.10},
    "HDPE": {"CH4": 0.12, "H2": 0.13},
    "PET": {"CO2": 0.20, "CO": 0.15},
    "PVC": {"HCl": 0.25, "Cl2": 0.05}
}
CATALYST_EFFECT = {"none": 1.0, "zeolite": 1.1, "alumina": 1.05, "silica": 1.03}
PRESSURE_EFFECT = {"vacuum": 1.15, "atmospheric": 1.0, "high": 0.9}
PARTICLE_EFFECT = {"small": 1.1, "medium": 1.0, "large": 0.9}
REACTOR_EFFECT = {"batch": 1.0, "fixed-bed": 1.05, "fluidized-bed": 1.1}
ENV_EFFECT = {"inert": 1.0, "oxidative": 0.7}
HEAT_SOURCE_EFFECT = {"electric": 1.1, "direct_fire": 0.95, "solar": 0.85}
ENERGY_CONTENT = {"oil": 43, "gas": 30}  # MJ/kg

# ====== Fungsi ======
def arrhenius_rate(A, Ea, T):
    R = 8.314
    return A * exp(-Ea / (R * T))

def estimate_reaction_time(A, Ea, T):
    k = arrhenius_rate(A, Ea, T)
    return 1 / k if k > 0 else float("inf")

def simulate(plastic_mix, total_mass, temp_profile, duration,
             catalyst, pressure, particle_size,
             reactor_type, environment, heat_source):
    comp_mass = {ptype: (ratio / sum(plastic_mix.values())) * total_mass for ptype, ratio in plastic_mix.items()}
    avg_temp = np.mean(temp_profile)
    temp_factor = np.clip((avg_temp - 350) / 150, 0.5, 1.5)

    factor = temp_factor * CATALYST_EFFECT[catalyst] * PRESSURE_EFFECT[pressure] * \
             PARTICLE_EFFECT[particle_size] * REACTOR_EFFECT[reactor_type] * \
             ENV_EFFECT[environment] * HEAT_SOURCE_EFFECT[heat_source]

    results = {"oil": 0, "gas": 0, "residue": 0, "hcl": 0}
    gases = {}

    for ptype, mass in comp_mass.items():
        yield_base = PLASTIC_YIELDS.get(ptype, {})
        gas_base = GAS_YIELDS.get(ptype, {})
        for k in results:
            if k in yield_base:
                results[k] += yield_base[k] * mass * factor
        for g, v in gas_base.items():
            gases[g] = gases.get(g, 0) + v * mass * factor

    return results, gases, factor, avg_temp
    

def generate_pdf_report(results, gases, energy, reaction_time, factor):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Laporan Simulasi Pirolisis Plastik", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Energi total: {energy:.2f} MJ", ln=True)
    pdf.cell(0, 10, f"Waktu reaksi: {reaction_time:.2f} s", ln=True)
    pdf.cell(0, 10, f"Faktor efisiensi total: {factor:.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Produk Utama:", ln=True)
    pdf.set_font("Arial", "", 12)
    for k, v in results.items():
        pdf.cell(0, 10, f"{k}: {v:.2f} gram", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Produk Gas:", ln=True)
    pdf.set_font("Arial", "", 12)
    for k, v in gases.items():
        pdf.cell(0, 10, f"{k}: {v:.2f} gram", ln=True)

    return pdf.output(dest="S").encode("latin1")

def build_realtime_chart(temp_profile, delay=0.1):
    temp_data = []
    chart = st.line_chart()
    for i, temp in enumerate(temp_profile):
        temp_data.append({"waktu": i, "suhu": temp})
        df_chart = pd.DataFrame(temp_data).set_index("waktu")
        chart.line_chart(df_chart)
        sleep(delay)

# ====== Streamlit App UI ======
st.set_page_config(page_title="PiroFuel Lab - Pirolisis Plastik v1.0", layout="wide")
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)


# ====== HEADER UTAMA ======
st.markdown("""
<h1 style='text-align: center;'>🧪 PiroFuel – Pirolisis Virtual Lab</h1>
<h4 style='text-align: center; margin-top: -15px;'>Version 1.0 · Developed by <span style='color: #4CAF50;'>TechCava</span></h4>
<p style='text-align: center;'>
       <a href='https://github.com/' target='_blank' style='text-decoration: none; margin: 0 10px;'>
        <i class='fab fa-github'></i> GitHub
    </a>
    |
    <a href='https://instagram.com/' target='_blank' style='text-decoration: none; margin: 0 10px;'>
        <i class='fab fa-instagram'></i> Instagram
    </a>
    |
    <a href='https://facebook.com/' target='_blank' style='text-decoration: none; margin: 0 10px;'>
        <i class='fab fa-facebook'></i> Facebook
    </a>
</p>
""", unsafe_allow_html=True)

st.markdown("""---""")  # Garis pemisah


with st.sidebar:
    st.header("📋 Input Parameter")
    
    plastic_mix = {pt: st.slider(f"{pt} (%)", 0, 100, 25) for pt in PLASTIC_YIELDS}
    total_mass = st.number_input("Massa Total (g)", 100, 10000, 1500)
    temp_profile = list(map(int, st.text_input("Profil Suhu (°C)", "300,350,400,450,500").split(",")))
    duration = st.slider("Durasi (menit)", 10, 180, 90)

    catalyst = st.selectbox("Katalis", list(CATALYST_EFFECT.keys()))
    pressure = st.selectbox("Tekanan", list(PRESSURE_EFFECT.keys()))
    particle_size = st.selectbox("Ukuran Partikel", list(PARTICLE_EFFECT.keys()))
    reactor_type = st.selectbox("Reaktor", list(REACTOR_EFFECT.keys()))
    environment = st.selectbox("Lingkungan", list(ENV_EFFECT.keys()))
    heat_source = st.selectbox("Sumber Panas", list(HEAT_SOURCE_EFFECT.keys()))

# Simulasi realtime
results, gases, factor, avg_temp = simulate(
    plastic_mix, total_mass, temp_profile, duration,
    catalyst, pressure, particle_size, reactor_type, environment, heat_source
)
energy = sum((results[k] / 1000) * v for k, v in ENERGY_CONTENT.items() if k in results)
reaction_time = estimate_reaction_time(1e7, 100000, avg_temp + 273.15)

#HASILL

col_head_left, col_head_right = st.columns([4, 1])

with col_head_left:
    st.header("📊 Hasil Simulasi Realtime")
with col_head_right:
    enable_batch = st.checkbox("🔁 Multi-Batch Mode", help="Aktifkan untuk membandingkan beberapa batch")

# === DATAFRAME PRODUK UTAMA ===
df = pd.DataFrame.from_dict(results, orient="index", columns=["Massa (g)"])
df["Fraksi (%)"] = df["Massa (g)"] / df["Massa (g)"].sum() * 100
st.dataframe(df)


col1, col2 = st.columns(2)

with col1:
    st.success(f"🔥 Energi total: {energy:.2f} MJ")

with col2:
    st.info(f"⏱️ Waktu reaksi estimasi: {reaction_time:.2f} detik")
st.caption(f"Faktor efisiensi: {factor:.2f} | Suhu rata-rata: {avg_temp:.1f}°C")


col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Grafik Fraksi Produk")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    df["Massa (g)"].plot.pie(autopct="%.1f%%", ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1)

with col_b:
    st.subheader("🧪 Komposisi Gas")
    gas_df = pd.DataFrame.from_dict(gases, orient="index", columns=["Gram"])
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    gas_df.plot.bar(ax=ax2, legend=False)
    ax2.set_ylabel("Gram")
    st.pyplot(fig2)
    st.dataframe(gas_df)


with st.expander("🌡️ Grafik Suhu Waktu (Realtime)"):
    build_realtime_chart(temp_profile)

st.download_button("📥 Unduh Laporan PDF",
                   data=generate_pdf_report(results, gases, energy, reaction_time, factor),
                   file_name="laporan_pirolisis.pdf")

# === MULTI BATCH MODE ===
if enable_batch:
    st.subheader("📈 Mode Perbandingan Multi-Batch")
    selected_catalysts = st.multiselect("Katalis", list(CATALYST_EFFECT.keys()), default=["none", "zeolite"])
    selected_temps = st.text_input("Profil Suhu Batch (dipisah `;`)", "300,350,400;400,450,500")

    try:
        batch_profiles = [list(map(int, s.strip().split(","))) for s in selected_temps.split(";")]
    except:
        st.error("Format suhu salah")
        batch_profiles = []

    batch_result = []
    for cat in selected_catalysts:
        for profile in batch_profiles:
            r, _, f, t = simulate(plastic_mix, total_mass, profile, duration,
                                  cat, pressure, particle_size, reactor_type, environment, heat_source)
            batch_result.append({"Katalis": cat, "Suhu Rata2": t, "Oil": r["oil"], "Gas": r["gas"], "Faktor": f})

    if batch_result:
        df_batch = pd.DataFrame(batch_result)
        st.dataframe(df_batch)

        fig, ax = plt.subplots()
        for cat in selected_catalysts:
            sub = df_batch[df_batch["Katalis"] == cat]
            ax.plot(sub["Suhu Rata2"], sub["Oil"], marker='o', label=f"Oil - {cat}")
            ax.plot(sub["Suhu Rata2"], sub["Gas"], marker='x', linestyle='--', label=f"Gas - {cat}")
        ax.legend()
        ax.set_title("Output Oil dan Gas")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        df_batch.groupby("Katalis")[["Oil", "Gas"]].sum().plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_title("Total Output per Katalis")
        st.pyplot(fig2)

        # AI Regression
        st.subheader("🤖 Prediksi AI (Linear Regression)")
        features = df_batch[["Suhu Rata2", "Faktor"]]
        oil_model = LinearRegression().fit(features, df_batch["Oil"])
        gas_model = LinearRegression().fit(features, df_batch["Gas"])

        pred_temp = st.slider("Prediksi Suhu (°C)", 300, 600, 450)
        pred_factor = st.slider("Prediksi Faktor", 0.5, 2.0, float(df_batch["Faktor"].mean()), step=0.01)
        pred_input = np.array([[pred_temp, pred_factor]])

        st.success(f"📈 Estimasi Oil: {oil_model.predict(pred_input)[0]:.2f} gram")
        st.success(f"📈 Estimasi Gas: {gas_model.predict(pred_input)[0]:.2f} gram")

        # Grafik 3D
        st.subheader("🌐 Grafik 3D Prediksi")
        T_range = np.linspace(300, 600, 30)
        F_range = np.linspace(0.5, 2.0, 30)
        T, F = np.meshgrid(T_range, F_range)
        X = np.c_[T.ravel(), F.ravel()]
        oil_surface = oil_model.predict(X).reshape(T.shape)
        gas_surface = gas_model.predict(X).reshape(T.shape)

        fig3 = plt.figure(figsize=(8, 5))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot_surface(T, F, oil_surface, cmap='viridis')
        ax3.set_title("Oil Output vs Suhu & Faktor")
        ax3.set_xlabel("Suhu")
        ax3.set_ylabel("Faktor")
        ax3.set_zlabel("Oil")
        st.pyplot(fig3)

        fig4 = plt.figure(figsize=(8, 5))
        ax4 = fig4.add_subplot(111, projection='3d')
        ax4.plot_surface(T, F, gas_surface, cmap='plasma')
        ax4.set_title("Gas Output vs Suhu & Faktor")
        ax4.set_xlabel("Suhu")
        ax4.set_ylabel("Faktor")
        ax4.set_zlabel("Gas")
        st.pyplot(fig4)

# ====== FOOTER ======
st.markdown("""<hr style="margin-top: 40px;"/>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: grey;'>
    🔬 © 2025 <strong>PiroFuel</strong> &nbsp;|&nbsp; Built with ❤️ by 
    <a href="https://github.com/" target="_blank">Rifki Maulana</a> 🚀
</div>
""", unsafe_allow_html=True)
