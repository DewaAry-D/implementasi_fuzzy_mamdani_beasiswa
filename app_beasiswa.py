"""
Antarmuka visual Fuzzy Mamdani - Kelayakan Beasiswa (Streamlit).
Jalankan:  streamlit run app_beasiswa.py
Pastikan file fuzzy_mamdani_beasiswa.py berada di folder yang sama.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import fuzzy_mamdani_beasiswa as fm

st.set_page_config(page_title="Fuzzy Mamdani Beasiswa", page_icon="🎓", layout="wide")

COLORS = ["#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93"]
CAT_COLOR = {"Tidak Layak": "#d62828", "Cukup Layak": "#f4a261",
             "Layak": "#2a9d8f", "Sangat Layak": "#1d6fb8"}

st.title("🎓 Sistem Kelayakan Penerima Beasiswa")
st.caption("Fuzzy Mamdani · AND = MIN · Implikasi MIN · Agregasi MAX · Centroid diskrit")

# ---------------- INPUT ----------------
with st.sidebar:
    st.header("Data Mahasiswa")
    ipk = st.slider("IPK", 0.0, 4.0, 3.20, 0.01)
    penghasilan = st.slider("Penghasilan orang tua (juta Rp/bulan)", 0.0, 15.0, 3.0, 0.1)
    tanggungan = st.slider("Jumlah tanggungan (orang)", 0, 8, 5, 1)
    prestasi = st.slider("Skor prestasi (0–100)", 0, 100, 75, 1)
    st.button("Hitung kelayakan", type="primary", width="stretch")
    st.caption("Hasil diperbarui otomatis saat slider digeser.")

inputs = {"ipk": ipk, "penghasilan": penghasilan,
          "tanggungan": tanggungan, "prestasi": prestasi}

try:
    res = fm.evaluate_student(inputs, show_detail=False)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------- HASIL UTAMA ----------------
cat = res["category"]
c1, c2, c3 = st.columns(3)
c1.metric("Skor kelayakan (centroid)", f"{res['score']:.3f}")
c2.metric("Kategori", cat)
c3.metric("Aturan aktif", f"{len(res['active_rules'])} dari {len(fm.RULES)}")

st.markdown(
    f"<div style='padding:14px;border-radius:10px;background:{CAT_COLOR[cat]}22;"
    f"border-left:6px solid {CAT_COLOR[cat]};font-size:1.1rem'>"
    f"Rekomendasi: <b>{cat}</b> (skor {res['score']:.2f} / 100)</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Fuzzifikasi", "2. Aturan aktif", "3. Agregasi & Defuzzifikasi", "4. Fungsi keanggotaan output"]
)

# ---------------- TAB 1: FUZZIFIKASI ----------------
with tab1:
    key_map = {"IPK": ipk, "Penghasilan": penghasilan,
               "Tanggungan": tanggungan, "Prestasi": prestasi}
    cols = st.columns(2)
    for i, (var, x) in enumerate(key_map.items()):
        lo, hi = fm.UNIVERSE[var]
        grid = np.linspace(lo, hi, 600)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for j, lab in enumerate(fm.MF_DEF[var]):
            ax.plot(grid, fm.mf_vector(var, lab, grid), lw=2, color=COLORS[j], label=lab)
            deg = res["fuzzification"][var][lab]
            if deg > 0:
                ax.plot([x], [deg], "o", color=COLORS[j])
        ax.axvline(x, color="black", ls="--", lw=1)
        ax.set_title(f"{var} = {x}")
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Nilai crisp")
        ax.set_ylabel("μ")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        with cols[i % 2]:
            st.pyplot(fig)
            st.dataframe(
                pd.DataFrame({"Himpunan": list(res["fuzzification"][var]),
                              "μ": [round(v, 6) for v in res["fuzzification"][var].values()]}),
                hide_index=True, width="stretch")
        plt.close(fig)

# ---------------- TAB 2: ATURAN AKTIF ----------------
with tab2:
    rows = [{
        "No": r["number"],
        "IPK": r["antecedent"][0],
        "Penghasilan": r["antecedent"][1],
        "Tanggungan": r["antecedent"][2],
        "Prestasi": r["antecedent"][3],
        "Konsekuen": r["consequent"],
        "α (MIN)": round(r["alpha"], 6),
        "Jenis": r["type"],
    } for r in res["active_rules"]]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("α (MIN)", ascending=False),
                     hide_index=True, width="stretch")
    else:
        st.warning("Tidak ada aturan aktif.")
    st.caption("α = min(μ_IPK, μ_Penghasilan, μ_Tanggungan, μ_Prestasi). "
               "Nomor aturan adalah urutan pembangkitan kombinasi, bukan R1–R81 pada lampiran.")

# ---------------- TAB 3: AGREGASI ----------------
with tab3:
    z, agg = res["output_grid"], res["aggregated"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in res["active_rules"]:
        ax.plot(z, r["implied"], color="gray", alpha=0.35, lw=1)
    ax.fill_between(z, agg, color=CAT_COLOR[cat], alpha=0.35, label="Agregasi MAX")
    ax.plot(z, agg, color=CAT_COLOR[cat], lw=2)
    ax.axvline(res["score"], color="black", ls="--", lw=2,
               label=f"Centroid z* = {res['score']:.3f}")
    ax.set_xlabel("Skor kelayakan (z)")
    ax.set_ylabel("μ agregat")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Perhitungan centroid diskrit** (grid 0–100, Δz = 0.1, 1.001 titik)")
    st.latex(r"z^* = \frac{\sum z_i\,\mu(z_i)}{\sum \mu(z_i)} = "
             rf"\frac{{{res['centroid_numerator']:.6f}}}{{{res['centroid_denominator']:.6f}}} "
             rf"= {res['score']:.6f}")

# ---------------- TAB 4: OUTPUT MF ----------------
with tab4:
    z = np.linspace(0, 100, 600)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for lab in fm.MF_DEF["Kelayakan"]:
        ax.plot(z, fm.mf_vector("Kelayakan", lab, z), lw=2,
                color=CAT_COLOR[lab], label=lab)
    ax.axvline(res["score"], color="black", ls="--", lw=1.5)
    ax.set_xlabel("Skor kelayakan")
    ax.set_ylabel("μ")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, loc="upper center")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        pd.DataFrame({"Himpunan output": list(res["output_memberships"]),
                      "μ pada skor akhir": [round(v, 6) for v in res["output_memberships"].values()]}),
        hide_index=True, width="stretch")
    st.caption("Kategori = himpunan output dengan derajat keanggotaan terbesar pada skor crisp.")
