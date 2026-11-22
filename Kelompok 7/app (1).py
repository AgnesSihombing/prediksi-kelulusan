import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", page_icon="🎓", layout="wide")

@st.cache_resource
def muat():
    with open("metadata.json") as f:
        meta = json.load(f)
    return joblib.load("model.joblib"), meta

model, meta = muat()
R = meta["rentang_fitur"]

st.title("🎓 Sistem Peringatan Dini Kelulusan Tepat Waktu")
st.caption(f"Kelompok 7 · Pembelajaran Mesin · Model: {meta['model']}")

tab1, tab2 = st.tabs(["Prediksi mahasiswa", "Tentang model"])

with tab1:
    st.sidebar.header("Profil akademik mahasiswa")
    gpa = st.sidebar.slider("IPK", 0.0, 4.0, round(R["gpa"]["median"], 2), 0.01)
    sks = st.sidebar.slider("SKS terselesaikan", 0, 160, int(R["credits_completed"]["median"]))
    hadir = st.sidebar.slider("Kehadiran (%)", 0.0, 100.0, round(R["attendance_pct"]["median"], 1), 0.5)
    gagal = st.sidebar.slider("Mata kuliah gagal", 0, 12, int(R["failed_courses"]["median"]))
    jam = st.sidebar.slider("Jam belajar per minggu", 0.0, 30.0, round(R["weekly_study_hours"]["median"], 1), 0.5)
    lms = st.sidebar.slider("Skor aktivitas LMS", 0, 100, int(R["lms_activity_score"]["median"]))

    data = pd.DataFrame([{"attendance_pct": hadir, "gpa": gpa, "credits_completed": sks,
                          "failed_courses": gagal, "weekly_study_hours": jam,
                          "lms_activity_score": lms}])[meta["fitur"]]
    p_risiko = float(model.predict_proba(data)[0, 1])

    c1, c2 = st.columns([1, 2])
    c1.metric("Peluang berisiko terlambat", f"{p_risiko:.0%}")
    c1.progress(min(max(p_risiko, 0.0), 1.0))
    if p_risiko >= 0.5:
        c2.error("⚠️ **Berisiko tidak lulus tepat waktu.** Disarankan bimbingan akademik dini "
                 "dan pemantauan SKS serta mata kuliah yang gagal.")
    else:
        c2.success("✅ **Diprediksi lulus tepat waktu.** Pertahankan progres akademik.")
    with c2.expander("Data yang dimasukkan"):
        st.dataframe(data, hide_index=True)

with tab2:
    st.subheader("Perbandingan model pada data uji")
    st.dataframe(pd.DataFrame(meta["perbandingan_model"]).set_index("Model").round(3))
    st.markdown("Model dipilih berdasarkan **F2-score** hasil cross validation, karena "
                "melewatkan mahasiswa berisiko (false negative) lebih merugikan daripada "
                "memberi perhatian ekstra pada mahasiswa yang sebenarnya aman.")
    st.info("Model dilatih pada data sintetis; gunakan sebagai alat bantu, bukan keputusan akhir.")
