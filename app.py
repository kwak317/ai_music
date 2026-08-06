"""Music Lens — Streamlit genre classifier UI.

Run locally:  streamlit run app.py
For real genre inference, add model_rf.joblib, scaler.joblib, and
label_encoder.joblib next to this file (the artifacts made in lesson 5/6).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GENRE_ICON = {
    "blues": "🎸", "classical": "🎻", "country": "🤠", "disco": "🪩",
    "hiphop": "🎤", "jazz": "🎷", "metal": "🤘", "pop": "🎵",
    "reggae": "🌴", "rock": "⚡",
}
FEATURE_COLS = [
    "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean",
    "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
] + [f"mfcc{i}_{stat}" for i in range(1, 21) for stat in ("mean", "var")]

st.set_page_config(page_title="Music Lens | 장르 분석", page_icon="🎧", layout="wide")


def inject_style() -> None:
    st.markdown(
        """<style>
        .stApp { background: radial-gradient(circle at 8% 0%, #26204e 0, #101124 38%, #090a13 100%); color: #f4f3ff; }
        [data-testid="stSidebar"] { background: #121326; }
        .hero { padding: 2.2rem 0 1.2rem; }
        .hero h1 { font-size: clamp(2.3rem, 5vw, 4.6rem); letter-spacing: -0.06em; margin: 0; }
        .hero p { color: #b9b6d6; font-size: 1.12rem; margin-top: .65rem; }
        .eyebrow { color: #a79aff; font-weight: 700; letter-spacing: .1em; font-size: .78rem; }
        .result-card { background: linear-gradient(135deg, #2d2666, #171a39); border: 1px solid #4a4485; border-radius: 18px; padding: 1.4rem; }
        .result-card h2 { margin: 0; font-size: 2rem; }
        .muted { color: #aaa7c5; }
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models():
    root = Path(__file__).parent
    files = [root / "model_rf.joblib", root / "label_encoder.joblib", root / "scaler.joblib"]
    if not all(path.exists() for path in files):
        return None
    return tuple(joblib.load(path) for path in files)


def extract_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, float]]:
    values: dict[str, float] = {}
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, percussion = librosa.effects.hpss(y)
    named = {
        "chroma_stft": chroma, "rms": rms, "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth, "rolloff": rolloff,
        "zero_crossing_rate": zcr, "harmony": harmony, "perceptr": percussion,
    }
    for name, data in named.items():
        values[f"{name}_mean"] = float(np.mean(data))
        values[f"{name}_var"] = float(np.var(data))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    values["tempo"] = float(np.asarray(tempo).item())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for index in range(20):
        values[f"mfcc{index + 1}_mean"] = float(np.mean(mfcc[index]))
        values[f"mfcc{index + 1}_var"] = float(np.var(mfcc[index]))
    return np.array([values[name] for name in FEATURE_COLS], dtype=np.float32).reshape(1, -1), values


def make_mel_figure(y: np.ndarray, sr: int, title: str):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap="magma", ax=ax)
    fig.colorbar(image, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def predict(vector: np.ndarray):
    models = load_models()
    if models is None:
        return None
    classifier, encoder, scaler = models
    probabilities = classifier.predict_proba(scaler.transform(vector))[0]
    pairs = sorted(zip(encoder.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
    return [(str(genre), float(probability)) for genre, probability in pairs]


def read_audio(uploaded_file) -> tuple[np.ndarray, int]:
    """Decode one uploaded file, always removing the temporary file."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name
    try:
        y, sr = librosa.load(temp_path, sr=22050, mono=True, duration=30)
    finally:
        os.unlink(temp_path)
    if y.size == 0:
        raise ValueError("오디오 데이터가 비어 있습니다.")
    return y, sr


def add_history(name: str, probabilities) -> None:
    if not probabilities:
        return
    top_genre, top_probability = probabilities[0]
    second_genre = probabilities[1][0] if len(probabilities) > 1 else "-"
    row = {
        "파일명": name,
        "1위 장르": top_genre,
        "신뢰도": f"{top_probability:.1%}",
        "2위 장르": second_genre,
        "판정": "높음" if top_probability >= 0.5 else "확인 필요",
    }
    key = (name, row["1위 장르"], row["신뢰도"])
    if key not in st.session_state.history_keys:
        st.session_state.history.append(row)
        st.session_state.history_keys.add(key)


def analyze(uploaded_file):
    y, sr = read_audio(uploaded_file)
    vector, stats = extract_features(y[: sr * 3], sr)
    return {"file": uploaded_file, "y": y, "sr": sr, "stats": stats, "probabilities": predict(vector)}


def render_single_result(result, top_n: int) -> None:
    uploaded, y, sr = result["file"], result["y"], result["sr"]
    stats, probabilities = result["stats"], result["probabilities"]
    st.audio(uploaded.getvalue(), format=uploaded.type or "audio/wav")
    left, right = st.columns([1.0, 1.25], gap="large")
    with left:
        st.subheader("장르 예측 결과")
        if probabilities:
            genre, confidence = probabilities[0]
            st.markdown(f'<div class="result-card"><span class="muted">TOP MATCH</span><h2>{GENRE_ICON.get(genre, "🎵")} {genre.upper()}</h2><p class="muted">신뢰도 {confidence:.1%}</p></div>', unsafe_allow_html=True)
            table = pd.DataFrame(probabilities[:top_n], columns=["장르", "확률"])
            st.bar_chart(table.set_index("장르"), color="#9b8cff")
        else:
            st.warning("모델 파일 3개를 추가하면 실제 장르 예측이 표시됩니다.")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("추정 BPM", f"{stats['tempo']:.0f}")
        metric_b.metric("길이", f"{len(y) / sr:.1f}초")
        st.caption(f"스펙트럼 중심: {stats['spectral_centroid_mean']:.0f} Hz · RMS 에너지: {stats['rms_mean']:.3f}")
    with right:
        st.subheader("멜스펙트로그램")
        figure = make_mel_figure(y, sr, uploaded.name)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)


def render_lesson6() -> None:
    st.markdown("""<div class="hero"><div class="eyebrow">LESSON 6 · STREAMLIT BASICS</div>
    <h1>음악 장르 예측기</h1><p>WAV 파일 한 곡을 올리고, AI의 Top-3 예측과 멜스펙트로그램을 확인하세요.</p></div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("WAV 파일 업로드", type=["wav"], key="lesson6_upload")
    if uploaded:
        try:
            with st.spinner("음악을 분석하는 중..."):
                render_single_result(analyze(uploaded), top_n=3)
        except Exception as error:
            st.error(f"오디오를 읽지 못했습니다: {error}")
    else:
        st.info("6강 실습: WAV 파일 하나를 업로드하세요.")


def render_lesson7() -> None:
    st.markdown("""<div class="hero"><div class="eyebrow">LESSON 7 · STREAMLIT ADVANCED</div>
    <h1>장르 예측기 v2</h1><p>여러 곡의 신뢰도를 비교하고, 예측 이력을 관리하세요.</p></div>""", unsafe_allow_html=True)
    prediction_tab, history_tab = st.tabs(["🎵 멀티파일 예측", "📋 예측 이력"])
    with prediction_tab:
        uploads = st.file_uploader("오디오 파일 업로드 (여러 개 가능)", type=["wav"], accept_multiple_files=True, key="lesson7_upload")
        if uploads:
            results = []
            with st.spinner("여러 곡을 분석하는 중..."):
                for uploaded in uploads:
                    try:
                        result = analyze(uploaded)
                        results.append(result)
                        add_history(uploaded.name, result["probabilities"])
                    except Exception as error:
                        st.error(f"{uploaded.name}: {error}")
            if len(results) == 1:
                render_single_result(results[0], top_n=10)
            elif results:
                st.subheader(f"{len(results)}개 곡 장르 확률 비교")
                if all(item["probabilities"] for item in results):
                    comparison = pd.DataFrame({item["file"].name: dict(item["probabilities"]) for item in results}).fillna(0)
                    st.bar_chart(comparison, color=["#9b8cff", "#55d6be", "#ffba6a", "#ff7c9d"])
                    rows = [{"파일명": item["file"].name, "1위 장르": item["probabilities"][0][0], "신뢰도": f"{item['probabilities'][0][1]:.1%}"} for item in results]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.warning("비교 차트는 모델 파일 3개를 추가하면 표시됩니다.")
        else:
            st.info("7강 실습: WAV 파일을 여러 개 업로드해 비교해 보세요.")
    with history_tab:
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
            if st.button("이력 초기화", key="clear_history"):
                st.session_state.history = []
                st.session_state.history_keys = set()
                st.rerun()
        else:
            st.info("예측 결과가 아직 없습니다.")


def main() -> None:
    inject_style()
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.history_keys = set()
    with st.sidebar:
        st.header("MUSIC LENS")
        st.caption("AI Music lessons")
        page = st.radio("강의 선택", ["6강 · 기본 장르 예측", "7강 · 고급 비교"], label_visibility="collapsed")
        st.divider()
        st.write("3초 구간의 리듬·음색·주파수 피처 57개를 사용합니다.")
        st.caption("지원 장르: " + " · ".join(GENRES))
        if load_models() is None:
            st.warning("모델 파일이 없어서 분석 화면 모드입니다.")
        else:
            st.success("학습 모델 연결됨")
    if page.startswith("6강"):
        render_lesson6()
    else:
        render_lesson7()


if __name__ == "__main__":
    main()
