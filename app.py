"""Music Lens — Streamlit genre classifier UI.

Run locally:  streamlit run app.py
For real genre inference, add model_rf.joblib, scaler.joblib, and
label_encoder.joblib next to this file (the artifacts made in lesson 5/6).
"""

from __future__ import annotations

import os
import tempfile
import base64
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

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
        .stApp { background: radial-gradient(circle at 5% 0%, #272050 0, #101124 40%, #090a13 100%); color: #f4f3ff; }
        .block-container, [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 5rem; }
        [data-testid="stSidebar"] { background: #101122; border-right: 1px solid #292b48; }
        [data-testid="stSidebar"] * { color: #e8e7f5; }
        .hero { padding: .5rem 0 1.65rem; }
        .hero h1 { font-size: clamp(2.15rem, 4vw, 3.45rem); line-height: 1.12; letter-spacing: -0.055em; margin: 0; }
        .hero p { color: #b9b6d6; font-size: 1.02rem; line-height: 1.65; margin: .65rem 0 0; max-width: 42rem; }
        .eyebrow { color: #aea3ff; font-weight: 750; letter-spacing: .11em; font-size: .72rem; }
        h2 { font-size: 1.45rem !important; letter-spacing: -.035em; margin-top: 1.65rem !important; margin-bottom: .4rem !important; }
        h3 { font-size: 1.12rem !important; }
        .result-card { background: linear-gradient(135deg, #30286e, #1b1d43); border: 1px solid #51489a; box-shadow: 0 12px 30px rgba(0,0,0,.18); border-radius: 16px; padding: 1.15rem 1.25rem; }
        .result-card h2 { margin: .25rem 0 .2rem !important; font-size: 2.05rem !important; }
        .muted { color: #bbb8d9; }
        [data-testid="stMetric"] { background: rgba(29, 31, 58, .75); border: 1px solid #343653; border-radius: 12px; padding: .7rem .85rem; }
        [data-testid="stMetricLabel"] { font-size: .78rem; color: #aaa7c5; }
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        [data-testid="stDataFrame"] { border: 1px solid #343653; border-radius: 12px; overflow: hidden; }
        [data-testid="stExpander"] { background: rgba(23, 25, 47, .72); border: 1px solid #343653; border-radius: 12px; }
        .stButton > button, .stDownloadButton > button, .stLinkButton > a { border-radius: 9px; font-weight: 650; min-height: 2.45rem; }
        [data-testid="stAudio"] { border: 1px solid #343653; border-radius: 12px; overflow: hidden; }
        .stApp { background: #f2f4f6; color: #191f28; }
        .block-container, [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.75rem; padding-bottom: 5rem; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e8eb; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: #191f28 !important; opacity: 1 !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *, [data-testid="stSidebar"] small { color: #6b7684 !important; }
        [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] + div { color: #3182f6 !important; font-weight: 700; }
        .hero { padding: .35rem 0 1.75rem; }
        .hero h1 { color: #191f28; font-size: clamp(2.1rem, 4vw, 3.25rem); line-height: 1.14; letter-spacing: -.06em; margin: 0; }
        .hero p { color: #6b7684; font-size: 1.03rem; line-height: 1.65; margin: .65rem 0 0; max-width: 42rem; }
        .eyebrow { color: #3182f6; font-weight: 800; letter-spacing: .08em; font-size: .73rem; }
        h2 { color: #191f28; font-size: 1.42rem !important; letter-spacing: -.035em; margin-top: 2.1rem !important; margin-bottom: .35rem !important; }
        h3 { color: #191f28; font-size: 1.08rem !important; }
        .result-card { background: linear-gradient(135deg, #3b8cff, #2874e8); border: 0; box-shadow: 0 10px 24px rgba(49,130,246,.22); border-radius: 18px; padding: 1.2rem 1.35rem; }
        .result-card h2 { color: white; margin: .25rem 0 .15rem !important; font-size: 2.05rem !important; }
        .result-card .muted { color: rgba(255,255,255,.78); }
        .muted { color: #6b7684; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #e5e8eb; border-radius: 14px; box-shadow: 0 2px 7px rgba(0,0,0,.025); padding: .75rem .9rem; }
        [data-testid="stMetricLabel"] { font-size: .77rem; color: #6b7684; }
        [data-testid="stMetricValue"] { color: #191f28; font-size: 1.25rem; }
        [data-testid="stDataFrame"] { border: 1px solid #e5e8eb; border-radius: 14px; overflow: hidden; background: #fff; }
        [data-testid="stExpander"] { background: #fff; border: 1px solid #e5e8eb; border-radius: 12px; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #e5e8eb !important; border-radius: 14px; }
        .stButton > button, .stDownloadButton > button, .stLinkButton > a { border-radius: 10px; font-weight: 700; min-height: 2.45rem; }
        [data-testid="stAudio"] { background: #fff; border: 1px solid #e5e8eb; border-radius: 12px; overflow: hidden; }
        [data-testid="stTabs"] button { font-weight: 700; color: #6b7684; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #3182f6; }
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
    fig, ax = plt.subplots(figsize=(6.7, 3.45), dpi=130)
    image = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap="magma", ax=ax)
    fig.colorbar(image, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def music_profile(stats: dict[str, float]) -> dict[str, float]:
    """Turn measurable audio features into a friendly 0–100 visual profile."""
    bpm = min(stats["tempo"] / 200 * 100, 100)
    energy = min(stats["rms_mean"] / 0.20 * 100, 100)
    brightness = min(stats["spectral_centroid_mean"] / 5000 * 100, 100)
    rhythm = min((bpm * 0.7) + (stats["zero_crossing_rate_mean"] / 0.12 * 30), 100)
    return {"BPM": round(bpm, 1), "에너지": round(energy, 1), "밝기": round(brightness, 1), "리듬감": round(rhythm, 1)}


def make_radar_figure(results):
    # Matplotlib's Linux runtime may lack Korean fonts; use universal labels here.
    labels = ["BPM", "ENERGY", "BRIGHT", "RHYTHM"]
    profile_keys = ["BPM", "에너지", "밝기", "리듬감"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.0, 4.25), dpi=130, subplot_kw={"polar": True})
    colors = ["#9b8cff", "#55d6be", "#ffba6a", "#ff7c9d"]
    colors = ["#3182f6", "#00a86b", "#ff9500", "#f04452"]
    for index, result in enumerate(results):
        profile = music_profile(result["stats"])
        values = [profile[label] for label in profile_keys] + [profile[profile_keys[0]]]
        color = colors[index % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2, label=result["file"].name)
        ax.fill(angles, values, color=color, alpha=0.14)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75], ["25", "50", "75"], color="#aaa7c5", fontsize=8)
    ax.grid(color="#66618a", alpha=0.45)
    ax.set_facecolor("#17192f")
    fig.patch.set_facecolor("#17192f")
    ax.tick_params(colors="#f4f3ff", labelsize=9, pad=8)
    ax.set_yticks([25, 50, 75], ["25", "50", "75"], color="#8b95a1", fontsize=8)
    ax.grid(color="#dfe4ea", alpha=1)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.tick_params(colors="#333d4b", labelsize=9, pad=8)
    if len(results) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.1), frameon=False, labelcolor="#f4f3ff")
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.1), frameon=False, labelcolor="#333d4b")
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


def analyze_segments(result, segment_seconds: int = 3) -> list[dict[str, object]]:
    """Predict every short section so genre changes are visible over time."""
    y, sr = result["y"], result["sr"]
    segments = []
    for start in range(0, len(y), sr * segment_seconds):
        clip = y[start : start + sr * segment_seconds]
        if len(clip) < sr:
            continue
        vector, _ = extract_features(clip, sr)
        probabilities = predict(vector)
        if probabilities:
            genre, confidence = probabilities[0]
            segments.append({"시작(초)": start / sr, "끝(초)": min((start + len(clip)) / sr, len(y) / sr), "장르": genre, "신뢰도": confidence})
    return segments


def report_frame(results) -> pd.DataFrame:
    """Create a portable, one-row-per-file summary for CSV export."""
    rows = []
    for result in results:
        probabilities = result["probabilities"] or []
        top_genre, top_probability = probabilities[0] if probabilities else ("모델 없음", None)
        rows.append({
            "파일명": result["file"].name,
            "예측 장르": top_genre,
            "신뢰도": round(top_probability, 4) if top_probability is not None else None,
            "추정 BPM": round(result["stats"]["tempo"], 1),
            "길이(초)": round(len(result["y"]) / result["sr"], 1),
            "스펙트럼 중심(Hz)": round(result["stats"]["spectral_centroid_mean"], 1),
            "RMS 에너지": round(result["stats"]["rms_mean"], 4),
        })
    return pd.DataFrame(rows)


def report_download(results, key: str) -> None:
    report = report_frame(results)
    st.download_button(
        "📥 분석 결과 CSV 다운로드",
        data=report.to_csv(index=False).encode("utf-8-sig"),
        file_name="music-lens-analysis.csv",
        mime="text/csv",
        key=key,
    )


def render_radar(results) -> None:
    st.subheader("오디오 특징 프로필")
    st.caption("BPM, 에너지, 밝기, 리듬감을 0~100으로 정규화한 비교 지표입니다.")
    chart_column, profile_column = st.columns([1, 1.15], gap="large")
    with chart_column:
        figure = make_radar_figure(results)
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
    with profile_column:
        st.markdown("##### 한눈에 보기")
        for result in results[:4]:
            profile = music_profile(result["stats"])
            with st.container(border=True):
                st.caption(result["file"].name)
                metrics = st.columns(2)
                for metric, (label, value) in zip(metrics * 2, profile.items()):
                    metric.metric(label, f"{value:.0f}")


def render_timeline(result) -> None:
    st.subheader("구간별 장르 분석")
    if load_models() is None:
        st.info("학습 모델 파일 3개를 추가하면 3초 단위 장르 타임라인을 볼 수 있습니다.")
        return
    segments = analyze_segments(result)
    if not segments:
        st.info("분석할 수 있는 구간이 충분하지 않습니다.")
        return
    timeline = pd.DataFrame(segments)
    timeline["구간"] = timeline.apply(lambda row: f"{row['시작(초)']:.0f}–{row['끝(초)']:.0f}초", axis=1)
    display = timeline[["구간", "장르", "신뢰도"]].copy()
    display["신뢰도"] = display["신뢰도"].map(lambda value: f"{value:.0%}")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.dataframe(display, hide_index=True, use_container_width=True, height=310)
    with right:
        st.caption("구간별 예측 신뢰도")
        st.bar_chart(timeline.set_index("구간")[["신뢰도"]], color="#9b8cff", height=270)
        st.bar_chart(timeline.set_index("구간")[["신뢰도"]], color="#3182f6", height=270)


def share_payload(result) -> str:
    probabilities = result["probabilities"] or []
    payload = {
        "file": result["file"].name,
        "genre": probabilities[0][0] if probabilities else "모델 없음",
        "confidence": round(probabilities[0][1], 4) if probabilities else None,
        "profile": music_profile(result["stats"]),
        "bpm": round(result["stats"]["tempo"], 1),
    }
    return base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode().rstrip("=")


def render_share_link(result) -> None:
    token = share_payload(result)
    parsed = urlsplit(st.context.url)
    base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    link = f"{base_url}?{urlencode({'view': 'shared', 'data': token})}"
    st.subheader("결과 공유")
    st.caption("오디오 파일은 포함하지 않고, 분석 요약만 담긴 링크입니다.")
    st.link_button("↗ 공유 결과 열기", link)
    with st.expander("공유 링크 보기 · 복사"):
        st.code(link, language=None)


def render_feedback(result) -> None:
    st.subheader("예측 피드백")
    with st.container(border=True):
        answer = st.radio("이 예측이 맞나요?", ["맞아요", "아니에요"], horizontal=True, key=f"feedback_answer_{result['file'].name}")
        correction = ""
        if answer == "아니에요":
            correction = st.selectbox("실제 장르를 선택하세요", GENRES, key=f"feedback_genre_{result['file'].name}")
        actions = st.columns([1, 3])
        with actions[0]:
            save = st.button("피드백 저장", key=f"feedback_save_{result['file'].name}")
        with actions[1]:
            if st.session_state.feedback:
                st.download_button(
                    "📥 피드백 CSV",
                    data=pd.DataFrame(st.session_state.feedback).to_csv(index=False).encode("utf-8-sig"),
                    file_name="music-lens-feedback.csv",
                    mime="text/csv",
                    key="download_feedback",
                )
        if save:
            probabilities = result["probabilities"] or []
            st.session_state.feedback.append({
                "파일명": result["file"].name,
                "예측 장르": probabilities[0][0] if probabilities else "모델 없음",
                "예측 신뢰도": round(probabilities[0][1], 4) if probabilities else None,
                "피드백": answer,
                "실제 장르": correction or None,
            })
            st.success("피드백을 저장했습니다.")


def decode_share_payload(token: str) -> dict[str, object] | None:
    try:
        padded = token + "=" * (-len(token) % 4)
        return json.loads(base64.urlsafe_b64decode(padded).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def render_shared_result(payload: dict[str, object]) -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>MUSIC LENS · SHARED RESULT</div><h1>음악 분석 결과</h1></div>", unsafe_allow_html=True)
    confidence = payload.get("confidence")
    confidence_text = f" · 신뢰도 {float(confidence):.1%}" if confidence is not None else ""
    st.success(f"🎧 {payload.get('file', '오디오')} · **{str(payload.get('genre', '-')).upper()}**{confidence_text}")
    profile = payload.get("profile", {})
    if isinstance(profile, dict):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, profile.items()):
            column.metric(str(label), f"{float(value):.0f}/100")
    st.caption(f"추정 BPM: {payload.get('bpm', '-')}")
    parsed = urlsplit(st.context.url)
    st.link_button("내 음악 분석하기", urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")))


def render_single_result(result, top_n: int) -> None:
    uploaded, y, sr = result["file"], result["y"], result["sr"]
    stats, probabilities = result["stats"], result["probabilities"]
    st.audio(uploaded.getvalue(), format=uploaded.type or "audio/wav")
    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.subheader("장르 예측 결과")
        if probabilities:
            genre, confidence = probabilities[0]
            st.markdown(f'<div class="result-card"><span class="muted">TOP MATCH</span><h2>{GENRE_ICON.get(genre, "🎵")} {genre.upper()}</h2><p class="muted">신뢰도 {confidence:.1%}</p></div>', unsafe_allow_html=True)
            table = pd.DataFrame(probabilities[:top_n], columns=["장르", "확률"])
            st.bar_chart(table.set_index("장르"), color="#9b8cff", height=250)
            st.bar_chart(table.set_index("장르"), color="#3182f6", height=250)
        else:
            st.warning("모델 파일 3개를 추가하면 실제 장르 예측이 표시됩니다.")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("추정 BPM", f"{stats['tempo']:.0f}")
        metric_b.metric("길이", f"{len(y) / sr:.1f}초")
        st.caption(f"스펙트럼 중심: {stats['spectral_centroid_mean']:.0f} Hz · RMS 에너지: {stats['rms_mean']:.3f}")
        report_download([result], key=f"download_{uploaded.name}")
    with right:
        st.subheader("멜스펙트로그램")
        figure = make_mel_figure(y, sr, uploaded.name)
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
    render_radar([result])
    render_timeline(result)
    feedback_column, share_column = st.columns(2, gap="large")
    with feedback_column:
        render_feedback(result)
    with share_column:
        render_share_link(result)


def render_lesson6() -> None:
    st.markdown("""<div class="hero"><div class="eyebrow">LESSON 6 · STREAMLIT BASICS</div>
    <h1>음악 장르 예측기</h1><p>오디오 한 곡을 올리고, AI의 Top-3 예측과 멜스펙트로그램을 확인하세요.</p></div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "오디오 파일 업로드",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="WAV, MP3, M4A, FLAC, OGG 파일을 올릴 수 있습니다.",
        key="lesson6_upload",
    )
    if uploaded:
        try:
            with st.spinner("음악을 분석하는 중..."):
                render_single_result(analyze(uploaded), top_n=3)
        except Exception as error:
            st.error(f"오디오를 읽지 못했습니다: {error}")
    else:
        st.info("6강 실습: 분석할 오디오 파일 하나를 업로드하세요.")


def render_lesson7() -> None:
    st.markdown("""<div class="hero"><div class="eyebrow">LESSON 7 · STREAMLIT ADVANCED</div>
    <h1>장르 예측기 v2</h1><p>여러 곡의 신뢰도를 비교하고, 예측 이력을 관리하세요.</p></div>""", unsafe_allow_html=True)
    prediction_tab, history_tab = st.tabs(["🎵 멀티파일 예측", "📋 예측 이력"])
    with prediction_tab:
        uploads = st.file_uploader(
            "오디오 파일 업로드 (여러 개 가능)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            accept_multiple_files=True,
            key="lesson7_upload",
        )
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
                summary = report_frame(results)
                metrics = summary[["파일명", "추정 BPM", "길이(초)", "스펙트럼 중심(Hz)", "RMS 에너지"]]
                st.dataframe(metrics, hide_index=True, use_container_width=True)
                render_radar(results)
                report_download(results, key="download_comparison")
                if all(item["probabilities"] for item in results):
                    comparison = pd.DataFrame({item["file"].name: dict(item["probabilities"]) for item in results}).fillna(0)
                    st.bar_chart(comparison, color=["#9b8cff", "#55d6be", "#ffba6a", "#ff7c9d"], height=310)
                    st.bar_chart(comparison, color=["#3182f6", "#00a86b", "#ff9500", "#f04452"], height=310)
                    rows = [{"파일명": item["file"].name, "1위 장르": item["probabilities"][0][0], "신뢰도": f"{item['probabilities'][0][1]:.1%}"} for item in results]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.warning("모델 파일을 추가하면 장르 확률 비교 차트도 표시됩니다.")
        else:
            st.info("7강 실습: 여러 오디오 파일을 업로드해 비교해 보세요.")
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
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    query = st.query_params
    if query.get("view") == "shared" and query.get("data"):
        payload = decode_share_payload(query["data"])
        if payload:
            render_shared_result(payload)
        else:
            st.error("유효하지 않거나 손상된 공유 링크입니다.")
        return
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