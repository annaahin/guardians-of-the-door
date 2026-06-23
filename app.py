"""
Guardians of the Door - 시연용 Streamlit UI.

이 파일은 최종 발표/보고서 시연을 위한 웹 화면입니다.
기존 main.py의 전체 특징 추출 흐름은 그대로 두고, 사용자가 영상 하나를 선택하면
특징 추출 결과와 위험행동 예측 결과를 화면에서 확인할 수 있게 합니다.
"""

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import pandas as pd
import streamlit as st

from main import DEFAULT_DANGER_ZONE, VIDEO_EXTENSIONS, collect_video_files
from modules.feature_extractor import draw_debug_view, extract_features_from_video
from modules.person_detector import PersonDetector
from risk_classifier import FEATURE_COLUMNS, RiskClassifier


ALERT_LOG_PATH = Path("results/alert_history.csv")
ALERT_COLUMNS = [
    "time",
    "level",
    "video_name",
    "message",
    "zone_stay_time",
    "zone_ratio",
    "direction_change",
]


CLASS_LABELS = {
    "normal": "안심",
    "suspicious": "주의 필요",
    "dangerous": "위험 의심",
}


CLASS_COLORS = {
    "normal": "#157347",
    "suspicious": "#b25a00",
    "dangerous": "#b42318",
    "Error": "#555555",
}


RESULT_COPY = {
    "normal": {
        "headline": "현재 영상에서는 위험 행동이 뚜렷하게 보이지 않습니다.",
        "action": "기록 확인용으로 보관하고, 추가 조치는 필요하지 않은 상황으로 볼 수 있습니다.",
    },
    "suspicious": {
        "headline": "문 앞에서 머무르거나 살피는 행동이 감지되었습니다.",
        "action": "거주자가 직접 영상을 한 번 더 확인하고, 반복 발생 여부를 살펴보는 것이 좋습니다.",
    },
    "dangerous": {
        "headline": "문 근처에서 위험 가능성이 높은 행동 패턴이 감지되었습니다.",
        "action": "문 앞 상황을 즉시 확인하고, 필요하면 주변 사람이나 관리실에 공유하세요.",
    },
    "Error": {
        "headline": "예측 모델을 불러오지 못했습니다.",
        "action": "models/risk_model.pkl 파일이 있는지 확인하세요.",
    },
}


def load_alert_history():
    """이전에 저장된 주의/위험 알림 기록을 불러옵니다."""
    if not ALERT_LOG_PATH.exists():
        return []

    try:
        return pd.read_csv(ALERT_LOG_PATH).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def save_alert_history(alerts):
    """시연 중 발생한 알림을 CSV로 저장해 나중에 다시 열어볼 수 있게 합니다."""
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alerts, columns=ALERT_COLUMNS).to_csv(ALERT_LOG_PATH, index=False)


def initialize_alert_state():
    """Streamlit 세션에 알림 목록을 준비합니다."""
    if "alerts" not in st.session_state:
        st.session_state.alerts = load_alert_history()


def add_alert_if_needed(prediction, features):
    """
    suspicious/dangerous 결과만 알림함에 저장합니다.

    normal은 별도 조치가 필요 없는 확인 기록이므로 알림으로 쌓지 않습니다.
    """
    if prediction not in {"suspicious", "dangerous"}:
        return False

    copy = RESULT_COPY[prediction]
    alert = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": CLASS_LABELS[prediction],
        "video_name": features["video_name"],
        "message": copy["headline"],
        "zone_stay_time": features["zone_stay_time"],
        "zone_ratio": features["zone_ratio"],
        "direction_change": features["direction_change"],
    }

    st.session_state.alerts.insert(0, alert)
    save_alert_history(st.session_state.alerts)
    return True


def find_sample_videos(video_root):
    """data/videos 폴더에 있는 영상 목록을 UI 선택용으로 모읍니다."""
    video_files = collect_video_files(video_root)
    return [path for path in video_files if path.suffix.lower() in VIDEO_EXTENSIONS]


def save_uploaded_video(uploaded_file):
    """
    Streamlit에 업로드한 영상은 메모리 객체라 OpenCV가 바로 읽기 어렵습니다.
    그래서 임시 파일로 저장한 뒤 그 경로를 OpenCV에 넘깁니다.
    """
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


@st.cache_resource
def load_detector(model_path, confidence_threshold):
    """
    YOLO 모델은 로딩 시간이 있으므로 Streamlit cache에 올려 재사용합니다.
    confidence_threshold가 바뀌면 새 detector가 만들어집니다.
    """
    return PersonDetector(model_path=model_path, confidence_threshold=confidence_threshold)


@st.cache_resource
def load_classifier(model_path):
    """학습된 위험행동 분류 모델을 한 번만 불러옵니다."""
    return RiskClassifier(model_path=model_path)


def predict_from_features(features, classifier):
    """특징 딕셔너리를 모델 입력 순서에 맞는 리스트로 바꿔 예측합니다."""
    feature_values = [features[column] for column in FEATURE_COLUMNS]
    return classifier.predict_risk(feature_values)


def format_video_label(path):
    """영상 선택 박스에 너무 긴 경로 대신 클래스와 파일명 중심으로 보여줍니다."""
    path = Path(path)
    parts = path.parts
    class_name = ""
    for name in ("normal", "suspicious", "dangerous"):
        if name in parts:
            class_name = name
            break
    if class_name:
        return f"{class_name} / {path.name}"
    return path.name


def make_feature_table(features):
    """상세 보기용으로 특징값 설명과 실제 값을 함께 보여주는 표를 만듭니다."""
    descriptions = {
        "duration": "영상 전체 길이(초)",
        "detected_ratio": "전체 프레임 중 사람이 감지된 비율",
        "zone_ratio": "감지된 프레임 중 위험구역 안에 있었던 비율",
        "zone_stay_time": "위험구역 안에 머문 시간(초)",
        "zone_entry_count": "위험구역 진입 횟수",
        "move_distance": "사람 중심점의 총 이동 거리(pixel)",
        "avg_speed": "평균 이동 속도(pixel/sec)",
        "direction_change": "이동 방향 전환 횟수",
        "max_box_area": "사람 bounding box 최대 면적",
    }

    rows = []
    for column in FEATURE_COLUMNS:
        rows.append(
            {
                "특징값": column,
                "값": features[column],
                "의미": descriptions[column],
            }
        )
    return pd.DataFrame(rows)


def capture_debug_frames(video_path, detector, danger_zone, max_images=3):
    """
    보고서/시연 화면에 넣기 좋은 분석 캡처 이미지를 만듭니다.

    전체 영상을 다시 저장하지 않고, 영상 중간중간 몇 장의 frame만 뽑아서
    danger zone, bounding box, center point를 그립니다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # 처음/중간/끝 근처 frame을 샘플로 사용합니다.
    frame_indexes = sorted(
        {
            max(0, int(total_frames * 0.2)),
            max(0, int(total_frames * 0.5)),
            max(0, int(total_frames * 0.8)),
        }
    )[:max_images]

    images = []
    for frame_index in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        person = detector.detect_main_person(frame)
        in_danger_zone = False
        if person is not None:
            in_danger_zone = (
                danger_zone["x1"] <= person["center_x"] <= danger_zone["x2"]
                and danger_zone["y1"] <= person["center_y"] <= danger_zone["y2"]
            )

        debug_frame = draw_debug_view(frame, person, danger_zone, in_danger_zone)

        # OpenCV는 BGR 색상 순서를 쓰고, Streamlit은 RGB를 기대합니다.
        images.append(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return images


def render_result_card(prediction):
    """거주자가 바로 이해할 수 있는 문장 중심의 결과 카드를 표시합니다."""
    color = CLASS_COLORS.get(prediction, "#555555")
    label = CLASS_LABELS.get(prediction, prediction)
    copy = RESULT_COPY.get(prediction, RESULT_COPY["Error"])

    st.markdown(
        f"""
        <div class="result-card" style="border-left-color: {color};">
            <p class="result-title">현관 앞 행동 판단</p>
            <p class="result-label" style="color: {color};">{label}</p>
            <p class="result-headline">{copy["headline"]}</p>
            <p class="result-action">{copy["action"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(title, value, helper):
    """숫자 특징을 거주자용 판단 근거 카드로 보여줍니다."""
    st.markdown(
        f"""
        <div class="signal-card">
            <p class="signal-title">{title}</p>
            <p class="signal-value">{value}</p>
            <p class="signal-helper">{helper}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_center():
    """거주자가 나중에 열어볼 수 있는 주의/위험 알림함을 표시합니다."""
    alert_count = len(st.session_state.alerts)
    st.markdown(
        f"""
        <div class="alert-header">
            <span class="alert-icon">⚠️</span>
            <span class="alert-title">주의 알림함</span>
            <span class="alert-count">{alert_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("알림 내역 보기", expanded=False):
        if alert_count == 0:
            st.write("아직 주의 또는 위험 알림이 없습니다.")
            st.caption("분석 결과가 주의 필요 또는 위험 의심으로 판단되면 이곳에 기록됩니다.")
            return

        st.write("주의 필요 또는 위험 의심으로 판단된 현관 앞 상황 기록입니다.")
        st.dataframe(pd.DataFrame(st.session_state.alerts), use_container_width=True, hide_index=True)

        if st.button("알림 기록 비우기"):
            st.session_state.alerts = []
            save_alert_history(st.session_state.alerts)
            st.rerun()


def render_styles():
    """Streamlit 기본 화면을 시연용으로 조금 정돈합니다."""
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .main .block-container {
            max-width: 1120px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.8rem 0 1rem;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.4rem;
        }
        .brand-title {
            color: #4b5563;
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .app-title {
            font-size: 2.35rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
            color: #1f2937;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 1.05rem;
            margin-bottom: 0.2rem;
            line-height: 1.55;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 760;
            margin: 1.2rem 0 0.5rem;
            color: #1f2937;
        }
        .result-card {
            border-left: 6px solid #555555;
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            background: #ffffff;
            box-shadow: 0 1px 8px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.1rem;
        }
        .result-title {
            margin: 0;
            color: #6b7280;
            font-size: 0.9rem;
            font-weight: 650;
        }
        .result-label {
            margin: 0.25rem 0 0;
            font-size: 2rem;
            font-weight: 800;
        }
        .result-headline {
            margin: 0.55rem 0 0;
            color: #1f2937;
            font-size: 1.02rem;
            font-weight: 650;
            line-height: 1.5;
        }
        .result-action {
            margin: 0.35rem 0 0;
            color: #4b5563;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .signal-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            min-height: 132px;
            background: #fafafa;
        }
        .signal-title {
            margin: 0;
            color: #6b7280;
            font-size: 0.88rem;
            font-weight: 700;
        }
        .signal-value {
            margin: 0.35rem 0 0;
            color: #111827;
            font-size: 1.75rem;
            font-weight: 800;
        }
        .signal-helper {
            margin: 0.35rem 0 0;
            color: #4b5563;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .alert-header {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.3rem 0 0.5rem;
            color: #1f2937;
            font-size: 1rem;
            font-weight: 760;
        }
        .alert-icon {
            font-size: 1.15rem;
            line-height: 1;
        }
        .alert-count {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.55rem;
            height: 1.55rem;
            padding: 0 0.42rem;
            border-radius: 999px;
            background: #fee2e2;
            color: #b42318;
            font-weight: 850;
            font-size: 0.9rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="현관 앞 안심 확인", layout="wide")
    initialize_alert_state()
    render_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="brand-title">Guardians of the Door</div>
            <div class="app-title">현관 앞 안심 확인</div>
            <div class="app-subtitle">
                현관 앞 영상에서 사람이 문 근처에 얼마나 머물렀는지, 반복적으로 움직였는지 확인해
                거주자가 주의해야 할 상황인지 알려줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_alert_center()

    model_path = "yolov8n.pt"
    classifier_path = "models/risk_model.pkl"
    confidence_threshold = 0.3
    zone_x1 = DEFAULT_DANGER_ZONE["x1"]
    zone_y1 = DEFAULT_DANGER_ZONE["y1"]
    zone_x2 = DEFAULT_DANGER_ZONE["x2"]
    zone_y2 = DEFAULT_DANGER_ZONE["y2"]

    with st.expander("고급 설정 열기 - 문 앞 영역 및 모델 설정", expanded=False):
        st.caption(
            "문 앞 집중 확인 영역은 사람이 현관문 가까이에 머무는 시간을 계산하기 위한 내부 기준입니다. "
            "기본 화면에서는 숨겨두고, 시연 환경이나 카메라 구도가 달라질 때만 열어서 조정합니다."
        )
        model_path = st.text_input("YOLO 모델", value="yolov8n.pt")
        classifier_path = st.text_input("분류 모델", value="models/risk_model.pkl")
        confidence_threshold = st.slider("YOLO confidence", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
        st.caption(
            "YOLO confidence는 사람 감지의 최소 확신 기준입니다. "
            "값을 높이면 확실한 사람만 감지하지만 놓치는 장면이 늘 수 있고, "
            "값을 낮추면 더 많이 감지하지만 잘못 감지할 가능성도 늘어납니다."
        )

        st.divider()
        st.subheader("문 앞 집중 확인 영역")
        zone_x1 = st.number_input("x1", value=DEFAULT_DANGER_ZONE["x1"], step=10)
        zone_y1 = st.number_input("y1", value=DEFAULT_DANGER_ZONE["y1"], step=10)
        zone_x2 = st.number_input("x2", value=DEFAULT_DANGER_ZONE["x2"], step=10)
        zone_y2 = st.number_input("y2", value=DEFAULT_DANGER_ZONE["y2"], step=10)

    danger_zone = {
        "x1": int(zone_x1),
        "y1": int(zone_y1),
        "x2": int(zone_x2),
        "y2": int(zone_y2),
    }

    st.markdown('<div class="section-title">확인할 현관 영상</div>', unsafe_allow_html=True)
    input_mode = st.radio("영상 입력 방식", ["저장된 영상으로 시연", "내 영상 업로드"], horizontal=True)
    selected_video_path = None
    uploaded_video_name = None

    if input_mode == "저장된 영상으로 시연":
        sample_videos = find_sample_videos("data/videos")
        if sample_videos:
            selected = st.selectbox("확인할 영상", sample_videos, format_func=format_video_label)
            selected_video_path = Path(selected)
        else:
            st.warning("data/videos 폴더에서 영상 파일을 찾지 못했습니다. 영상 파일 업로드 방식을 사용하세요.")
    else:
        uploaded_file = st.file_uploader("현관 앞 영상 파일을 선택하세요.", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            selected_video_path = save_uploaded_video(uploaded_file)
            uploaded_video_name = uploaded_file.name

    analyze = st.button("현관 상황 확인하기", type="primary", disabled=selected_video_path is None)

    if selected_video_path is not None:
        st.markdown('<div class="section-title">입력 영상</div>', unsafe_allow_html=True)
        st.video(str(selected_video_path))

    if not analyze:
        return

    with st.spinner("현관 앞 행동을 확인하는 중입니다..."):
        detector = load_detector(model_path, confidence_threshold)
        classifier = load_classifier(classifier_path)
        features = extract_features_from_video(
            video_path=selected_video_path,
            detector=detector,
            danger_zone=danger_zone,
            show_video=False,
        )
        if uploaded_video_name is not None:
            features["video_name"] = uploaded_video_name

        prediction = predict_from_features(features, classifier)
        alert_saved = add_alert_if_needed(prediction, features)
        debug_images = capture_debug_frames(selected_video_path, detector, danger_zone)

    st.markdown('<div class="section-title">확인 결과</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.15], gap="large")

    with left:
        render_result_card(prediction)
        if alert_saved:
            st.warning("주의 알림함에 이번 상황이 저장되었습니다.")

        card_a, card_b = st.columns(2)
        with card_a:
            render_signal_card(
                "문 앞 머문 시간",
                f"{features['zone_stay_time']}초",
                "문 가까운 영역에 머문 시간을 계산했습니다.",
            )
        with card_b:
            render_signal_card(
                "문 근처 체류 비율",
                f"{features['zone_ratio']}",
                "사람이 보인 시간 중 문 근처에 있었던 비율입니다.",
            )

        card_c, card_d = st.columns(2)
        with card_c:
            render_signal_card(
                "움직임 변화",
                f"{features['direction_change']}회",
                "왔다 갔다 하거나 방향을 바꾼 정도를 봅니다.",
            )
        with card_d:
            render_signal_card(
                "문 근처 진입",
                f"{features['zone_entry_count']}회",
                "문 앞 영역 안으로 들어온 횟수입니다.",
            )

    with right:
        st.markdown("#### 판단 근거")
        st.write(
            "이 화면은 사람의 얼굴이나 음성을 인식하지 않고, 현관 앞에서의 위치와 움직임만 사용합니다. "
            "문 앞에 오래 머무르거나 문 근처를 반복적으로 오가는 행동일수록 주의 신호로 해석됩니다."
        )
        with st.expander("상세 분석값 보기"):
            st.dataframe(make_feature_table(features), use_container_width=True, hide_index=True)

    if debug_images:
        st.markdown('<div class="section-title">현관 확인 장면</div>', unsafe_allow_html=True)
        st.caption("초록/빨간 사각형은 문 앞 집중 확인 영역이며, 파란 박스는 감지된 사람입니다.")
        image_columns = st.columns(len(debug_images))
        for index, image in enumerate(debug_images):
            image_columns[index].image(image, caption=f"확인 장면 {index + 1}", use_container_width=True)


if __name__ == "__main__":
    main()
