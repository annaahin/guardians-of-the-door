"""
영상 하나에서 사람의 움직임 기반 행동 특징을 추출하는 모듈.

이 모듈의 결과는 RandomForest/SVM 같은 분류 모델의 입력 데이터로 사용될
data/features.csv의 한 행(row)이 됩니다.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "video_name",
    "duration",
    "detected_ratio",
    "zone_ratio",
    "zone_stay_time",
    "zone_entry_count",
    "move_distance",
    "avg_speed",
    "direction_change",
    "max_box_area",
]


DEFAULT_DANGER_ZONE = {
    # 영상 좌표는 왼쪽 위가 (0, 0)이고, 아래로 갈수록 y 값이 커집니다.
    # 위험구역은 문 자체뿐 아니라 문 앞에 선 사람의 center point까지 포함하도록 잡습니다.
    "x1": 80,
    "y1": 450,
    "x2": 780,
    "y2": 1450,
}


def is_in_danger_zone(center_x, center_y, danger_zone):
    """
    사람의 center point가 위험구역 사각형 안에 있는지 확인합니다.

    center point는 bounding box 중앙 좌표입니다.
    위험구역은 문 앞처럼 특별히 주의해서 볼 화면 영역을 사각형으로 지정한 것입니다.
    """
    return (
        danger_zone["x1"] <= center_x <= danger_zone["x2"]
        and danger_zone["y1"] <= center_y <= danger_zone["y2"]
    )


def calculate_move_distance(center_points):
    """
    중심점들이 이동한 총 거리를 계산합니다.

    move_distance는 프레임마다 기록한 center point 사이의 직선 거리를 모두 더한 값입니다.
    단위는 픽셀(pixel)입니다.
    """
    if len(center_points) < 2:
        return 0.0

    distance = 0.0
    for previous_point, current_point in zip(center_points, center_points[1:]):
        distance += float(np.linalg.norm(np.array(current_point) - np.array(previous_point)))
    return distance


def calculate_direction_change(center_points, angle_threshold=45):
    """
    이동 방향이 크게 바뀐 횟수를 계산합니다.

    direction_change는 배회하거나 왔다 갔다 하는 행동을 단순하게 표현하기 위한 특징입니다.
    세 점 A, B, C가 있을 때 A->B 방향과 B->C 방향의 각도 차이가 angle_threshold보다 크면
    방향 전환 1회로 계산합니다.
    """
    if len(center_points) < 3:
        return 0

    direction_change = 0

    for first, second, third in zip(center_points, center_points[1:], center_points[2:]):
        vector1 = np.array(second) - np.array(first)
        vector2 = np.array(third) - np.array(second)

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        # 거의 움직이지 않은 경우에는 방향을 판단하기 어렵기 때문에 건너뜁니다.
        if norm1 == 0 or norm2 == 0:
            continue

        cosine = np.dot(vector1, vector2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        if angle >= angle_threshold:
            direction_change += 1

    return direction_change


def draw_debug_view(frame, person, danger_zone, in_danger_zone):
    """
    테스트용 화면 표시 함수입니다.

    - 사람 bounding box를 그립니다.
    - center point를 점으로 표시합니다.
    - 위험구역 사각형을 그립니다.
    - 위험구역 안에 있으면 빨간색, 아니면 초록색으로 표시합니다.
    """
    zone_color = (0, 0, 255) if in_danger_zone else (0, 255, 0)

    cv2.rectangle(
        frame,
        (danger_zone["x1"], danger_zone["y1"]),
        (danger_zone["x2"], danger_zone["y2"]),
        zone_color,
        2,
    )

    if person is not None:
        cv2.rectangle(frame, (person["x1"], person["y1"]), (person["x2"], person["y2"]), (255, 0, 0), 2)
        cv2.circle(frame, (person["center_x"], person["center_y"]), 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"person {person['confidence']:.2f}",
            (person["x1"], max(20, person["y1"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    cv2.putText(
        frame,
        "IN DANGER ZONE" if in_danger_zone else "OUT OF DANGER ZONE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        zone_color,
        2,
    )

    return frame


def extract_features_from_video(video_path, detector, danger_zone=None, show_video=False):
    """
    영상 파일 하나를 읽고 행동 특징 딕셔너리를 반환합니다.

    Args:
        video_path: 분석할 영상 파일 경로입니다.
        detector: modules.person_detector.PersonDetector 객체입니다.
        danger_zone: 위험구역 좌표 딕셔너리입니다.
        show_video: True이면 분석 중인 영상을 화면에 띄워 디버깅합니다.

    Returns:
        features.csv에 저장할 수 있는 특징값 딕셔너리입니다.
    """
    video_path = Path(video_path)
    danger_zone = danger_zone or DEFAULT_DANGER_ZONE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # FPS는 Frames Per Second의 약자로, 영상 1초에 몇 장의 frame이 들어 있는지를 뜻합니다.
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 일부 영상 파일은 FPS 값이 0으로 읽힐 수 있습니다. 이 경우 나눗셈 오류를 막기 위해 보정합니다.
    if fps <= 0:
        fps = 30.0

    duration = total_frames / fps if total_frames > 0 else 0.0

    processed_frames = 0
    detected_frames = 0
    danger_zone_frames = 0
    zone_entry_count = 0
    was_in_danger_zone = False
    center_points = []
    max_box_area = 0

    while True:
        ret, frame = cap.read()

        # ret이 False이면 더 이상 읽을 frame이 없다는 뜻입니다.
        if not ret:
            break

        processed_frames += 1
        person = detector.detect_main_person(frame)
        in_danger_zone = False

        if person is not None:
            detected_frames += 1
            center_x = person["center_x"]
            center_y = person["center_y"]
            center_points.append((center_x, center_y))
            max_box_area = max(max_box_area, person["box_area"])

            in_danger_zone = is_in_danger_zone(center_x, center_y, danger_zone)

            if in_danger_zone:
                danger_zone_frames += 1

            # 위험구역 밖에 있다가 안으로 들어오는 순간만 진입 횟수로 계산합니다.
            if in_danger_zone and not was_in_danger_zone:
                zone_entry_count += 1

            was_in_danger_zone = in_danger_zone
        else:
            # 사람이 안 보이는 프레임은 위험구역 안에 있다고 판단하지 않습니다.
            was_in_danger_zone = False

        if show_video:
            debug_frame = draw_debug_view(frame, person, danger_zone, in_danger_zone)
            cv2.imshow("Guardians of the Door - Feature Extraction", debug_frame)

            # q 키를 누르면 테스트 영상을 중간에 종료합니다.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    if show_video:
        cv2.destroyAllWindows()

    # detected_ratio는 전체 frame 중 사람이 감지된 frame의 비율입니다.
    # 사람이 영상에 얼마나 안정적으로 잡혔는지 확인하는 특징입니다.
    detected_ratio = detected_frames / processed_frames if processed_frames > 0 else 0.0

    # zone_ratio는 사람이 감지된 frame 중 center point가 위험구역 안에 있었던 비율입니다.
    # 문 앞 가까운 영역에 오래 머무르는 행동을 표현할 수 있습니다.
    zone_ratio = danger_zone_frames / detected_frames if detected_frames > 0 else 0.0

    # zone_stay_time은 사람이 위험구역 안에 머문 시간을 초 단위로 계산한 값입니다.
    # 위험구역 안에 있었던 frame 수를 FPS로 나누면 몇 초 동안 머물렀는지 알 수 있습니다.
    zone_stay_time = danger_zone_frames / fps if fps > 0 else 0.0

    move_distance = calculate_move_distance(center_points)

    # avg_speed는 평균 이동 속도입니다.
    # 여기서는 픽셀 단위 이동 거리 / 영상 길이(초)로 계산합니다.
    avg_speed = move_distance / duration if duration > 0 else 0.0

    direction_change = calculate_direction_change(center_points)

    return {
        "video_name": video_path.name,
        "duration": round(duration, 3),
        "detected_ratio": round(detected_ratio, 4),
        "zone_ratio": round(zone_ratio, 4),
        "zone_stay_time": round(zone_stay_time, 3),
        "zone_entry_count": zone_entry_count,
        "move_distance": round(move_distance, 3),
        "avg_speed": round(avg_speed, 3),
        "direction_change": direction_change,
        "max_box_area": max_box_area,
    }


def save_features_to_csv(features, output_path):
    """여러 영상에서 추출한 특징 리스트를 features.csv로 저장합니다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df.to_csv(output_path, index=False)
    return df
