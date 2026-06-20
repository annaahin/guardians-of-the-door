"""
Guardians of the Door - 영상 기반 인물 탐지 및 행동 특징 추출 실행 파일.

기본 실행:
    python main.py

영상 하나만 테스트:
    python main.py --video data/videos/normal/normal_001.mp4 --show
"""

import argparse
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
DEFAULT_VIDEO_ROOT = Path("data/videos")
DEFAULT_OUTPUT_CSV = Path("data/features.csv")
DEFAULT_DANGER_ZONE = {
    # 영상 좌표는 왼쪽 위가 (0, 0)이고, 아래로 갈수록 y 값이 커집니다.
    # 위험구역은 문 자체뿐 아니라 문 앞에 선 사람의 center point까지 포함하도록 잡습니다.
    "x1": 80,
    "y1": 450,
    "x2": 780,
    "y2": 1450,
}


def collect_video_files(video_root):
    """
    data/videos/normal, data/videos/suspicious, data/videos/dangerous 안의 영상들을 모읍니다.

    클래스 폴더 안에 행동별 하위 폴더가 있어도 처리할 수 있도록 모든 하위 폴더를 재귀적으로 탐색합니다.
    예:
        data/videos/normal/1. 문 앞 지나가기/sample.mp4
        data/videos/suspicious/문 앞 10초 이상 서 있기/sample.mp4

    label은 features.csv에 넣지 않습니다.
    labels.csv와 나중에 합칠 기준은 video_name 컬럼입니다.
    """
    video_root = Path(video_root)
    class_folders = ["normal", "suspicious", "dangerous"]
    video_files = []

    for class_name in class_folders:
        class_dir = video_root / class_name
        if not class_dir.exists():
            print(f"[WARN] Folder not found, skipped: {class_dir}")
            continue

        for video_path in sorted(class_dir.rglob("*")):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(video_path)

    return video_files


def warn_duplicate_video_names(video_files):
    """
    features.csv에는 video_name으로 파일명만 저장합니다.
    서로 다른 폴더에 같은 파일명이 있으면 labels.csv와 합칠 때 헷갈릴 수 있으므로 경고합니다.
    """
    seen = {}
    duplicated = []

    for video_path in video_files:
        if video_path.name in seen:
            duplicated.append(video_path.name)
        else:
            seen[video_path.name] = video_path

    if duplicated:
        print("[WARN] Duplicate video file names found.")
        print("[WARN] Please rename them because features.csv uses video_name as the merge key.")
        for name in sorted(set(duplicated)):
            print(f"  - {name}")


def process_single_video(video_path, detector, danger_zone, show_video=False):
    """테스트용으로 영상 하나만 분석합니다."""
    from modules.feature_extractor import extract_features_from_video

    print(f"[INFO] Processing one video: {video_path}")
    features = extract_features_from_video(
        video_path=video_path,
        detector=detector,
        danger_zone=danger_zone,
        show_video=show_video,
    )
    print("[INFO] Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    return features


def process_all_videos(video_root, output_csv, detector, danger_zone, show_video=False):
    """전체 영상 폴더를 순회하면서 features.csv를 생성합니다."""
    from modules.feature_extractor import extract_features_from_video, save_features_to_csv

    video_files = collect_video_files(video_root)
    warn_duplicate_video_names(video_files)

    if not video_files:
        print(f"[WARN] No video files found in {video_root}")
        save_features_to_csv([], output_csv)
        print(f"[INFO] Empty CSV created: {output_csv}")
        return []

    all_features = []

    for index, video_path in enumerate(video_files, start=1):
        print(f"[INFO] ({index}/{len(video_files)}) Processing: {video_path}")
        features = extract_features_from_video(
            video_path=video_path,
            detector=detector,
            danger_zone=danger_zone,
            show_video=show_video,
        )
        all_features.append(features)

    save_features_to_csv(all_features, output_csv)
    print(f"[INFO] Saved features CSV: {output_csv}")
    return all_features


def parse_args():
    """명령행 옵션을 읽습니다."""
    parser = argparse.ArgumentParser(description="Extract person movement features from entrance videos.")
    parser.add_argument("--video", type=str, default=None, help="Analyze only one video file.")
    parser.add_argument("--video-root", type=str, default=str(DEFAULT_VIDEO_ROOT), help="Root folder containing videos.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Output features.csv path.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--confidence", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--show", action="store_true", help="Show video with bounding boxes and danger zone.")
    parser.add_argument("--zone-x1", type=int, default=DEFAULT_DANGER_ZONE["x1"], help="Danger zone left x.")
    parser.add_argument("--zone-y1", type=int, default=DEFAULT_DANGER_ZONE["y1"], help="Danger zone top y.")
    parser.add_argument("--zone-x2", type=int, default=DEFAULT_DANGER_ZONE["x2"], help="Danger zone right x.")
    parser.add_argument("--zone-y2", type=int, default=DEFAULT_DANGER_ZONE["y2"], help="Danger zone bottom y.")
    return parser.parse_args()


def main():
    args = parse_args()

    from modules.feature_extractor import save_features_to_csv
    from modules.person_detector import PersonDetector

    danger_zone = {
        "x1": args.zone_x1,
        "y1": args.zone_y1,
        "x2": args.zone_x2,
        "y2": args.zone_y2,
    }

    detector = PersonDetector(model_path=args.model, confidence_threshold=args.confidence)

    if args.video:
        features = process_single_video(args.video, detector, danger_zone, show_video=args.show)
        save_features_to_csv([features], args.output)
        print(f"[INFO] Saved single-video features CSV: {args.output}")
    else:
        process_all_videos(args.video_root, args.output, detector, danger_zone, show_video=args.show)


if __name__ == "__main__":
    main()
