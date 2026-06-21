import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 결과물 및 모델 저장용 폴더 자동 생성
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_and_preprocess_real_data(features_path='data/features.csv', labels_path='data/labels.csv'):
    print("실제 데이터셋을 불러오는 중...")
    
    # 1. 특징 데이터 읽기
    df_features = pd.read_csv(features_path, encoding='cp949')
    
    # 2. 라벨 데이터 읽기 (header=2 로 세팅하여 3번째 줄을 컬럼명으로 사용)
    df_labels = pd.read_csv(labels_path, header=2, encoding='cp949')
    
    # 3. 파일명만 추출하여 'video_name' 열 생성
    df_labels['video_name'] = df_labels['video_path'].apply(lambda x: str(x).split('/')[-1].split('\\')[-1])
    
    # 4. 데이터 병합
    df_merged = pd.merge(df_features, df_labels, on='video_name')
    print(f"데이터 병합 완료. 총 샘플 수: {len(df_merged)}개")
    
    # 5. 특징 컬럼 선택
    feature_cols = [
        'duration', 'detected_ratio', 'zone_ratio', 'zone_stay_time', 
        'zone_entry_count', 'move_distance', 'avg_speed', 'direction_change', 'max_box_area'
    ]
    
    X = df_merged[feature_cols]
    
    # [핵심] 마법의 코드: 띄어쓰기 제거 및 소문자 변환
    y = df_merged['label'].astype(str).str.strip().str.lower()
    
    # 6. 학습/테스트 셋 분리 (반드시 X와 y를 깔끔하게 다듬은 '직후'에 분리해야 개수가 맞아!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("실제 데이터로 모델 학습을 시작합니다...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("모델 학습 완료.")
    return model

def evaluate_model(model, X_test, y_test, output_dir='results'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n실제 데이터 모델 평가 결과")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    
    with open(f'{output_dir}/model_result.txt', 'w') as f:
        f.write(f"Real Data Model Accuracy: {acc:.4f}\n\n")
        f.write(f"Classification Report:\n{report}")
        
    labels = ['normal', 'suspicious', 'dangerous']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Real Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    print(f"Confusion Matrix가 '{output_dir}/confusion_matrix.png'에 저장되었습니다.")

def save_model(model, filepath='models/risk_model.pkl'):
    joblib.dump(model, filepath)
    print(f"위험행동 분류 모델 저장 완료: {filepath}")

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_real_data('data/features.csv', 'data/labels.csv')
        trained_model = train_model(X_train, y_train)
        evaluate_model(trained_model, X_test, y_test)
        save_model(trained_model, 'models/risk_model.pkl')
    except FileNotFoundError as e:
        print(f"파일 경로 오류 발생: {e}")
        print("features.csv와 labels.csv 파일이 'data' 폴더 안에 있는지 확인해 주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")