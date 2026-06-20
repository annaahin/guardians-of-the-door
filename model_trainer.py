import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def create_dummy_data(filepath='data/features_with_label_dummy.csv', num_samples=300):
    """정의된 feature 형식에 맞춘 테스트용 가짜 데이터 생성"""
    np.random.seed(42)
    data = []
    
    for i in range(num_samples):
        video_name = f"video_{i:03d}"
        label_idx = np.random.choice([0, 1, 2]) # 0: normal, 1: suspicious, 2: dangerous
        
        if label_idx == 0: # Normal
            dur, det_r, z_r, z_stay, z_cnt = 5.0, 0.8, 0.1, 0.5, 0
            dist, speed, dir_chg, max_area = 10.0, 2.0, 0, 5000
            label = 'normal'
        elif label_idx == 1: # Suspicious
            dur, det_r, z_r, z_stay, z_cnt = 20.0, 0.9, 0.6, 12.0, 3
            dist, speed, dir_chg, max_area = 30.0, 1.5, 4, 12000
            label = 'suspicious'
        else: # Dangerous
            dur, det_r, z_r, z_stay, z_cnt = 30.0, 0.95, 0.9, 27.0, 1
            dist, speed, dir_chg, max_area = 5.0, 0.2, 1, 30000
            label = 'dangerous'
            
        # 약간의 노이즈 추가
        data.append([
            video_name,
            abs(dur + np.random.normal(0, 2)),
            min(1.0, abs(det_r + np.random.normal(0, 0.1))),
            min(1.0, abs(z_r + np.random.normal(0, 0.1))),
            abs(z_stay + np.random.normal(0, 2)),
            abs(z_cnt + np.random.randint(-1, 2)),
            abs(dist + np.random.normal(0, 5)),
            abs(speed + np.random.normal(0, 0.5)),
            abs(dir_chg + np.random.randint(-1, 2)),
            abs(max_area + np.random.normal(0, 1000)),
            label
        ])
        
    columns = ['video_name', 'duration', 'detected_ratio', 'zone_ratio', 'zone_stay_time', 
               'zone_entry_count', 'move_distance', 'avg_speed', 'direction_change', 'max_box_area', 'label']
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)
    print(f"테스트용 행동 특징 데이터 생성 완료: {filepath}")

def load_and_preprocess_dummy_data(filepath='data/features_with_label_dummy.csv'):
    df = pd.read_csv(filepath)
    X = df.drop(['video_name', 'label'], axis=1) # 학습할 때 이름과 정답지는 뺌
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, output_dir='results'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    with open(f'{output_dir}/model_result.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
    labels = ['normal', 'suspicious', 'dangerous']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def save_model(model, filepath='models/risk_model.pkl'):
    joblib.dump(model, filepath)

if __name__ == "__main__":
    create_dummy_data()
    X_train, X_test, y_train, y_test = load_and_preprocess_dummy_data()
    
    trained_model = train_model(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    save_model(trained_model)
