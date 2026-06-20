import joblib
import numpy as np

class RiskClassifier:
    def __init__(self, model_path='models/risk_model.pkl'):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print("❌ 모델 파일 없음")
            self.model = None

    def predict_risk(self, features):
        """
        features 순서 (9개):
        [duration, detected_ratio, zone_ratio, zone_stay_time, zone_entry_count, 
         move_distance, avg_speed, direction_change, max_box_area]
        """
        if self.model is None:
            return "Error"
        
        input_data = np.array(features).reshape(1, -1)
        return self.model.predict(input_data)[0]

if __name__ == "__main__":
    classifier = RiskClassifier('models/risk_model.pkl') 
    
    # 순서: duration, det_ratio, z_ratio, z_stay, z_cnt, dist, speed, dir_chg, max_area
    test_normal = [5.0, 0.8, 0.1, 0.5, 0, 10.0, 2.0, 0, 5000]       
    test_dangerous = [30.0, 0.95, 0.9, 27.0, 1, 5.0, 0.2, 1, 30000]   
    
    print(f"정상 수치 예측: {classifier.predict_risk(test_normal)}")
    print(f"위험 수치 예측: {classifier.predict_risk(test_dangerous)}")