import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class EnsembleStacking:
    def __init__(self, base_models, meta_classifier=None):
        self.base_models = base_models
        self.meta_classifier = meta_classifier if meta_classifier else LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            meta_features.append(proba)
        
        meta_X = np.column_stack(meta_features)
        meta_X_scaled = self.scaler.fit_transform(meta_X)
        
        self.meta_classifier.fit(meta_X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            meta_features.append(proba)
        
        meta_X = np.column_stack(meta_features)
        meta_X_scaled = self.scaler.transform(meta_X)
        
        return self.meta_classifier.predict(meta_X_scaled)
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            meta_features.append(proba)
        
        meta_X = np.column_stack(meta_features)
        meta_X_scaled = self.scaler.transform(meta_X)
        
        return self.meta_classifier.predict_proba(meta_X_scaled)
    
    def predict_with_confidence(self, X):
        proba = self.predict_proba(X)
        predictions = (proba[:, 1] > 0.5).astype(int)
        confidence = np.max(proba, axis=1)
        
        return predictions, confidence
    
    def get_individual_predictions(self, X):
        results = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            results[name] = proba
        
        return results
    
    def save(self, path):
        joblib.dump({
            'meta_classifier': self.meta_classifier,
            'scaler': self.scaler
        }, path)
    
    def load(self, path):
        data = joblib.load(path)
        self.meta_classifier = data['meta_classifier']
        self.scaler = data['scaler']
        self.is_fitted = True
        return self