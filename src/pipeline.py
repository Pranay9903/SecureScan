import numpy as np
import pandas as pd
import joblib
from src.features import URLFeatureExtractor, HTMLFeatureExtractor, NLPFeatureExtractor
from src.models import RandomForestModel, XGBoostModel, LSTMModel, BERTModel, EnsembleStacking


class PhishingDetectionPipeline:
    def __init__(self):
        self.url_extractor = URLFeatureExtractor()
        self.html_extractor = HTMLFeatureExtractor()
        self.nlp_extractor = NLPFeatureExtractor()
        
        self.rf_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.bert_model = None
        self.ensemble = None
        
        self.is_trained = False
        self.nlp_initialized = False
    
    def extract_features(self, url, html_content=None):
        url_features = self.url_extractor.extract(url)
        
        if html_content:
            html_features = self.html_extractor.extract(html_content, url)
            nlp_features = self.nlp_extractor.extract_all(html_content)
            
            nlp_tfidf = nlp_features.get('tfidf', np.zeros(50))
            nlp_word2vec = nlp_features.get('word2vec', np.zeros(100))
            nlp_bert = nlp_features.get('bert', np.zeros(768))
            urgency = nlp_features.get('urgency_score', 0)
            
            all_features = {**url_features, **html_features}
            all_features['urgency_score'] = urgency
            
            for i, val in enumerate(nlp_tfidf):
                all_features[f'tfidf_{i}'] = val
            for i, val in enumerate(nlp_word2vec):
                all_features[f'word2vec_{i}'] = val
            for i, val in enumerate(nlp_bert):
                all_features[f'bert_{i}'] = val
        else:
            all_features = url_features
            for i in range(50):
                all_features[f'tfidf_{i}'] = 0
            for i in range(100):
                all_features[f'word2vec_{i}'] = 0
            for i in range(768):
                all_features[f'bert_{i}'] = 0
            all_features['urgency_score'] = 0
        
        return all_features
    
    def _prepare_features(self, urls, html_contents=None):
        if html_contents is None:
            html_contents = [None] * len(urls)
        
        features_list = []
        for url, html in zip(urls, html_contents):
            features = self.extract_features(url, html)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        return df.values
    
    def _initialize_nlp(self):
        if not self.nlp_initialized:
            dummy_texts = ['sample text for initialization']
            self.nlp_extractor.initialize_tfidf(dummy_texts)
            self.nlp_extractor.initialize_word2vec()
            self.nlp_extractor.load_bert()
            self.nlp_initialized = True
    
    def train(self, urls, labels, html_contents=None):
        print("Initializing NLP models...")
        self._initialize_nlp()
        
        print("Extracting features...")
        X = self._prepare_features(urls, html_contents)
        y = np.array(labels)
        
        print(f"Training Random Forest...")
        self.rf_model = RandomForestModel(n_estimators=100)
        self.rf_model.fit(X, y)
        
        print(f"Training XGBoost...")
        self.xgb_model = XGBoostModel(learning_rate=0.1, max_depth=6)
        self.xgb_model.fit(X, y)
        
        print(f"Training LSTM...")
        self.lstm_model = LSTMModel()
        self.lstm_model.fit(urls, y, epochs=5, batch_size=32)
        
        print(f"Training BERT...")
        self.bert_model = BERTModel(epochs=3)
        self.bert_model.fit(urls, y)
        
        print(f"Training Ensemble Stacking...")
        base_models = {
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'lstm': self.lstm_model,
            'bert': self.bert_model
        }
        self.ensemble = EnsembleStacking(base_models)
        self.ensemble.fit(X, y)
        
        self.is_trained = True
        print("Training completed!")
        return self
    
    def predict(self, url, html_content=None):
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        features = self.extract_features(url, html_content)
        X = np.array([list(features.values())])
        
        prediction, confidence = self.ensemble.predict_with_confidence(X)
        
        individual_preds = self.ensemble.get_individual_predictions(X)
        
        return {
            'prediction': 'Phishing' if prediction[0] == 1 else 'Safe',
            'confidence': float(confidence[0]),
            'individual_predictions': {
                name: {
                    'probability': float(prob[0]),
                    'prediction': 'Phishing' if prob[0] > 0.5 else 'Safe'
                }
                for name, prob in individual_preds.items()
            }
        }
    
    def predict_batch(self, urls, html_contents=None):
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        results = []
        for url, html in zip(urls, html_contents if html_contents else [None] * len(urls)):
            results.append(self.predict(url, html))
        return results
    
    def save(self, path):
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Cannot save.")
        
        joblib.dump({
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'lstm_model': self.lstm_model,
            'bert_model': self.bert_model,
            'ensemble': self.ensemble,
            'nlp_initialized': self.nlp_initialized
        }, path)
    
    def load(self, path):
        data = joblib.load(path)
        self.rf_model = data['rf_model']
        self.xgb_model = data['xgb_model']
        self.lstm_model = data['lstm_model']
        self.bert_model = data['bert_model']
        self.ensemble = data['ensemble']
        self.nlp_initialized = data['nlp_initialized']
        
        base_models = {
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'lstm': self.lstm_model,
            'bert': self.bert_model
        }
        self.ensemble.base_models = base_models
        
        self.is_trained = True
        return self