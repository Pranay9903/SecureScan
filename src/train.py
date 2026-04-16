import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.features import URLFeatureExtractor, HTMLFeatureExtractor, NLPFeatureExtractor
from src.models import RandomForestModel, XGBoostModel, LSTMModel, BERTModel, EnsembleStacking
from src.collect_urls import URLCollector


class TrainingPipeline:
    def __init__(self):
        self.url_extractor = URLFeatureExtractor()
        self.html_extractor = HTMLFeatureExtractor()
        self.nlp_extractor = NLPFeatureExtractor()
        
        self.rf_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.bert_model = None
        self.ensemble = None
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        urls = df['url'].tolist()
        labels = df['label'].tolist()
        
        return urls, labels
    
    def extract_features_batch(self, urls, html_contents=None):
        print("Extracting features...")
        
        if html_contents is None:
            html_contents = [None] * len(urls)
        
        all_features = []
        
        for i, (url, html) in enumerate(tqdm(zip(urls, html_contents), total=len(urls))):
            url_features = self.url_extractor.extract(url)
            
            if html:
                html_features = self.html_extractor.extract(html, url)
                nlp_features = self.nlp_extractor.extract_all(html)
                
                combined = {**url_features, **html_features}
                combined['urgency_score'] = nlp_features.get('urgency_score', 0)
                
                tfidf = nlp_features.get('tfidf', np.zeros(50))
                word2vec = nlp_features.get('word2vec', np.zeros(100))
                bert = nlp_features.get('bert', np.zeros(768))
                
                for j, val in enumerate(tfidf):
                    combined[f'tfidf_{j}'] = val
                for j, val in enumerate(word2vec):
                    combined[f'word2vec_{j}'] = val
                for j, val in enumerate(bert):
                    combined[f'bert_{j}'] = val
            else:
                combined = url_features
                for j in range(50):
                    combined[f'tfidf_{j}'] = 0
                for j in range(100):
                    combined[f'word2vec_{j}'] = 0
                for j in range(768):
                    combined[f'bert_{j}'] = 0
                combined['urgency_score'] = 0
            
            all_features.append(combined)
        
        return pd.DataFrame(all_features)
    
    def train(self, X, y):
        print("Initializing NLP models...")
        dummy_texts = ['sample text for initialization']
        self.nlp_extractor.initialize_tfidf(dummy_texts)
        self.nlp_extractor.initialize_word2vec()
        
        print("Training Random Forest...")
        self.rf_model = RandomForestModel(n_estimators=100)
        self.rf_model.fit(X, y)
        
        print("Training XGBoost...")
        self.xgb_model = XGBoostModel(learning_rate=0.1, max_depth=6)
        self.xgb_model.fit(X, y)
        
        print("Training LSTM...")
        self.lstm_model = LSTMModel(max_length=200, vocab_size=128)
        
        lstm_urls = [f"url_{i}" for i in range(len(y))]
        self.lstm_model.fit(lstm_urls, y, epochs=3, batch_size=32)
        
        print("Training BERT (this may take a while)...")
        self.bert_model = BERTModel(epochs=1, batch_size=8)
        self.bert_model.fit(lstm_urls, y)
        
        print("Training Ensemble Stacking...")
        base_models = {
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'lstm': self.lstm_model,
            'bert': self.bert_model
        }
        self.ensemble = EnsembleStacking(base_models)
        self.ensemble.fit(X, y)
        
        print("Training completed!")
        return self
    
    def evaluate(self, X_test, y_test):
        print("\nEvaluating models...")
        
        rf_pred = self.rf_model.predict(X_test)
        xgb_pred = self.xgb_model.predict(X_test)
        ensemble_pred, ensemble_conf = self.ensemble.predict_with_confidence(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {
            'Random Forest': {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred),
                'recall': recall_score(y_test, rf_pred),
                'f1': f1_score(y_test, rf_pred)
            },
            'XGBoost': {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred),
                'recall': recall_score(y_test, xgb_pred),
                'f1': f1_score(y_test, xgb_pred)
            },
            'Ensemble': {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred),
                'recall': recall_score(y_test, ensemble_pred),
                'f1': f1_score(y_test, ensemble_pred)
            }
        }
        
        for model, metrics in results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_models(self, output_dir='models'):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Saving models...")
        
        self.rf_model.save(f'{output_dir}/rf_model.joblib')
        self.xgb_model.save(f'{output_dir}/xgb_model.joblib')
        self.lstm_model.save(f'{output_dir}/lstm_model.joblib')
        self.bert_model.save(f'{output_dir}/bert_model.joblib')
        self.ensemble.save(f'{output_dir}/ensemble_model.joblib')
        
        print(f"Models saved to {output_dir}/")


def main():
    from sklearn.model_selection import train_test_split
    
    data_path = 'data/raw/urls.csv'
    
    if not os.path.exists(data_path):
        print("Data file not found. Collecting URLs...")
        collector = URLCollector()
        collector.collect_all(legitimate_count=1000, phishing_count=1000)
    
    pipeline = TrainingPipeline()
    
    urls, labels = pipeline.load_data(data_path)
    
    print(f"Total samples: {len(urls)}")
    print(f"Positive (phishing): {sum(labels)}")
    print(f"Negative (legitimate): {len(labels) - sum(labels)}")
    
    X = pipeline.extract_features_batch(urls)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    pipeline.train(X_train.values, y_train)
    
    pipeline.evaluate(X_test.values, y_test)
    
    pipeline.save_models()


if __name__ == '__main__':
    main()