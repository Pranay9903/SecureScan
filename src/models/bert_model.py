import numpy as np
import joblib
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class BERTModel:
    def __init__(self, model_name='bert-base-uncased', max_length=128, epochs=3, batch_size=16, learning_rate=2e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _get_bert_embeddings(self, texts):
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def fit(self, X_text, y):
        embeddings = self._get_bert_embeddings(X_text)
        
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        input_dim = embeddings_scaled.shape[1]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
        
        X_tensor = torch.FloatTensor(embeddings_scaled)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCELoss()
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.is_fitted = True
        return self
    
    def predict(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        embeddings = self._get_bert_embeddings(X_text)
        embeddings_scaled = self.scaler.transform(embeddings)
        
        X_tensor = torch.FloatTensor(embeddings_scaled)
        
        with torch.no_grad():
            outputs = self.classifier(X_tensor)
            predictions = (outputs.numpy() > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        embeddings = self._get_bert_embeddings(X_text)
        embeddings_scaled = self.scaler.transform(embeddings)
        
        X_tensor = torch.FloatTensor(embeddings_scaled)
        
        with torch.no_grad():
            outputs = self.classifier(X_tensor).numpy().flatten()
        
        return np.column_stack([1 - outputs, outputs])
    
    def save(self, path):
        torch.save(self.classifier.state_dict(), path.replace('.joblib', '_classifier.pt'))
        joblib.dump({
            'scaler': self.scaler,
            'max_length': self.max_length,
            'model_name': self.model_name
        }, path)
    
    def load(self, path):
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.max_length = data['max_length']
        self.model_name = data['model_name']
        
        input_dim = self.scaler.mean_.shape[0]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
        self.classifier.load_state_dict(torch.load(path.replace('.joblib', '_classifier.pt')))
        self.is_fitted = True
        return self