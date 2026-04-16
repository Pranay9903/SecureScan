import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.fc1(hidden))
        output = torch.sigmoid(self.fc2(hidden))
        return output


class LSTMModel:
    def __init__(self, max_length=200, vocab_size=128, embedding_dim=64, lstm_units=64):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        self.char_to_idx = {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _text_to_indices(self, texts):
        all_indices = []
        for text in texts:
            indices = []
            for char in text[:self.max_length]:
                if char not in self.char_to_idx:
                    if len(self.char_to_idx) < self.vocab_size - 1:
                        self.char_to_idx[char] = len(self.char_to_idx) + 1
                    else:
                        continue
                indices.append(self.char_to_idx.get(char, 0))
            
            while len(indices) < self.max_length:
                indices.append(0)
            
            all_indices.append(indices)
        
        return np.array(all_indices)
    
    def _build_model(self):
        self.model = LSTMClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.lstm_units,
            max_length=self.max_length
        )
        self.model.to(self.device)
    
    def fit(self, X_text, y, epochs=10, batch_size=32, validation_split=0.1):
        X_indices = self._text_to_indices(X_text)
        
        X_tensor = torch.LongTensor(X_indices)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        self._build_model()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X).squeeze()
                    val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_indices = self._text_to_indices(X_text)
        X_tensor = torch.LongTensor(X_indices).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).cpu().numpy()
        
        return predictions.astype(int)
    
    def predict_proba(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_indices = self._text_to_indices(X_text)
        X_tensor = torch.LongTensor(X_indices).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze().cpu().numpy()
        
        return outputs
    
    def save(self, path):
        torch.save(self.model.state_dict(), path.replace('.joblib', '_lstm.pt'))
        joblib.dump({
            'char_to_idx': self.char_to_idx,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'scaler': self.scaler
        }, path)
    
    def load(self, path):
        data = joblib.load(path)
        self.char_to_idx = data['char_to_idx']
        self.max_length = data['max_length']
        self.vocab_size = data['vocab_size']
        self.scaler = data['scaler']
        
        self._build_model()
        self.model.load_state_dict(torch.load(path.replace('.joblib', '_lstm.pt')))
        self.model.to(self.device)
        self.is_fitted = True
        return self