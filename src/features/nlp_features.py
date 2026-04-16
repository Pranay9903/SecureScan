import numpy as np
import re
import torch
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class NLPFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.svd = None
        self.word_vectors = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.is_bert_loaded = False
        
        self.urgency_keywords = [
            'login', 'verify', 'password', 'bank', 'account suspended',
            'urgent', 'confirm', 'security', 'update', 'action required',
            'unauthorized', 'suspended', 'restricted', 'immediately',
            'click here', 'submit', 'continue', 'account', 'credit',
            'debit', 'invoice', 'payment', 'gift', 'winner', 'prize'
        ]
    
    def initialize_tfidf(self, texts, n_components=50):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        
        return self
    
    def extract_tfidf(self, text):
        if self.tfidf_vectorizer is None or self.svd is None:
            return np.zeros(50)
        
        try:
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            reduced = self.svd.transform(tfidf_vector)
            return reduced[0]
        except:
            return np.zeros(50)
    
    def initialize_word2vec(self):
        try:
            import gensim.downloader as api
            self.word_vectors = api.load('glove-wiki-gigaword-100')
            return True
        except Exception as e:
            print(f"Warning: Could not load Word2Vec/GloVe: {e}")
            return False
    
    def extract_word2vec(self, text):
        if self.word_vectors is None:
            return np.zeros(100)
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            try:
                vectors.append(self.word_vectors[word])
            except:
                continue
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(100)
    
    def load_bert(self):
        if self.is_bert_loaded:
            return
        
        try:
            from transformers import BertModel, BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            self.is_bert_loaded = True
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
    
    def extract_bert(self, text, max_length=512):
        if not self.is_bert_loaded:
            self.load_bert()
        
        if self.bert_model is None or self.bert_tokenizer is None:
            return np.zeros(768)
        
        try:
            inputs = self.bert_tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            return embedding
        except Exception as e:
            print(f"BERT extraction error: {e}")
            return np.zeros(768)
    
    def extract_urgency_keywords(self, html_content):
        if not html_content:
            return {'urgency_score': 0, 'detected_keywords': []}
        
        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text().lower()
        
        detected = []
        for keyword in self.urgency_keywords:
            if keyword in text:
                detected.append(keyword)
        
        return {
            'urgency_score': len(detected),
            'detected_keywords': detected
        }
    
    def extract_all(self, html_content):
        if not html_content:
            return {
                'tfidf': np.zeros(50),
                'word2vec': np.zeros(100),
                'bert': np.zeros(768),
                'urgency_score': 0
            }
        
        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text()
        cleaned_text = self._clean_text(text)
        
        features = {
            'tfidf': self.extract_tfidf(cleaned_text),
            'word2vec': self.extract_word2vec(cleaned_text),
            'bert': self.extract_bert(cleaned_text[:1000]),
            'urgency_score': self.extract_urgency_keywords(html_content)['urgency_score']
        }
        
        return features
    
    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()[:5000]
    
    def extract_batch(self, html_contents):
        return [self.extract_all(html) for html in html_contents]