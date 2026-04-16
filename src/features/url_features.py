import re
from urllib.parse import urlparse
import ipaddress


class URLFeatureExtractor:
    def __init__(self):
        self.urgency_keywords = [
            'login', 'verify', 'password', 'bank', 'account suspended',
            'urgent', 'confirm', 'security', 'update', 'action required',
            'unauthorized', 'suspended', 'restricted'
        ]
    
    def extract(self, url):
        features = {}
        
        parsed = urlparse(url)
        
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed.netloc)
        features['path_length'] = len(parsed.path)
        features['query_length'] = len(parsed.query)
        
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_equals'] = url.count('=')
        features['num_slash'] = url.count('/')
        features['num_question'] = url.count('?')
        features['num_colon'] = url.count(':')
        
        features['has_ip'] = self._is_ip_address(parsed.netloc)
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        
        features['num_subdomains'] = self._count_subdomains(parsed.netloc)
        features['num_params'] = parsed.query.count('&') + (1 if parsed.query else 0)
        
        features['path_depth'] = parsed.path.count('/')
        
        features['has_port'] = 1 if parsed.netloc.count(':') > 0 else 0
        
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['digit_ratio'] = features['num_digits'] / max(len(url), 1)
        
        features['has_https_in_path'] = 1 if 'https' in url.lower() else 0
        
        features['suspicious_tld'] = 1 if self._is_suspicious_tld(parsed.netloc) else 0
        
        domain = parsed.netloc
        features['domain_entropy'] = self._calculate_entropy(domain)
        
        features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
        
        url_lower = url.lower()
        features['urgency_score'] = sum(1 for kw in self.urgency_keywords if kw in url_lower)
        
        return features
    
    def _is_ip_address(self, hostname):
        try:
            ipaddress.ip_address(hostname)
            return 1
        except:
            return 0
    
    def _count_subdomains(self, netloc):
        parts = netloc.split('.')
        return max(0, len(parts) - 2)
    
    def _is_suspicious_tld(self, netloc):
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click', '.loan']
        return any(netloc.lower().endswith(tld) for tld in suspicious_tlds)
    
    def _calculate_entropy(self, text):
        if not text:
            return 0
        from collections import Counter
        import math
        counter = Counter(text)
        length = len(text)
        entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
        return entropy
    
    def extract_batch(self, urls):
        return [self.extract(url) for url in urls]