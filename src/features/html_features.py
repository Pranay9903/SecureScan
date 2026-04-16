from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


class HTMLFeatureExtractor:
    def __init__(self):
        self.urgency_keywords = [
            'login', 'verify', 'password', 'bank', 'account suspended',
            'urgent', 'confirm', 'security', 'update', 'action required',
            'unauthorized', 'suspended', 'restricted', 'immediately',
            'click here', 'submit', 'continue'
        ]
    
    def extract(self, html_content, base_url):
        features = {}
        
        if not html_content:
            return self._empty_features()
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        
        input_count = 0
        form_actions = []
        for form in forms:
            inputs = form.find_all('input')
            input_count += len(inputs)
            action = form.get('action', '')
            if action:
                form_actions.append(action)
        
        features['num_inputs'] = input_count
        features['num_form_actions'] = len(form_actions)
        
        hidden_iframes = soup.find_all('iframe', style=re.compile(r'display:\s*none|visibility:\s*hidden', re.I))
        features['num_hidden_iframes'] = len(hidden_iframes)
        
        all_iframes = soup.find_all('iframe')
        features['num_iframes'] = len(all_iframes)
        
        links = soup.find_all('a', href=True)
        internal_links = 0
        external_links = 0
        
        for link in links:
            href = link['href']
            if href.startswith('#') or href.startswith('javascript:'):
                continue
            
            try:
                full_url = urljoin(base_url, href)
                if self._is_internal_link(full_url, base_url):
                    internal_links += 1
                else:
                    external_links += 1
            except:
                external_links += 1
        
        total_links = internal_links + external_links
        features['internal_link_ratio'] = internal_links / max(total_links, 1)
        features['external_link_ratio'] = external_links / max(total_links, 1)
        
        forms_to_different_domain = 0
        parsed_base = self._extract_domain(base_url)
        
        for action in form_actions:
            try:
                full_action = urljoin(base_url, action)
                action_domain = self._extract_domain(full_action)
                if action_domain and action_domain != parsed_base:
                    forms_to_different_domain += 1
            except:
                pass
        
        features['forms_to_different_domain'] = forms_to_different_domain
        
        script_tags = soup.find_all('script')
        features['num_scripts'] = len(script_tags)
        
        meta_tags = soup.find_all('meta')
        features['num_meta_tags'] = len(meta_tags)
        
        title = soup.find('title')
        features['has_title'] = 1 if title and title.text.strip() else 0
        
        text_content = soup.get_text().lower()
        features['urgency_keyword_count'] = sum(1 for kw in self.urgency_keywords if kw in text_content)
        
        login_keywords = ['login', 'signin', 'password', 'username', 'email']
        features['has_login_form'] = 1 if any(kw in text_content for kw in login_keywords) else 0
        
        return features
    
    def _empty_features(self):
        return {
            'num_forms': 0, 'num_inputs': 0, 'num_hidden_iframes': 0,
            'num_iframes': 0, 'internal_link_ratio': 0, 'external_link_ratio': 0,
            'forms_to_different_domain': 0, 'num_scripts': 0, 'num_meta_tags': 0,
            'has_title': 0, 'urgency_keyword_count': 0, 'has_login_form': 0
        }
    
    def _is_internal_link(self, url, base_url):
        from urllib.parse import urlparse
        return urlparse(url).netloc == urlparse(base_url).netloc
    
    def _extract_domain(self, url):
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    def extract_batch(self, html_contents, base_urls):
        return [self.extract(html, url) for html, url in zip(html_contents, base_urls)]