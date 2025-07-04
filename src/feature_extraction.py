import re
from urllib.parse import urlparse

URL_SHORTENERS = [
    'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'bit.do', 'shorte.st',
    'adf.ly', 'bitly.com', 'cutt.ly', 'is.gd', 'soo.gd', 's2r.co', 'lnkd.in'
]

BRAND_KEYWORDS = ['apple', 'paypal', 'google', 'amazon', 'microsoft', 'bank', 'facebook', 'instagram']

SUSPICIOUS_WORDS = ['login', 'secure', 'account', 'update', 'verify', 'signin', 'password']

PHISHY_TLDS = ['.info', '.xyz', '.tk', '.ml', '.ga', '.cf']

def has_ip_address(url):
    return 1 if re.match(r'http[s]?://(?:\d{1,3}\.){3}\d{1,3}', url) else 0

def count_special_chars(url, chars):
    return sum(url.count(c) for c in chars)

def uses_url_shortener(domain):
    return 1 if domain in URL_SHORTENERS else 0

def count_digits(url):
    return sum(c.isdigit() for c in url)

def has_suspicious_words(url):
    url_lower = url.lower()
    return int(any(word in url_lower for word in SUSPICIOUS_WORDS))

def has_brand_misuse(domain):
    for brand in BRAND_KEYWORDS:
        if brand in domain and not domain.endswith(f"{brand}.com"):
            return 1
    return 0

def uses_phishy_tld(domain):
    return int(any(domain.endswith(tld) for tld in PHISHY_TLDS))

def num_query_params(url):
    return url.count('?') + url.count('&')

def first_dir_length(path):
    parts = path.strip('/').split('/')
    return len(parts[0]) if parts else 0

def domain_in_path(domain, path):
    return 1 if domain in path else 0

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path

    features = {
        "url_length": len(url),
        "count_dots": url.count('.'),
        "count_hyphens": url.count('-'),
        "has_at_symbol": 1 if '@' in url else 0,
        "has_ip": has_ip_address(url),
        "has_https": 1 if url.lower().startswith('https') else 0,
        "special_char_count": count_special_chars(url, ['?', '%', '=', '&', '_']),
        "num_subdirs": path.count('/'),
        "domain_length": len(domain),
        "uses_shortener": uses_url_shortener(domain),
        "count_digits": count_digits(url),
        "has_suspicious_words": has_suspicious_words(url),
        "num_query_params": num_query_params(url),
        "first_dir_length": first_dir_length(path),
        "domain_in_path": domain_in_path(domain, path),

        # 🔥 New Features
        "has_brand_misuse": has_brand_misuse(domain),
        "uses_phishy_tld": uses_phishy_tld(domain)
    }

    return features
