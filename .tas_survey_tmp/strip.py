import re, sys, html
p = sys.argv[1]
t = open(p, encoding='utf-8', errors='ignore').read()
t = re.sub(r'(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>', ' ', t)
t = re.sub(r'(?s)<[^>]+>', ' ', t)
t = html.unescape(t)
t = re.sub(r'[ \t\xa0]+', ' ', t)
t = re.sub(r'\n\s*\n+', '\n', t)
print(t)
