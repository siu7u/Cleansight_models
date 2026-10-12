import re, sys, html
src = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
# find all figure blocks that contain a table, print caption text
for m in re.finditer(r'(?is)<figure[^>]*>(.*?)</figure>', src):
    blk = m.group(1)
    if '<table' not in blk: continue
    cap = re.search(r'(?is)<figcaption.*?</figcaption>', blk)
    if not cap: continue
    c = re.sub(r'<[^>]+>',' ', cap.group(0)); c = re.sub(r'\s+',' ', html.unescape(c)).strip()
    print(c[:160])
