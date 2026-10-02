import re, sys, html
src = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
want = sys.argv[2] if len(sys.argv)>2 else None
tables = re.findall(r'(?is)<table.*?</table>', src)
for i,t in enumerate(tables):
    cap = re.search(r'(?is)<figcaption.*?</figcaption>', t)
    cap = re.sub(r'<[^>]+>',' ', cap.group(0)) if cap else ''
    cap = re.sub(r'\s+',' ', html.unescape(cap)).strip()
    if want and want.lower() not in cap.lower(): continue
    print(f"\n===== TABLE {i}: {cap[:200]} =====")
    for tr in re.findall(r'(?is)<tr.*?</tr>', t):
        cells=[]
        for c in re.findall(r'(?is)<t[hd][^>]*>(.*?)</t[hd]>', tr):
            c = re.sub(r'<[^>]+>',' ', c)
            c = re.sub(r'\s+',' ', html.unescape(c)).strip()
            cells.append(c)
        if cells: print(' | '.join(cells))
