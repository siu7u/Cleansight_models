import sys,re,html
h=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
want=sys.argv[2] if len(sys.argv)>2 else None
# find figures/tables
for m in re.finditer(r'<figure[^>]*id="(S\d+[^"]*)"[^>]*>(.*?)</figure>',h,re.S):
    fid,body=m.group(1),m.group(2)
    cap=re.search(r'<figcaption.*?>(.*?)</figcaption>',body,re.S)
    captxt=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',cap.group(1)))) if cap else ''
    if want and want.lower() not in captxt.lower(): continue
    print('='*90); print('ID:',fid); print('CAP:',captxt[:400])
    for tm in re.finditer(r'<table.*?</table>',body,re.S):
        tb=tm.group(0)
        rows=re.findall(r'<tr.*?</tr>',tb,re.S)
        for r in rows:
            cells=re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',r,re.S)
            vals=[]
            for c in cells:
                v=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',c))).strip()
                vals.append(v)
            print(' | '.join(vals))
