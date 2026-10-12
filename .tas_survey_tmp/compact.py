import re,sys,html,paper,time
def meta(aid):
    t=paper.curl(f'https://arxiv.org/abs/{aid}')
    def grab(p):
        m=re.search(p,t,re.S)
        return html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>','',m.group(1)))).strip() if m else ''
    return dict(title=grab(r'<h1 class="title mathjax">(.*?)</h1>').replace('Title:','').strip(),
                comments=grab(r'<td class="tablecell comments[^"]*">(.*?)</td>'),
                jref=grab(r'<td class="tablecell jref">(.*?)</td>'),
                abs=grab(r'<blockquote class="abstract mathjax">(.*?)</blockquote>').replace('Abstract:','').strip())
for aid in sys.argv[1:]:
    try: m=meta(aid)
    except Exception as e: print(aid,'META ERR',e); continue
    print('#'*95); print(f'[{aid}] {m["title"]}')
    print('  VENUE-COMMENT:',m['comments'][:160],'| JREF:',m['jref'][:120])
    print('  ABS:',m['abs'][:900])
    t=paper.arxiv_text(aid)
    gh=sorted(set(re.findall(r'github\.com/[\w\-\.]+/[\w\-\.]+',t)))[:3]
    print('  CODE:',gh if gh else 'none in full text')
    s=re.sub(r'\s+',' ',t)
    for m2 in list(re.finditer(r'(Params?|params?)\s*[:\(]?\s*([\d\.]+\s*M|[\d\.]+\s*million)',s))[:2]:
        print('  PARAMS:',s[max(0,m2.start()-120):m2.start()+120])
    time.sleep(3)
