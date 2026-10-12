import sys,re,html,urllib.parse,subprocess
def get(url):
    return subprocess.run(['curl','-sL','--max-time','45','-A','Mozilla/5.0 (X11; Linux x86_64)',url],
                          capture_output=True,text=True).stdout
def strip(s): return html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',s))).strip()
def search(q,size=25):
    u=f"https://arxiv.org/search/?searchtype=all&query={urllib.parse.quote_plus(q)}&size=25"
    t=get(u)
    blocks=re.split(r'arxiv-result',t)[1:]
    out=[]
    for b in blocks:
        idm=re.search(r'arxiv\.org/abs/([\d.]+)',b)
        tm=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',b,re.S)
        am=re.search(r'<p class="authors">(.*?)</p>',b,re.S)
        cm=re.search(r'<p class="is-size-7">(.*?)</p>',b,re.S)
        ab=re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</span>',b,re.S)
        out.append(dict(id=idm.group(1) if idm else '?',
                        title=strip(tm.group(1)) if tm else '?',
                        authors=strip(am.group(1)).replace('Authors:','')[:150] if am else '?',
                        comments=strip(cm.group(1))[:150] if cm else '',
                        abs=strip(ab.group(1)).replace('△ Less','')[:900] if ab else ''))
    return out
if __name__=='__main__':
    for q in sys.argv[1:]:
        print('#'*100); print('QUERY:',q)
        try: res=search(q)
        except Exception as e: print('ERR',e); continue
        if not res: print('(none)')
        for r in res:
            print(f"\n[{r['id']}] {r['title']}\n  {r['authors']}\n  META: {r['comments']}\n  {r['abs'][:620]}")
