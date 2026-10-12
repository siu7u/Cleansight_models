import sys,re,html,urllib.parse,subprocess,time
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
def get(url,tries=6,wait=20):
    r=''
    for i in range(tries):
        p=subprocess.run(['curl','-sL','--max-time','75','-A',UA,'-w','\n@@HTTP:%{http_code}',url],
                         capture_output=True,text=True)
        out=p.stdout
        code=out.rsplit('@@HTTP:',1)[-1].strip()
        r=out.rsplit('\n@@HTTP:',1)[0]
        if code=='200' and 'arxiv-result' in r: return r
        if code=='200' and 'query' in url: return r   # genuine 0 results page
        sys.stderr.write(f'  [retry {i+1} http={code}] {url[:70]}\n')
        time.sleep(wait)
    return r
def strip(s): return html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',s))).strip()
def search(q,stype='all',size=25):
    u=f"https://arxiv.org/search/?searchtype={stype}&query={urllib.parse.quote_plus(q)}&size={size}"
    t=get(u)
    blocks=re.split(r'arxiv-result',t)[1:]
    out=[]
    for b in blocks:
        idm=re.search(r'arxiv\.org/abs/([\d.]+)',b)
        tm=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',b,re.S)
        cm=re.search(r'<p class="is-size-7">(.*?)</p>',b,re.S)
        ab=re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</span>',b,re.S)
        out.append(dict(id=idm.group(1) if idm else '?',
                        title=strip(tm.group(1)) if tm else '?',
                        comments=strip(cm.group(1))[:200] if cm else '',
                        abs=strip(ab.group(1)).replace('△ Less','')[:1100] if ab else ''))
    return out
if __name__=='__main__':
    stype=sys.argv[1]
    for q in sys.argv[2:]:
        print('#'*100); print('QUERY:',q,'| searchtype:',stype)
        try: res=search(q,stype)
        except Exception as e: print('ERR',e); continue
        if not res: print('(none)')
        for r in res:
            print(f"\n[{r['id']}] {r['title']}\n  META: {r['comments']}\n  {r['abs'][:750]}")
        time.sleep(8)
