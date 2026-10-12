import sys,re,html,urllib.parse,subprocess,time
def get(url):
    return subprocess.run(['curl','-sL','--max-time','60','-A','Mozilla/5.0 (X11; Linux x86_64)',url],capture_output=True,text=True).stdout
def strip(s): return html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',s))).strip()
def search(q,size=200,order='-announced_date_first',stype='all'):
    u=(f"https://arxiv.org/search/?searchtype={stype}&query={urllib.parse.quote_plus(q)}"
       f"&size={size}&order={order}")
    for _ in range(4):
        t=get(u)
        if 'arxiv-result' in t or 'Sorry, your query returned no results' in t: break
        time.sleep(5)
    blocks=re.split(r'arxiv-result',t)[1:]
    out=[]
    for b in blocks:
        idm=re.search(r'/abs/([\d.]+)',b)
        tm=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',b,re.S)
        dm=re.search(r'<p class="is-size-7">(.*?)(?:</p>|Submitted)',b,re.S)
        cm=re.search(r'<p class="is-size-7">(.*?)</p>',b,re.S)
        ab=re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</span>',b,re.S)
        if not idm: continue
        out.append((idm.group(1), strip(tm.group(1)) if tm else '?', strip(cm.group(1))[:90] if cm else '',
                    strip(ab.group(1)).replace('△ Less','')[:300]))
    return out
if __name__=='__main__':
    q=sys.argv[1] if len(sys.argv)>1 else 'temporal action segmentation'
    n=int(sys.argv[2]) if len(sys.argv)>2 else 200
    r=search(q,n)
    print(f'# {len(r)} results for: {q}')
    for a,b,c,d in r:
        print(f"[{a}] {b}\n    {c}")
