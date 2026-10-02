#!/bin/bash
python3 - "$@" <<'PY'
import sys, re, html, urllib.parse, subprocess, time
def get(url):
    return subprocess.run(['curl','-sL','--max-time','45','-A','Mozilla/5.0 (X11; Linux x86_64)',url],capture_output=True,text=True).stdout
def strip(s): return html.unescape(re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',s))).strip()
def search(q):
    for attempt in range(3):
        t=get(f"https://arxiv.org/search/?searchtype=all&query={urllib.parse.quote_plus(q)}&size=25")
        if 'arxiv-result' in t: break
        time.sleep(4)
    blocks=re.split(r'arxiv-result',t)[1:]
    out=[]
    for b in blocks:
        idm=re.search(r'/abs/([\d.]+)',b)
        tm=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',b,re.S)
        cm=re.search(r'<p class="is-size-7">(.*?)</p>',b,re.S)
        ab=re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</span>',b,re.S)
        if not idm: continue
        out.append((idm.group(1), strip(tm.group(1)) if tm else '?', strip(cm.group(1))[:110] if cm else '', strip(ab.group(1)).replace('△ Less','')[:420]))
    return out
for q in sys.argv[1:]:
    print('='*95); print('Q:',q)
    try: r=search(q)
    except Exception as e: print('ERR',e); continue
    if not r: print(' (none)')
    for i,(a,b,c,d) in enumerate(r[:10]):
        print(f"[{a}] {b}\n    META: {c}\n    {d}\n")
    time.sleep(2)
PY
