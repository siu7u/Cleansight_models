import sys,re,html,subprocess,os,warnings,logging
warnings.filterwarnings('ignore')
logging.getLogger('pypdf').setLevel(logging.CRITICAL)
def curl(url,out=None):
    a=['curl','-sL','--max-time','60','-A','Mozilla/5.0 (X11; Linux x86_64)',url]
    if out:
        subprocess.run(a+['-o',out],stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL); return out
    r=subprocess.run(a,capture_output=True,text=True)
    return r.stdout
def strip(s):
    s=re.sub(r'<(script|style)[^>]*>.*?</\1>','',s,flags=re.S|re.I)
    s=re.sub(r'<[^>]+>',' ',s)
    return re.sub(r'[ \t]{2,}',' ',re.sub(r'\n{3,}','\n\n',html.unescape(s)))
def arxiv_text(aid,cache='cache'):
    os.makedirs(cache,exist_ok=True); f=f'{cache}/{aid}.txt'
    if os.path.exists(f) and os.path.getsize(f)>3000: return open(f).read()
    t=curl(f'https://arxiv.org/html/{aid}')
    if len(t)<5000 or 'No HTML' in t[:2000]: t=curl(f'https://ar5iv.labs.arxiv.org/html/{aid}')
    t=strip(t) if len(t)>3000 else strip(curl(f'https://arxiv.org/abs/{aid}'))
    open(f,'w').write(t); return t
def pdf_text(url,cache='cache',save=None):
    os.makedirs(cache,exist_ok=True)
    f=save or (f'{cache}/'+re.sub(r'\W+','_',url)[-70:]+'.pdf')
    curl(url,f)
    try:
        import pypdf
        r=pypdf.PdfReader(f)
        txt='\n'.join((p.extract_text() or '') for p in r.pages)
        if save: open(save.replace('.pdf','.txt'),'w').write(txt)
        return txt
    except Exception as e: return f'PDF_ERR {e}'
def show(t,kw,n=3,w=300,label=''):
    t=re.sub(r'\s+',' ',t); print(f'== {label}')
    for m in list(re.finditer(kw,t,re.I))[:n]:
        print(' -',t[max(0,m.start()-w):m.start()+w],'\n')
if __name__=='__main__':
    mode=sys.argv[1]; src=sys.argv[2]
    t=arxiv_text(src) if mode=='ax' else pdf_text(src)
    open('cache/out.txt','w').write(t); print(f'LEN={len(t)} -> cache/out.txt')
