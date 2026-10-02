import re,sys,zlib
def streams(data):
    out=[]
    for m in re.finditer(rb'stream\r?\n',data):
        s=m.end()
        e=data.find(b'endstream',s)
        if e<0: continue
        out.append(data[s:e])
    return out
def decode(s):
    try: return zlib.decompress(s)
    except Exception:
        try: return zlib.decompressobj().decompress(s)
        except Exception: return None
def unescape(b):
    b=b.replace(b'\\(',b'(').replace(b'\\)',b')').replace(b'\\\\',b'\\')
    b=re.sub(rb'\\[0-7]{1,3}',lambda m:bytes([int(m.group(0)[1:],8)&0xff]),b)
    b=b.replace(b'\\n',b'\n').replace(b'\\r',b'').replace(b'\\t',b' ')
    return b
def extract(path):
    data=open(path,'rb').read()
    chunks=[]
    for raw in streams(data):
        d=decode(raw)
        if not d: continue
        if b'Tj' not in d and b'TJ' not in d: continue
        txt=[]
        # handle TJ arrays and Tj strings in order
        for m in re.finditer(rb'\[(.*?)\]\s*TJ|\((?:[^()\\]|\\.)*\)\s*Tj|T\*|TD|Td|ET',d,re.S):
            g=m.group(0)
            if g.endswith(b'TJ'):
                inner=m.group(1)
                parts=re.findall(rb'\((?:[^()\\]|\\.)*\)|-?\d+\.?\d*',inner)
                s=b''
                for p in parts:
                    if p.startswith(b'('):
                        s+=unescape(p[1:-1])
                    else:
                        try:
                            if float(p) < -120: s+=b' '
                        except: pass
                txt.append(s.decode('latin-1'))
            elif g.endswith(b'Tj'):
                txt.append(unescape(g[g.find(b'(')+1:g.rfind(b')')]).decode('latin-1'))
            else:
                txt.append('\n')
        t=''.join(txt)
        if t.strip(): chunks.append(t)
    return '\n'.join(chunks)
if __name__=='__main__':
    t=extract(sys.argv[1])
    t=re.sub(r'[ \t]+',' ',t)
    print(t)
