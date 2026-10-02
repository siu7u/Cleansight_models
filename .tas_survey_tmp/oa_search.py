import sys,json,urllib.parse,subprocess,time
def fetch(url):
    for i in range(4):
        p=subprocess.run(['curl','-sL','--max-time','60','-A','research-bot/1.0',
                          '-w','\n@@%{http_code}',url],capture_output=True,text=True)
        out=p.stdout; code=out.rsplit('@@',1)[-1].strip(); body=out.rsplit('\n@@',1)[0]
        if code=='200':
            try: return json.loads(body)
            except: pass
        time.sleep(45)
    return {}
def search(q,per=25,field=None):
    f=f"title_and_abstract.search:{q}"
    if field: f+=f",primary_topic.field.id:{field}"
    u=f"https://api.openalex.org/works?filter={urllib.parse.quote(f,safe=':,')}&per-page={per}&mailto=research@example.org"
    d=fetch(u)
    return d.get('meta',{}).get('count'),d.get('results',[])
if __name__=='__main__':
    for q in sys.argv[1:]:
        n,res=search(q)
        print('#'*95); print(f'QUERY: {q}   (OpenAlex title+abstract hit count: {n})')
        for w in res:
            src=((w.get('primary_location') or {}).get('source') or {}).get('display_name')
            print(f"\n[{w.get('publication_year')}] {w.get('display_name')}\n   venue={src} | doi={w.get('doi')} | cited={w.get('cited_by_count')}")
            ab=w.get('abstract_inverted_index')
            if ab:
                pos={}
                for tok,idxs in ab.items():
                    for i in idxs: pos[i]=tok
                txt=' '.join(pos[k] for k in sorted(pos))
                print('   ABS:',txt[:900])
        time.sleep(2)
