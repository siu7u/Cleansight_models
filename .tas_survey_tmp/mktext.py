import sys,re,html
p=sys.argv[1]
h=open(p,encoding='utf-8',errors='ignore').read()
h=re.sub(r'<script.*?</script>','',h,flags=re.S)
h=re.sub(r'<style.*?</style>','',h,flags=re.S)
h=re.sub(r'<(br|/p|/div|/h\d|/li|/tr)[^>]*>','\n',h)
h=re.sub(r'<[^>]+>',' ',h)
h=html.unescape(h)
h=re.sub(r'[ \t]+',' ',h)
h=re.sub(r'\n\s*\n+','\n',h)
print(h)
