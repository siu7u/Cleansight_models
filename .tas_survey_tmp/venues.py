import json,re,subprocess,urllib.parse,sys,time
def cr(title):
    u='https://api.crossref.org/works?rows=2&mailto=lit@example.com&query.bibliographic='+urllib.parse.quote(title)
    r=subprocess.run(['curl','-s','--max-time','30',u],capture_output=True,text=True).stdout
    try: items=json.loads(r)['message']['items']
    except Exception: return None
    return items
titles=[
 "ASFormer: Transformer for Action Segmentation",
 "Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation",
 "C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation",
 "Diffusion Action Segmentation",
 "How Much Temporal Long-Term Context is Needed for Action Segmentation",
 "Efficient Temporal Action Segmentation via Boundary-aware Query Voting",
 "FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation",
 "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation",
 "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation",
 "Decoupling Individual Identification and Temporal Reasoning for Action Segmentation",
 "Temporal Segment Transformer for Action Segmentation",
 "BIT: Bi-Level Temporal Modeling for Efficient Supervised Action Segmentation",
 "Activity Grammars for Temporal Action Segmentation",
 "Do we really need temporal convolutions in action segmentation",
 "Temporal Action Segmentation: An Analysis of Modern Techniques",
 "D3TW: Discriminative Differentiable Dynamic Time Warping for Weakly Supervised Action Alignment and Segmentation",
 "SMC-NCA: Semantic-guided Multi-level Contrast for Semi-supervised Temporal Action Segmentation",
 "Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment",
 "CLIP: Cheap Lipschitz Training for Neural Networks",
]
for t in titles:
    it=cr(t)
    print('T:',t[:70])
    if not it: print('   (no crossref)'); continue
    for x in it[:2]:
        y=(x.get('published') or x.get('issued') or {}).get('date-parts',[[None]])[0][0]
        print('   >',(x.get('title') or ['?'])[0][:75],'||',(x.get('container-title') or ['?'])[0][:60],'||',y,'||',x.get('DOI'))
    time.sleep(0.5)
