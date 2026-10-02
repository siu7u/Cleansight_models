#!/bin/bash
cd "$(dirname "$0")"
Q=(
"label noise temporal action segmentation"
"noisy labels temporal action segmentation"
"annotation errors action segmentation"
"annotator disagreement video action annotation"
"boundary ambiguity temporal action segmentation"
"annotation ambiguity dense video labeling"
"temporal action segmentation annotation quality"
"soft labels boundary smoothing action segmentation"
"action segmentation label errors robustness"
"inter-annotator agreement video dataset annotation"
"annotator variance fine-grained action recognition"
"dense video annotation throughput cost"
"unreliable annotations temporal action detection"
"label noise procedural activity recognition"
)
for q in "${Q[@]}"; do timeout 400 python3 oa_search.py "$q"; sleep 40; done
