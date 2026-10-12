#!/bin/bash
cd "$(dirname "$0")"
run(){ echo "===== TYPE=$1 Q=$2"; timeout 300 python3 ln_search.py "$1" "$2" 2>/dev/null; sleep 10; }
run title "label noise action segmentation"
run all "annotator variance video annotation"
run all "boundary ambiguity action segmentation"
run title "noisy temporal action segmentation"
run all "annotation errors temporal action segmentation"
run all "dense video annotation throughput"
run title "temporal action segmentation annotation quality"
