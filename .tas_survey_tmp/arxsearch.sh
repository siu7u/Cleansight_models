#!/bin/bash
# $1 = searchtype, $2 = query (already url-encoded or with + ), $3 = outfile
for i in 1 2 3; do
  curl -sL "https://arxiv.org/search/?searchtype=$1&query=$2&size=25" -o "$3"
  n=$(grep -c arxiv-result "$3" 2>/dev/null)
  if [ "$n" -gt 0 ]; then echo "OK $n results in $3"; exit 0; fi
  sleep 3
done
echo "FAIL $3"; exit 1
