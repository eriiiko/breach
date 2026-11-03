#!/bin/sh
# Download the nine raw Krea characters (Meshy V7, 2026-10-23..25) listed in
# SOURCES.tsv from Krea's public upload store. Krea's own Download button
# fails on these; the public URL does not. Raw files stay local (not in git).
cd "$(dirname "$0")"
while IFS="$(printf '\t')" read -r name uuid; do
    [ -f "$name" ] || curl -fsSL -o "$name" "https://app-uploads.krea.ai/public/$uuid.glb"
    echo "$name"
done < SOURCES.tsv
