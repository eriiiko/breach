#!/usr/bin/env bash
# Download every finished Krea prop in krea_urls.tsv that is not yet local and
# run prop_gameready.py on every one not yet game-ready.
#   process_batch.sh <jobs.tsv> <urls.tsv> <src_dir> <worktree>
# jobs.tsv: job_id name model size_flag   urls.tsv: name url
# A name may carry a front override in fronts.tsv beside urls.tsv (name axis).
set -u
JOBS=$1 URLS=$2 SRC=$3 WT=$4
B="/c/Program Files/Blender Foundation/Blender 4.5/blender.exe"
FRONTS="$(dirname "$URLS")/fronts.tsv"
while IFS=$'\t' read -r name url; do
  [ -z "$name" ] && continue
  [ -f "$SRC/$name.glb" ] || curl -s -o "$SRC/$name.glb" "$url"
  out="$WT/assets/models/props/krea/$name/$name.glb"
  front=$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$FRONTS" 2>/dev/null)
  [ -f "$out" ] && [ -z "${FORCE:-}" ] && continue
  size=$(awk -F'\t' -v n="$name" '$2==n{print $4}' "$JOBS")
  "$B" -b -P "$WT/art/props/scripts/prop_gameready.py" -- "$SRC/$name.glb" "$out" \
      "$WT/art/props/previews" $size --front "${front:--y}" 2>&1 | grep -E "SIZE|TRIS|Error|Traceback"
done < "$URLS"
