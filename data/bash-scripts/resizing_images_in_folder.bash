#!/bin/bash
# To use convert command, install imagemagick

for f in "$1"/*.{jpg,png}; do
	[ -f "$f" ] || continue
	base=$(basename "$f")
	convert -resize 300x170 "$f" "$base"
done
