#!/bin/bash

for f in build/*test; do
  [[ -x "$f" && -f "$f" ]] || continue
  "$f"
done
