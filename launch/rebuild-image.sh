#!/bin/bash
# Rebuild dsv4-a100:devel from this fork's patches against haosdent's
# CURRENT tip (12810046c) -- rebase-12810046c branch, NOT the c3046d1 base
# main still uses. Unlike c3046d1, 12810046c is normally git-reachable (no
# force-push has orphaned it -- yet), so this needs a plain clone, not the
# tarball-recovery dance in main's version of this script.
#
# Caches everything that doesn't need to change on a patch-only rebuild:
#   - the cloned base tree (skipped entirely if already present -- see
#     BASE_DIR below)
#   - the docker layers for apt/venv/pip-build-tools/rust/torch (see the
#     layering note at the top of docker/Dockerfile.fullbuild -- those RUN
#     instructions are unconditionally cached by podman as long as this file
#     doesn't change)
#
# Only the patch-apply-and-compile step reruns when patches/ changes.
#
# Run from anywhere; only needs $FORK_DIR checked out and podman installed.
set -euo pipefail

FORK_DIR="${FORK_DIR:-$HOME/cmp170hx-deepseek-v4-flash}"
WORK_DIR="${WORK_DIR:-$HOME/build-12810046c}"
BASE="12810046c799cbe874967e19b1c0fa134ab7b209"

BASE_DIR="$WORK_DIR/vllm-base"    # pristine tree handed to podman as build context
CTX_DIR="$WORK_DIR/ctx"           # build context: vllm-base/ + patches/ + Dockerfile
MARKER="$BASE_DIR/.base-sha"

mkdir -p "$WORK_DIR"

echo "== pulling latest patches from the fork (rebase-12810046c branch) =="
git -C "$FORK_DIR" pull --ff-only

if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$BASE" ]; then
  echo "== base tree already cloned, reusing $BASE_DIR =="
else
  echo "== cloning haosdent/vllm and checking out $BASE =="
  rm -rf "$BASE_DIR" "$WORK_DIR/clone-tmp"
  git init -q "$WORK_DIR/clone-tmp"
  git -C "$WORK_DIR/clone-tmp" remote add origin https://github.com/haosdent/vllm.git
  git -C "$WORK_DIR/clone-tmp" fetch --depth 1 origin "$BASE"
  git -C "$WORK_DIR/clone-tmp" checkout FETCH_HEAD

  echo "== copying tree to $BASE_DIR (no .git -- keeps podman COPY stable) =="
  mkdir -p "$BASE_DIR"
  rsync -a --delete --exclude='.git' "$WORK_DIR/clone-tmp/" "$BASE_DIR/"
  rm -rf "$WORK_DIR/clone-tmp"
  echo "$BASE" > "$MARKER"
fi

echo "== staging patches (skip 0001/0009 -- already in $BASE; 0006 dropped entirely, see docs/rebase-12810046c.md) =="
rm -rf "$CTX_DIR/patches"
mkdir -p "$CTX_DIR/patches"
for p in "$FORK_DIR"/patches/*.patch; do
  case "$p" in *0001-*|*0009-*) continue ;; esac
  cp "$p" "$CTX_DIR/patches/"
done
# rebase-12810046c-fixups/ isn't picked up by the glob above (subdirectory) --
# copy it in explicitly, named to sort after 0019 and before 0020 so it lands
# in the right place in the patch-apply loop's glob order. Only the one
# hunk of 0019 that 12810046c doesn't already have -- see the patch header.
cp "$FORK_DIR/patches/rebase-12810046c-fixups/0019-fixup-validate-tokens-ex.patch" \
  "$CTX_DIR/patches/0019z-fixup-validate-tokens-ex.patch"

echo "== syntax-checking patches against the base tree before spending build time =="
CHECK_DIR=$(mktemp -d)
rsync -a "$BASE_DIR/" "$CHECK_DIR/"
cd "$CHECK_DIR"
CHANGED_PY=""
for p in "$CTX_DIR"/patches/*.patch; do
  # NOTE: don't pipe `patch`'s own stdout through grep here -- a failed
  # hunk's diagnostic text only goes to stdout, and grep filtering for
  # "^patching file" silently discards it, so `set -e`/pipefail abort
  # the whole script with ZERO error output (bit us once already on
  # this branch). Capture full output, check patch's exit code
  # explicitly, and print everything if it failed.
  out=$(patch -p1 --forward < "$p" 2>&1) || {
    # Known, expected exception on this branch: 0019's backend_xgrammar.py
    # hunks are a package deal to `patch` (one redundant here, one not --
    # see docs/rebase-12810046c.md), so it reports the whole file "Reversed
    # or previously applied, skipping" and exits non-zero even though this
    # is fine -- the very next patch in the loop (0019z fixup) supplies the
    # one hunk that's actually missing. Anything else is a real failure.
    if [ "$(basename "$p")" = "0019-grammar-salvage-and-dspark-desync-guards.patch" ] \
      && printf '%s\n' "$out" | grep -q 'Reversed (or previously applied)' \
      && [ "$(printf '%s\n' "$out" | grep -c '^patching file')" -eq 4 ]; then
      echo "== 0019: expected backend_xgrammar.py partial-skip (fixup follows), continuing =="
      find . -name '*.rej' -delete
    else
      echo "FAILED: $(basename "$p")" >&2
      echo "$out" >&2
      exit 1
    fi
  }
  files=$(echo "$out" | grep '^patching file' | awk '{print $3}')
  CHANGED_PY="$CHANGED_PY $files"
done
for f in $CHANGED_PY; do
  case "$f" in *.py) python3 -m py_compile "$f" ;; esac
done
cd - > /dev/null
rm -rf "$CHECK_DIR"
echo "== patches apply clean and compile =="

# A symlink here doesn't work -- podman's copier refuses to dereference a
# symlink pointing outside the build context for COPY, even when the target
# exists and is readable. Real copy instead; vllm-base is source-only
# (~110M), so this costs seconds, and the COPY layer still cache-hits since
# podman keys on content, not on how it got into the context.
#
# rm the destination first and unconditionally: an earlier run of this
# script (before this fix) could have left CTX_DIR/vllm-base as a symlink
# back to BASE_DIR itself, and `rsync -a --delete SRC/ DEST/` where DEST
# resolves to SRC via that symlink syncs the tree into itself and DELETES
# it. A plain directory here is harmless to remove; a stale self-pointing
# symlink is exactly what must never reach the rsync below.
rm -rf "$CTX_DIR/vllm-base"
rsync -a "$BASE_DIR/" "$CTX_DIR/vllm-base/"
cp "$FORK_DIR/docker/Dockerfile.fullbuild" "$CTX_DIR/Dockerfile.fullbuild"
cp "$FORK_DIR/docker/dockerignore.txt" "$CTX_DIR/.dockerignore"

echo "== building image (only the patch layer + compile step redo on a patch-only change) =="
cd "$CTX_DIR"
podman build -f Dockerfile.fullbuild -t dsv4-a100:devel .

echo "== done: dsv4-a100:devel rebuilt =="
podman images dsv4-a100:devel
