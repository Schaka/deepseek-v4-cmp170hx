#!/bin/bash
# Rebuild dsv4-a100:devel from this fork's patches against the c3046d1 base.
#
# Caches everything that doesn't need to change on a patch-only rebuild:
#   - the reconstructed c3046d1 base tree (skipped entirely if already present
#     and verified -- see BASE_DIR below)
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
WORK_DIR="${WORK_DIR:-$HOME/build}"
BASE="c3046d1ebd2dae9b94ad2ef5f966ea153632251e"
UPSTREAM_PARENT="f8ea5bb163c161ef38b401d055cc5fd4a934091a"
EXPECT_TREE="d13ae12b9a6621ef8d218f53741e59c6db2f68d2"

RECON_DIR="$WORK_DIR/recon"       # scratch git repo, only used to verify the tree
BASE_DIR="$WORK_DIR/vllm-base"    # pristine tree handed to podman as build context
CTX_DIR="$WORK_DIR/ctx"           # build context: vllm-base/ + patches/ + Dockerfile
MARKER="$BASE_DIR/.recon-tree-sha"

mkdir -p "$WORK_DIR"

echo "== pulling latest patches from the fork =="
git -C "$FORK_DIR" pull --ff-only

if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$EXPECT_TREE" ]; then
  echo "== base tree already reconstructed and verified, reusing $BASE_DIR =="
else
  echo "== reconstructing $BASE (unreachable by any git method -- tarball recovery) =="
  rm -rf "$RECON_DIR" "$BASE_DIR"
  git clone https://github.com/haosdent/vllm.git "$RECON_DIR"
  cd "$RECON_DIR"

  curl -sL -o /tmp/c3046d1.tar.gz \
    "https://codeload.github.com/haosdent/vllm/tar.gz/$BASE"
  rm -rf /tmp/c3046d1-src
  mkdir -p /tmp/c3046d1-src
  tar xzf /tmp/c3046d1.tar.gz -C /tmp/c3046d1-src --strip-components=1

  export GIT_INDEX_FILE=/tmp/c3046d1.index
  git read-tree --empty
  git --work-tree=/tmp/c3046d1-src add -Af
  TREE=$(git write-tree)
  if [ "$TREE" != "$EXPECT_TREE" ]; then
    echo "FATAL: reconstructed tree $TREE != expected $EXPECT_TREE" >&2
    exit 1
  fi
  export GIT_AUTHOR_NAME=build GIT_AUTHOR_EMAIL=build@local
  export GIT_COMMITTER_NAME=build GIT_COMMITTER_EMAIL=build@local
  git tag -f c3046d1-recon "$(git commit-tree "$TREE" -p "$UPSTREAM_PARENT" -m 'c3046d1 reconstructed from tarball')"
  unset GIT_INDEX_FILE
  git checkout -B rebase-c3046d1 c3046d1-recon

  echo "== copying verified pristine tree to $BASE_DIR (no .git -- keeps podman COPY stable) =="
  mkdir -p "$BASE_DIR"
  rsync -a --delete --exclude='.git' "$RECON_DIR/" "$BASE_DIR/"
  echo "$EXPECT_TREE" > "$MARKER"
fi

echo "== staging patches (skip 0001 -- upstreamed on this base) =="
rm -rf "$CTX_DIR/patches"
mkdir -p "$CTX_DIR/patches"
for p in "$FORK_DIR"/patches/*.patch; do
  case "$p" in *0001-*) continue ;; esac
  cp "$p" "$CTX_DIR/patches/"
done

echo "== syntax-checking patches against the base tree before spending build time =="
CHECK_DIR=$(mktemp -d)
rsync -a "$BASE_DIR/" "$CHECK_DIR/"
cd "$CHECK_DIR"
CHANGED_PY=""
for p in "$CTX_DIR"/patches/*.patch; do
  files=$(patch -p1 --forward < "$p" | grep '^patching file' | awk '{print $3}')
  CHANGED_PY="$CHANGED_PY $files"
done
for f in $CHANGED_PY; do
  case "$f" in *.py) python3 -m py_compile "$f" ;; esac
done
cd - > /dev/null
rm -rf "$CHECK_DIR"
echo "== patches apply clean and compile =="

ln -sfn "$BASE_DIR" "$CTX_DIR/vllm-base"
cp "$FORK_DIR/docker/Dockerfile.fullbuild" "$CTX_DIR/Dockerfile.fullbuild"
cp "$FORK_DIR/docker/dockerignore.txt" "$CTX_DIR/.dockerignore"

echo "== building image (only the patch layer + compile step redo on a patch-only change) =="
cd "$CTX_DIR"
podman build -f Dockerfile.fullbuild -t dsv4-a100:devel .

echo "== done: dsv4-a100:devel rebuilt =="
podman images dsv4-a100:devel
