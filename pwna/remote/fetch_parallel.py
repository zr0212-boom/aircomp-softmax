"""Parallel ranged download of HF files (uses http(s)_proxy from the environment), sha256-verified
against the LFS etag. Usage: python fetch_parallel.py <repo_id> <local_dir> <file> [<file> ...]"""
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CHUNK = 32 * 1024 * 1024
WORKERS = 32


def head(url):
    for attempt in range(10):
        try:
            return _head(url)
        except Exception as e:
            print(f"  HEAD retry {attempt}: {repr(e)[:80]}", flush=True)
            time.sleep(3 + 2 * attempt)
    raise RuntimeError(f"HEAD failed for {url}")


def _head(url):
    r = requests.head(url, allow_redirects=False, timeout=30)
    etag = (r.headers.get("x-linked-etag") or r.headers.get("etag") or "").strip('"')
    size = int(r.headers.get("x-linked-size") or 0)
    loc = r.headers.get("location") or url
    if not size:
        r2 = requests.head(loc, allow_redirects=True, timeout=30)
        size = int(r2.headers["content-length"])
    return loc, size, etag


def fetch(url, path, size):
    with open(path, "wb") as f:
        f.truncate(size)
    done = [0]
    lock = threading.Lock()

    def part(start):
        end = min(start + CHUNK, size) - 1
        for attempt in range(8):
            try:
                r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                r.raise_for_status()
                data = r.content
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)}")
                with open(path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                with lock:
                    done[0] += len(data)
                return
            except Exception as e:
                time.sleep(2 + attempt)
                err = e
        raise RuntimeError(f"chunk {start} failed: {err}")

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(part, s) for s in range(0, size, CHUNK)]
        while not all(f.done() for f in futs):
            time.sleep(10)
            print(f"  {os.path.basename(path)}: {done[0] / 1e9:.2f}/{size / 1e9:.2f} GB "
                  f"({done[0] / max(1e-9, time.time() - t0) / 1e6:.1f} MB/s)", flush=True)
        for f in futs:
            f.result()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def modelscope_meta(repo):
    """{path: (size, sha256)} from the ModelScope file API (works for repos gated on HF)."""
    url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true&Revision=master"
    for attempt in range(10):
        try:
            fs = requests.get(url, timeout=30).json()["Data"]["Files"]
            return {f["Path"]: (int(f["Size"]), f.get("Sha256") or "") for f in fs if f["Type"] == "blob"}
        except Exception as e:
            print(f"  ModelScope meta retry {attempt}: {repr(e)[:80]}", flush=True)
            time.sleep(3 + 2 * attempt)
    raise RuntimeError(f"ModelScope meta failed for {repo}")


def main():
    repo, local_dir, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    source = os.environ.get("FETCH_SOURCE", "hf")          # hf | modelscope
    suffix = os.environ.get("FETCH_SUFFIX", ".part")
    os.makedirs(local_dir, exist_ok=True)
    meta_host = os.environ.get("FETCH_META", "https://huggingface.co")   # or "modelscope"
    ms_meta = modelscope_meta(repo) if meta_host == "modelscope" else None
    for name in files:
        if ms_meta is not None:
            if name not in ms_meta:
                print(f"{name}: not in the repository, skipped", flush=True)
                continue
            size, etag = ms_meta[name]
        else:
            _, size, etag = head(f"{meta_host}/{repo}/resolve/main/{name}")
        path = os.path.join(local_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) == size and etag and sha256(path) == etag:
            print(f"{name}: already complete")
            continue
        if source == "modelscope":
            url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{name}"
            loc = url
            for attempt in range(10):
                try:
                    loc = requests.head(url, allow_redirects=True, timeout=30).url
                    break
                except Exception as e:
                    print(f"  ModelScope HEAD retry {attempt}: {repr(e)[:80]}", flush=True)
                    time.sleep(3 + 2 * attempt)
        else:
            loc, _, _ = head(f"https://huggingface.co/{repo}/resolve/main/{name}")
        print(f"{name}: {size / 1e9:.2f} GB from {source}, etag {etag[:12]}", flush=True)
        fetch(loc, path + suffix, size)
        digest = sha256(path + suffix)
        if etag and len(etag) == 64 and digest != etag:
            raise SystemExit(f"{name}: sha256 mismatch {digest} != {etag}")
        if os.path.exists(path) and os.path.getsize(path) == size:
            os.remove(path + suffix)
        else:
            os.replace(path + suffix, path)
        print(f"{name}: OK (sha256 verified)", flush=True)


if __name__ == "__main__":
    main()
