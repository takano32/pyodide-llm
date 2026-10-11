# tests/fetching.py (T357)
# Fetching for the tools that take a model's files from huggingface.co in CI, in one place: a whole file to disk and a
# range of one. Both hold what came to the length that was promised: http.client's read() returns what came when the
# connection closes early, without a word, and a 4.5 GB GGUF that stopped short was "The file ended before all of its
# tensors were read." a minute into its conversion, a failure that named the converter (the review of T247; AGENTS.md).
# tests/fixed_outputs.py had the check; the reference tools' copies of the same loop (reference_llama.py,
# reference_qwen35.py, and reference_lfm2.py through it) did not.
# T374.4: and the size of a file, for whoever reads a file by its ranges and must know where it ends (ranged() holds
# an answer to the length asked for: a range past the end of the file is never right). The tools reach all three
# through tests/conducting.py's answerers, or call them as they are.
import http.client
import time
import urllib.error
import urllib.request

CHUNK = 8 << 20


def download(url, target, tries=3, timeout=60, optional=False):
    """The file at url as target (a Path), fetched unless it is there, through target + ".part" (a file that is there
    is a whole one). A download that stopped short of its Content-Length, a connection that broke and a server's own
    failure (5xx) are asked for again, tries times in all; what the server refuses (4xx: a file the repository does
    not have) is not, and raises urllib.error.HTTPError, or returns None where the file is optional."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(tries):  # huggingface.co drops a connection now and then
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as out:
                expected, written = response.headers.get("Content-Length"), 0
                while block := response.read(CHUNK):
                    out.write(block)
                    written += len(block)
            if expected is not None and written != int(expected):
                raise OSError(f"{url}: {written:,} of {int(expected):,} bytes came")
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 and optional:
                return None
            if error.code < 500 or attempt == tries - 1:
                raise  # (T192: a split model's 404 is how its index is found)
        except (OSError, http.client.HTTPException):  # (a chunked body that ends early: IncompleteRead is no OSError)
            if attempt == tries - 1:
                raise
    partial.rename(target)
    return target


def ranged(url, start, length, tries=4, timeout=120, wait=0, said=None):
    """length bytes of the file at url from start, by a Range request (the redirect to the CDN is followed): all of
    them, or it is asked again (tries times in all, wait seconds more before each next one; said(error): told of each
    try that failed but the last, which raises its OSError)."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if len(data) != length:
                raise OSError(f"{len(data)} bytes of {length}")
            return data
        except (OSError, http.client.HTTPException) as error:  # (URLError and HTTPError are OSErrors; read() of a body that
            # closed before its Content-Length raises IncompleteRead, which is not: T357's review)
            if attempt == tries - 1:
                raise
            if said:
                said(error)
            time.sleep(wait * (attempt + 1))


def sized(url, tries=4, timeout=120):
    """The size of the file at url, or None where there is no such file (a 404): by a Range request for its first byte
    (the redirect to the CDN is followed), whose answer says of how many it is one. Asked again as ranged() asks."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                said = response.headers.get("Content-Range") or ""  # "bytes 0-0/1234"
                if said.rpartition("/")[2].isdigit():
                    return int(said.rpartition("/")[2])
                # (a server that sends the whole file instead says its length)
                length = response.headers.get("Content-Length")
                if getattr(response, "status", 200) == 200 and length is not None:
                    return int(length)
                raise OSError(f"{url}: the answer does not say the size of the file")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code < 500 or attempt == tries - 1:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
