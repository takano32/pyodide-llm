#!/usr/bin/env python3
"""python_archive.py (T367.2): the site's Python as two archives, which the worker fetches whole and unpacks in Pyodide
(public/worker/pyodide.js): one request where there were fifteen, and nineteen for a conversion.

    python3 python_archive.py [<the folder of the sources> [<the folder to write into>]]     (src/python, public)

    engine.zip      llama2_numpy.py and every .py of engine/       what every visit imports
    converter.zip   llama2_convert.py and every .py of convert/    fetched when a model is first converted

What goes in is found by walking: a module's window and every .py under its package's folder. No list is kept (the
worker had one, public/python.js, and a test to hold it to the folder: a part that was not listed was not fetched).
tests/python-archive-check.mjs holds the archives to what the tools walk (tests/tree.mjs) and to every .py there is.

The same sources give the same bytes, on any day and whatever the files' dates and modes are: the names in order, every
date 1980-01-01, every mode 644, deflate at level 9. So an archive's bytes change when the Python does and not
otherwise (another version of zlib may deflate to other bytes: a deployment is built in one place).

No binary is committed: the archives are built before every build and every `npm run dev` (package.json), and by
`make python`. Only the standard library is used."""
import io
import sys
import zipfile
from pathlib import Path

# an archive's name -> its module's window and its package's folder
ARCHIVES = {"engine.zip": ("llama2_numpy.py", "engine"), "converter.zip": ("llama2_convert.py", "convert")}
DATE = (1980, 1, 1, 0, 0, 0)  # the first date a zip can say


def names(sources, window, package):
    """The files of one archive, as their paths from the sources' folder with "/" between, in order."""
    sources = Path(sources)
    parts = [path.relative_to(sources).as_posix() for path in (sources / package).rglob("*.py") if "__pycache__" not in path.parts and path.is_file()]
    return sorted([window, *parts])


def archive(sources, window, package):
    """One archive's bytes."""
    held = io.BytesIO()
    with zipfile.ZipFile(held, "w") as file:
        for name in names(sources, window, package):
            entry = zipfile.ZipInfo(name, date_time=DATE)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3  # (Unix, whatever machine builds: the field is in the bytes)
            entry.external_attr = 0o100644 << 16
            file.writestr(entry, (Path(sources) / name).read_bytes(), compresslevel=9)
    return held.getvalue()


def main(sources="src/python", out="public"):
    here = Path(__file__).resolve().parent
    sources, out = (here / sources).resolve(), (here / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, (window, package) in ARCHIVES.items():
        made = archive(sources, window, package)
        (out / name).write_bytes(made)
        print(f"{name}: {len(names(sources, window, package))} files, {len(made):,} bytes")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
