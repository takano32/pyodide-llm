# fetch_translations.py (T648.5)
# Mozilla's Firefox Translations models of English and Japanese, for the site: each model's file, its vocabularies and
# its shortlist, as Firefox fetches them, into <folder>/<model>/<the record's name>. The Makefile cuts the models and the
# shortlists into the site's parts of 8 MiB and copies the vocabularies (public/models/translations/; T648.4's page
# reads them there).
#
#   python3 fetch_translations.py <folder>          (make models: translations/)
#
# Firefox lists them in Remote Settings' collection translations-models-v2 (FIREFOX's records) and fetches each
# attachment from its CDN, which says no CORS, so a page cannot: the site serves them (T648). What is fetched is
# written down here, record by record, as a Hugging Face entry names its revision: the attachment's address, its
# length and sha256 as the collection gave them, and the length and sha256 of what zstd unpacks it to
# (decompressedSize, decompressedHash). The newest version is not looked up: a new one of Mozilla's comes in by a
# change to this table, and every build of one commit serves the same bytes. A file that is there and holds its length
# and sha256 is not fetched again; one that does not is an error (it is deleted, to be fetched by the next build).
#
# The three models (the owner, T648 (4)): en-ja 3.0, and ja-en in two versions: 3.0, the desktop's ("base", 6 + 2
# layers of 512), and 3.1, Android's ("base-memory", 6 + 4 of 384; filter_expression env.appinfo.OS == 'Android').
# ja-en's two vocabularies are one file (the same sha256), kept by each model as its own.
#
# The files are Mozilla's, under the MPL-2.0 (mozilla/translations' README): LICENSE.txt beside them says so, and holds
# the licence's text (this repository's LICENSE, which is the MPL-2.0's as it is).
#
# zstd is Python's own (compression.zstd, from 3.14): no tool outside it.
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
from fetching import download  # noqa: E402 (the tools' one way to fetch a file: asked again, its length held)

try:
    from compression import zstd
except ImportError:
    sys.exit(f"fetch_translations.py unpacks zstd with Python's compression.zstd, which Python 3.14 has: this is {sys.version.split()[0]}")

CDN = "https://firefox-settings-attachments.cdn.mozilla.net/"
FIREFOX = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models-v2/records"
# a model -> its files: (the record's name, the attachment's location, its length, its sha256, the unpacked length,
# its sha256). Read off FIREFOX on 2026-11-05.
MODELS = {
    "en-ja-3.0": [
        ("model.enja.intgemm.alphas.bin", "main-workspace/translations-models-v2/a01a5318-41c7-4266-8fe1-66c105d6d711.zst",
         32_431_590, "f92259b9904812350b71f440c3328f3e2e856f31c1a3ab6002756380dd908e00",
         43_849_787, "59ae659f9bb63e4f81f474fe3c03d3f4499434b5f9e779fab7c12a45f31fd562"),
        ("srcvocab.enja.spm", "main-workspace/translations-models-v2/d10e4482-b9ba-4565-ab44-5ba3cebcf94b.zst",
         345_747, "dd1e93d625d9a8d8a42f75255343f302ea37527933eda1a490652b66a7408a64",
         796_275, "970c98d174fc01e0339fbabbf45af36a4be3f26f819ec1a5ea1189f71e091889"),
        ("trgvocab.enja.spm", "main-workspace/translations-models-v2/e54d177c-1131-4fbf-90ea-6741a98d6675.zst",
         361_180, "5384abf34f0e1298f795bd14909a577272059041e60f91a5eb10500e35d0ee98",
         827_144, "3b3d9f8f3a034d98d0a476f1794fa79c01e4e98a967ceb6777a66ba2d03ec1e1"),
        ("lex.50.50.enja.s2t.bin", "main-workspace/translations-models-v2/83e22dc9-40c4-4ac1-8c25-5d3877b5f628.zst",
         1_751_977, "8954d5b63432366992b22ad2aaa11895e2643014835049fe677655540a705915",
         4_128_360, "edfb7eb47b98a2689b804ab3614c31b24f1257aa54b1154da3d85e1ae8152d9f"),
    ],
    "ja-en-3.0": [
        ("model.jaen.intgemm.alphas.bin", "main-workspace/translations-models-v2/53a52e41-5abb-4705-9ba8-8dbc07614bdc.zst",
         41_929_252, "c0c9f2e48979f022e16271049d96dbef9aa28f8fddc2d1db8fa0c399dec903f7",
         59_504_955, "a9bf800679bba570520e1161d7b4fbfcb957add32ca35812134add85689752ad"),
        ("vocab.jaen.spm", "main-workspace/translations-models-v2/2566c4c6-4b9f-4d94-8898-38dbda231813.zst",
         617_539, "f41f32e36f879643a47fedbd0616fa12f416da62e23af8620ee4fc7705217acc",
         1_443_222, "5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919"),
        ("lex.50.50.jaen.s2t.bin", "main-workspace/translations-models-v2/3fe6eec6-c9fa-4262-85d1-a773ece70b6c.zst",
         3_358_329, "e6715f3e8a9d2ff08004b8185305de73906d7c31352f99a7f0424b4bf96a9422",
         9_346_816, "8f858a72fcbaa476c582577b04d6f5f89d645d2335b0b4a794c2706d4b1f75ff"),
    ],
    "ja-en-3.1": [
        ("model.jaen.intgemm.alphas.bin", "main-workspace/translations-models-v2/74c6f502-909a-4683-a5f3-e1283ad086a9.zst",
         31_973_916, "dadb385632174c77ab57c130c9e3cab1cd1a6874fe85b799a8b0af439fe78c91",
         43_977_787, "3a603e20bfe1be86071913f9e23ab5129075bc0a8490151020ac4821e4f17302"),
        ("vocab.jaen.spm", "main-workspace/translations-models-v2/9c331ae2-c079-425a-908b-c1a3192dcfaa.zst",
         617_539, "f41f32e36f879643a47fedbd0616fa12f416da62e23af8620ee4fc7705217acc",
         1_443_222, "5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919"),
        ("lex.50.50.jaen.s2t.bin", "main-workspace/translations-models-v2/55f9a9f0-891f-448b-85b7-d62488b45b1e.zst",
         3_359_102, "571403c44adfa4d545e9e4d537e0e7e56ab6bc3858669db7545c70b0bf39f31e",
         9_348_172, "525f412f0d210536c2933c78ae395fa0bf2b5ee6cc5dda61ebc2e79410ebaee4"),
    ],
}
NOTICE = """Mozilla's Firefox Translations models: en-ja 3.0, ja-en 3.0 and ja-en 3.1, with their vocabularies and shortlists,
as Firefox fetches them (Remote Settings' collection translations-models-v2, unpacked from zstd and not changed).
They are Mozilla's, under the Mozilla Public License 2.0 (https://github.com/mozilla/translations), whose text follows.

"""


def held(data, length, sha256):
    """None if data is length bytes of that sha256, else what is wrong, in words"""
    if len(data) != length:
        return f"{len(data):,} bytes, not {length:,}"
    if (got := hashlib.sha256(data).hexdigest()) != sha256:
        return f"sha256 {got}, not {sha256}"
    return None


def fetched(folder, name, location, size, sha256, length, unpacked):
    """folder/name, unpacked from the attachment at location, held to both lengths and sha256s"""
    target = folder / name
    if target.exists():
        if (wrong := held(target.read_bytes(), length, unpacked)) is None:
            return target
        target.unlink()
        raise SystemExit(f"{target}: {wrong} (deleted: the next build fetches it again)")
    packed = download(CDN + location, folder / (name + ".zst"), said=lambda error: print(f"{name}: {error}; again", file=sys.stderr))
    try:
        if wrong := held(packed.read_bytes(), size, sha256):
            raise SystemExit(f"{CDN + location} ({name}): {wrong}")
        data = zstd.decompress(packed.read_bytes())
        if wrong := held(data, length, unpacked):
            raise SystemExit(f"{CDN + location} ({name}) unpacked to {wrong}")
    finally:
        packed.unlink()
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(data)
    partial.rename(target)
    return target


def main(root):
    for model, files in MODELS.items():
        for name, *facts in files:
            path = fetched(root / model, name, *facts)
            print(f"{model}/{name}: {path.stat().st_size:,} bytes")
    (root / "LICENSE.txt").write_text(NOTICE + (HERE / "LICENSE").read_text())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("python3 fetch_translations.py <folder>")
    main(Path(sys.argv[1]))
