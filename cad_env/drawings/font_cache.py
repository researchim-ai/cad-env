"""Explicit install of pinned external font assets into a user cache, never the repo."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "afbc9134f90733eb5eea0a1c70c56a6d5dd5b758"
BASE = f"https://raw.githubusercontent.com/hikikomori82/osifont/{COMMIT}/"
FONT_SHA = "bed989874c949d88e7803486b35aa56571574827d9f03bbb96b14915cf5fb612"
ASSETS = {
    "osifont.ttf": (BASE + "osifont-lgpl3fe.ttf", FONT_SHA),
    "osifont.sfd": (
        BASE + "osifont.sfd",
        "2cb41cdf8416312d3f68dc7424137d36e06a2ff6dcbd04fc33c09030ad7de75e",
    ),
    "UPSTREAM_README.md": (
        BASE + "README.md",
        "82128d84a2d4929263c41a8c1ff28a2df10d1eda5c3affc68c064fc81eaa1463",
    ),
    "COPYING.GPL-3": (
        "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-15.2.0/COPYING3",
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    ),
    "COPYING.LGPL-3": (
        "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-15.2.0/COPYING3.LIB",
        "a853c2ffec17057872340eee242ae4d96cbf2b520ae27d903e1b2fef1a5f9d1c",
    ),
}


def cache_dir():
    root = Path(
        os.environ.get(
            "CAD_ENV_CACHE",
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")),
        )
    )
    return root / "cad-env" / "fonts" / COMMIT


def font_path():
    return cache_dir() / "osifont.ttf"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_font():
    path = font_path()
    if not path.is_file():
        raise FileNotFoundError(
            "Technical font is not installed. Run: python -m cad_env.drawings.font_cache --install"
        )
    if digest(path) != FONT_SHA:
        raise ValueError(
            "Cached font checksum mismatch; reinstall with python -m cad_env.drawings.font_cache --install"
        )
    return path


def provenance():
    return {
        "name": "osifont",
        "upstream_commit": COMMIT,
        "font_sha256": FONT_SHA,
        "download_url": ASSETS["osifont.ttf"][0],
        "copyright": "Zefram Cochrane / hikikomori82 and contributors",
        "license": "LGPL version 3 with GPL font exception; editable source carries its own GPLv3/font-exception notice",
        "upstream": "https://github.com/hikikomori82/osifont",
        "font_binary_bundled": False,
    }


def install(destination=None, fetch=None):
    folder = Path(destination) if destination else cache_dir()
    folder.mkdir(parents=True, exist_ok=True)

    def download(url):
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(3000001)
        if len(data) > 3000000:
            raise ValueError("Font download exceeds expected size limit")
        return data

    fetch = fetch or download
    for name, (url, expected) in ASSETS.items():
        path = folder / name
        if path.is_file() and digest(path) == expected:
            continue
        data = fetch(url)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Downloaded font asset checksum mismatch: " + name)
        # A failed/network-interrupted download never replaces a verified asset.
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
            stream.write(data)
            temporary = Path(stream.name)
        temporary.replace(path)
    (folder / "provenance.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    return folder


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--install", action="store_true")
    args = p.parse_args()
    if args.install:
        print(install())
    else:
        print(require_font())


if __name__ == "__main__":
    main()
