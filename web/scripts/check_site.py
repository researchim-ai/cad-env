"""Static integration checks for the buildless bilingual site and drawing assets."""

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

WEB = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((WEB / "translations.json").read_text())
    pages = {p: BeautifulSoup(p.read_text(), "html.parser") for p in WEB.glob("*.html")}
    checked = 0
    for page, soup in pages.items():
        ids = [e["id"] for e in soup.select("[id]")]
        assert len(ids) == len(set(ids)), (page.name, "duplicate IDs")
        for node in soup.select("[data-i18n]"):
            key = node["data-i18n"]
            assert set(catalog[key]) == {"en", "ru"}, key
            assert all(catalog[key].values()), key
        assert len(soup.select("h1")) == 1, page.name
        assert soup.select_one('button[data-language="en"]')
        assert soup.select_one('button[data-language="ru"]')
        for node in soup.select("[href],script[src],img[src]"):
            value = node.get("href") or node.get("src")
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            assert target.is_file(), (page.name, value, "missing asset")
            if url.fragment and target.suffix == ".html":
                dest = pages.get(target) or BeautifulSoup(
                    target.read_text(), "html.parser"
                )
                assert dest.find(id=url.fragment), (page.name, value, "missing anchor")
            checked += 1
    provenance = json.loads((WEB / "assets/drawings/provenance.json").read_text())
    assert len(provenance["examples"]) == 3
    for example in provenance["examples"]:
        assert example["split"] == "train"
        assert set(example["profiles"]) == {"iso", "gost"}
        for profile in example["profiles"].values():
            assert not profile["full_conformance"]
            for item in profile["files"].values():
                path = WEB / "assets/drawings" / item["path"]
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
                ), path
    assert not list(WEB.rglob("*.ttf"))
    assert not list(WEB.rglob("*.sfd"))
    result = json.loads((WEB / "assets/drawing-pilot-results.json").read_text())
    assert result["profiles"]["iso"]["after_successes"] == 3
    assert result["profiles"]["gost"]["after_successes"] == 6
    print(
        f"Checked {len(pages)} bilingual pages, {checked} local links/assets, drawing hashes and recorded pilot counts."
    )


if __name__ == "__main__":
    main()
