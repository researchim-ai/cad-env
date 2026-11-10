"""Static integration checks for the buildless bilingual site and drawing assets."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

WEB = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((WEB / "translations.json").read_text())
    pages = {p: BeautifulSoup(p.read_text(), "html.parser") for p in WEB.glob("*.html")}
    checked = 0
    for page, soup in pages.items():
        assert not soup.select("#snapshot-time, footer time, footer [data-updated]"), (
            page.name, "automatic update stamps must not appear in page footers"
        )
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
    solutions = json.loads((WEB / "assets/student-solutions/manifest.json").read_text())
    assert solutions["results_by_representation"] == {
        "iso": {"tasks": 90, "successes": 47},
        "gost": {"tasks": 90, "successes": 47},
    }
    assert sum(solutions["paired_task_outcomes"].values()) == 90
    table = pages[WEB / "tuning-results.html"].select_one(".tuning-table")
    for level, expected in {
        "easy": ["0/30", "30/30", "0/30", "25/30"],
        "medium": ["0/30", "14/30", "0/30", "16/30"],
        "hard": ["0/30", "3/30", "0/30", "6/30"],
        "total": ["0/90", "47/90", "0/90", "47/90"],
    }.items():
        assert [
            n.get_text() for n in table.select(f'tr[data-level="{level}"] td')
        ] == expected
    for key, entry in catalog.items():
        if key.startswith(("writeup_", "showcase_")):
            assert all(
                not any(
                    old in value
                    for old in (
                        "94/180",
                        "0/180",
                        "52.2%",
                        "52,2%",
                        "55/60",
                        "30/60",
                        "9/60",
                    )
                )
                for value in entry.values()
            ), key
    assert len(solutions["examples"]) == 6
    assert sum(len(e["solutions"]) for e in solutions["examples"]) == 8
    assert sum(len(e["baseline"]) for e in solutions["examples"]) == 12
    assert Counter(e["difficulty"] for e in solutions["examples"]) == {
        "easy": 2,
        "medium": 2,
        "hard": 2,
    }
    for level in ("easy", "medium", "hard"):
        section = pages[WEB / "tuning-results.html"].select_one("#examples-" + level)
        assert section and len(section.select(".student-example")) == 2
    assert {
        e["family"] for e in solutions["examples"] if e["difficulty"] == "medium"
    } == {"heterogeneous_plate", "multi_support_fixture"}
    for example in solutions["examples"]:
        assert example["split"] == "validation"
        if example["difficulty"] == "hard":
            assert example["pads"] == 18 and example["holes"] == 35
        for group in ("solutions", "baseline"):
            for rep in example[group].values():
                for name, digest in (("file", "sha256"), ("actions", "actions_sha256")):
                    path = WEB / "assets/student-solutions" / rep[name]
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest() == rep[digest]
                    ), path
        for preview in example["previews"].values():
            path = WEB / "assets/student-solutions" / preview["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == preview["sha256"], (
                path
            )
        assert set(example["inputs"]) == {"iso", "gost"}
        for standard, rep in example["inputs"].items():
            assert rep["task_success"] == (standard in example["solutions"])
            assert [im["ordinal"] for im in rep["images"]] == list(
                range(1, len(rep["images"]) + 1)
            )
            for im in rep["images"]:
                path = WEB / "assets/student-solutions" / im["file"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == im["sha256"], (
                    path
                )
    assert not list((WEB / "assets/student-solutions").glob("*.zip"))
    assert not pages[WEB / "tuning-results.html"].select('a[href$=".zip"]')
    assert len(pages[WEB / "tuning-results.html"].select(".model-input")) == 6
    print(
        f"Checked {len(pages)} bilingual pages, {checked} local links/assets, drawing hashes and recorded pilot counts."
    )


if __name__ == "__main__":
    main()
