"""Add country attributes to resources/country_aliases.csv.

    python -m shared.add_country_attributes
    python -m shared.add_country_attributes --offline

mledoze/countries: world_region, world_subregion, center_lat, center_lng, landlocked
GeoNames: continent, neighbors, languages, main_language
Wikidata P1622: driving_side ("left", "right", or empty)
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Paths relative to the repository root, so the script works from any folder
ROOT = Path(__file__).resolve().parent.parent
ALIASES_CSV = ROOT / "resources" / "country_aliases.csv"
RAW_DIR = ROOT / "resources" / "country_sources"

MLEDOZE_COMMIT = "6031f5df137c41c76d180c55a0782185ae3ed05b"
MLEDOZE_URL = f"https://raw.githubusercontent.com/mledoze/countries/{MLEDOZE_COMMIT}/countries.json"
GEONAMES_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "OpenSI-course-project/1.0 (https://github.com/mhammer3366/ai_and_visualisation_OpenSI)"

WIKIDATA_QUERY = """
SELECT ?item ?iso2 ?sideLabel WHERE {
  ?item wdt:P297 ?iso2 ;
        wdt:P1622 ?side .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# iso2 -> (side, source url). Only after the report lists the country as missing or conflicting.
DRIVING_SIDE_OVERRIDES = {
    # switched to right-hand traffic in 1945, Wikidata still lists both values
    "AR": ("right", "https://en.wikipedia.org/wiki/Left-_and_right-hand_traffic"),
}

NEW_COLUMNS = [
    "world_region", "world_subregion", "center_lat", "center_lng", "landlocked",
    "continent", "neighbors", "languages", "main_language", "driving_side",
]
COLUMN_SOURCE = {
    "world_region": "mledoze", "world_subregion": "mledoze", "center_lat": "mledoze",
    "center_lng": "mledoze", "landlocked": "mledoze",
    "continent": "GeoNames", "neighbors": "GeoNames",
    "languages": "GeoNames", "main_language": "GeoNames",
    "driving_side": "Wikidata",
}
RAW_FILES = {
    "mledoze": RAW_DIR / "mledoze_extract.json",
    "geonames": RAW_DIR / "geonames_countryInfo.txt",
    "wikidata": RAW_DIR / "wikidata_driving_side.json",
}


def download(url, params=None, accept=None):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        return response.read().decode("utf-8")


def write_source_info(retrieved_at=None):
    path = RAW_DIR / "sources.json"
    if retrieved_at is None and path.exists():
        retrieved_at = json.loads(path.read_text(encoding="utf-8")).get("retrieved_at_utc")
    info = {
        "retrieved_at_utc": retrieved_at,
        "mledoze": {
            "url": MLEDOZE_URL, "commit": MLEDOZE_COMMIT, "license": "ODbL-1.0",
            "columns": ["world_region", "world_subregion", "center_lat", "center_lng", "landlocked"],
        },
        "geonames": {
            "url": GEONAMES_URL, "license": "CC BY 4.0",
            "columns": ["continent", "neighbors", "languages", "main_language"],
        },
        "wikidata": {
            "url": WIKIDATA_URL, "license": "CC0", "property": "P1622 (driving side)",
            "query": WIKIDATA_QUERY.strip(), "columns": ["driving_side"],
        },
        "driving_side_overrides": {
            iso2: {"side": side, "source": source}
            for iso2, (side, source) in DRIVING_SIDE_OVERRIDES.items()
        },
    }
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def fetch_sources():
    print("Downloading mledoze/countries ...")
    keep = ["cca2", "cca3", "region", "subregion", "latlng", "landlocked"]
    extract = [{key: entry.get(key) for key in keep} for entry in json.loads(download(MLEDOZE_URL))]

    print("Downloading GeoNames countryInfo.txt ...")
    geonames = download(GEONAMES_URL)

    print("Querying Wikidata (driving side) ...")
    wikidata = download(WIKIDATA_URL, {"query": WIKIDATA_QUERY, "format": "json"},
                        accept="application/sparql-results+json")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FILES["mledoze"].write_text(json.dumps(extract, ensure_ascii=False, indent=1), encoding="utf-8")
    RAW_FILES["geonames"].write_text(geonames, encoding="utf-8")
    RAW_FILES["wikidata"].write_text(wikidata, encoding="utf-8")
    write_source_info(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return {name: path.read_text(encoding="utf-8") for name, path in RAW_FILES.items()}


def load_saved_sources():
    missing = [str(path) for path in RAW_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"saved copies missing: {missing} - run without --offline first")
    return {name: path.read_text(encoding="utf-8") for name, path in RAW_FILES.items()}


def parse_mledoze(text):
    rows = {}
    for entry in json.loads(text):
        latlng = entry.get("latlng") or []
        rows[entry["cca2"]] = {
            "world_region": entry.get("region") or "",
            "world_subregion": entry.get("subregion") or "",
            "center_lat": latlng[0] if len(latlng) == 2 else "",
            "center_lng": latlng[1] if len(latlng) == 2 else "",
            "landlocked": int(bool(entry.get("landlocked"))),
        }
    return rows


def parse_geonames(text):
    header, rows = None, {}
    for line in text.splitlines():
        if line.startswith("#ISO"):
            header = line[1:].split("\t")
            continue
        if not line.strip() or line.startswith("#"):
            continue
        if header is None:
            raise ValueError("GeoNames header '#ISO ...' not found")
        record = dict(zip(header, line.split("\t")))
        missing = [c for c in ("ISO", "Continent", "Languages", "neighbours") if c not in record]
        if missing:
            raise ValueError(f"GeoNames columns {missing} not in header {header}")
        base = list(dict.fromkeys(
            code.split("-")[0].lower() for code in record["Languages"].split(",") if code
        ))
        neighbors = sorted(code for code in record["neighbours"].split(",") if code)
        rows[record["ISO"]] = {
            "continent": record["Continent"],
            "neighbors": ";".join(neighbors),
            "languages": ";".join(base),
            "main_language": base[0] if base else "",
        }
    return rows


def parse_wikidata(text):
    sides = {}
    for binding in json.loads(text)["results"]["bindings"]:
        iso2 = binding["iso2"]["value"].strip().upper()
        label = binding["sideLabel"]["value"].strip().lower()
        side = "left" if "left" in label else "right" if "right" in label else label
        sides.setdefault(iso2, set()).add(side)

    rows, conflicts = {}, {}
    for iso2, values in sides.items():
        if len(values) == 1:
            rows[iso2] = {"driving_side": next(iter(values))}
        else:
            conflicts[iso2] = sorted(values)
            rows[iso2] = {"driving_side": ""}
    return rows, conflicts


def report(table, sources, conflicts):
    names = dict(zip(table["iso2"], table["prompt_name"]))
    candidates = set(table.loc[table["in_candidates_v1"] == "1", "iso2"])

    def show(codes):
        return ", ".join(f"{c} ({names.get(c, '?')}){' *' if c in candidates else ''}" for c in codes) or "-"

    print("\nNot found per source (* = candidate in v1):")
    for name, rows in sources.items():
        print(f"  {name:9s} {show([c for c in table['iso2'] if c not in rows])}")

    no_side = table.loc[table["driving_side"] == "", "iso2"].tolist()
    print(f"\nNo driving side ({len(no_side)}): {show(no_side)}")
    if conflicts:
        print("Conflicting driving side in Wikidata (left empty):")
        for iso2, values in sorted(conflicts.items()):
            print(f"  {iso2} ({names.get(iso2, '?')}): {values}")
    print("Fill these in DRIVING_SIDE_OVERRIDES after checking a source, then rerun with --offline.")

    try:
        from .data import DRIVE_SIDE_FIXES        # python -m shared.add_country_attributes
    except ImportError:
        try:                                      # python shared/add_country_attributes.py
            sys.path.insert(0, str(ROOT))
            from shared.data import DRIVE_SIDE_FIXES
        except ImportError:
            DRIVE_SIDE_FIXES = None
            print("\nDRIVE_SIDE_FIXES not found in shared/data.py, cross-check skipped (git pull?)")
    if DRIVE_SIDE_FIXES:
        side = dict(zip(table["iso2"], table["driving_side"]))
        disagree = [c for c in DRIVE_SIDE_FIXES if side.get(c) != "left"]
        status = "all left, consistent" if not disagree else f"not left in Wikidata: {show(disagree)}"
        print(f"\nDRIVE_SIDE_FIXES vs Wikidata: {status}")

    left = int((table["driving_side"] == "left").sum())
    right = int((table["driving_side"] == "right").sum())
    print(f"\nDriving side: {left} left, {right} right, {len(no_side)} empty")
    cols = ["iso2", "world_subregion", "continent", "neighbors", "main_language", "driving_side"]
    print(table.loc[table["iso2"].isin(["AT", "DE", "GB", "JP", "IN", "BR"]), cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Add country attributes to country_aliases.csv")
    parser.add_argument("--offline", action="store_true", help="rebuild from copies in resources/country_sources/")
    args = parser.parse_args()

    if not ALIASES_CSV.exists():
        raise FileNotFoundError(f"{ALIASES_CSV} not found - is the script inside shared/?")
    texts = load_saved_sources() if args.offline else fetch_sources()
    mledoze = parse_mledoze(texts["mledoze"])
    geonames = parse_geonames(texts["geonames"])
    wikidata, conflicts = parse_wikidata(texts["wikidata"])
    for iso2, (side, _source) in DRIVING_SIDE_OVERRIDES.items():
        wikidata[iso2] = {"driving_side": side}
        conflicts.pop(iso2, None)

    table = pd.read_csv(ALIASES_CSV, dtype=str, keep_default_na=False)
    table = table.drop(columns=[c for c in NEW_COLUMNS if c in table.columns])
    sources = {"mledoze": mledoze, "GeoNames": geonames, "Wikidata": wikidata}
    for column in NEW_COLUMNS:
        rows = sources[COLUMN_SOURCE[column]]
        table[column] = [rows.get(iso2, {}).get(column, "") for iso2 in table["iso2"]]

    table.to_csv(ALIASES_CSV, index=False)
    if args.offline:
        write_source_info()
    print(f"\nUpdated {ALIASES_CSV.relative_to(ROOT)}: {len(table)} countries, new columns: {', '.join(NEW_COLUMNS)}")
    report(table, sources, conflicts)


if __name__ == "__main__":
    main()