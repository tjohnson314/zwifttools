import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


MASTER_URL = "https://zwiftinsider.com/routes/"
WORLD_IDS = {
    "Watopia": 1, "Richmond": 2, "London": 3, "New York": 4,
    "Innsbruck": 5, "Bologna": 6, "Yorkshire": 7, "Crit City": 8,
    "Makuri Islands": 9, "France": 10, "Paris": 11,
    "Gravel Mountain": 12, "Scotland": 13,
}


class RoutePageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.parts = []
        self.row = None
        self.cell = None
        self.link = None
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in ("script", "style"):
            self.ignored += 1
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = {"text": [], "links": [], "link_texts": {}}
        elif tag == "a" and self.cell is not None:
            self.cell["links"].append(attributes.get("href", ""))
            self.link = (attributes.get("href", ""), [])

    def handle_data(self, data):
        if self.ignored:
            return
        self.parts.append(data)
        if self.cell is not None:
            self.cell["text"].append(data)
        if self.link is not None:
            self.link[1].append(data)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored = max(0, self.ignored - 1)
        if tag == "a" and self.link is not None:
            if self.cell is not None:
                self.cell["link_texts"][self.link[0]] = " ".join(" ".join(self.link[1]).split())
            self.link = None
        if tag in ("td", "th") and self.cell is not None:
            self.cell["text"] = " ".join(" ".join(self.cell["text"]).split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    @property
    def text(self):
        return " ".join(" ".join(self.parts).split())


def normalize_name(value):
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.casefold())


def metric_distance(value):
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(km|m)\b", value, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1).replace(",", ""))
    return number * (1000 if match.group(2).casefold() == "km" else 1)


def parse_master_routes(html):
    parser = RoutePageParser()
    parser.feed(html)
    routes = []
    for cells in parser.rows:
        if len(cells) != 6 or cells[1]["text"] not in WORLD_IDS:
            continue
        links = [urljoin(MASTER_URL, link) for link in cells[0]["links"]
                 if "/route/" in link]
        if len(links) != 1:
            continue
        routes.append({
            "name": cells[0]["text"], "world": cells[1]["text"],
            "map_id": WORLD_IDS[cells[1]["text"]], "url": links[0],
            "master_distance_text": cells[2]["text"],
            "master_distance_m": metric_distance(cells[2]["text"]),
            "master_leadin_text": cells[4]["text"],
            "master_leadin_m": metric_distance(cells[4]["text"]),
            "restriction": cells[5]["text"],
        })
    if not routes or len({route["url"] for route in routes}) != len(routes):
        raise ValueError("Master route rows are missing or contain duplicate detail URLs")
    return routes


def parse_route_page(html):
    parser = RoutePageParser()
    parser.feed(html)
    text = parser.text
    beginning = text.find("Route Basics")
    ending = text.find("Start & Finish", beginning)
    basics = text[beginning:ending if ending > beginning else beginning + 2500] if beginning >= 0 else ""
    length = re.search(r"Length:\s*([\d.,]+\s*km)\b", basics, re.IGNORECASE)
    leadin = re.search(r"\+\s*([\d.,]+\s*km).*?lead-in", basics, re.IGNORECASE)
    segments = []
    section_present = False
    section_kind = None
    collecting = False
    for cells in parser.rows:
        headers = [cell["text"] for cell in cells if cell["text"] in ("Leaderboard Segments", "Sprint & KQOM Segments")]
        if headers:
            collecting = section_present = True
            section_kind = headers[0]
            continue
        if not collecting:
            continue
        for cell in cells:
            for link in cell["links"]:
                if "/segment/" in link:
                    segments.append({
                        "name": re.sub(r"\s*\([\d.,]+\s*km.*$", "",
                                       cell["link_texts"].get(link) or cell["text"]).strip(),
                        "url": urljoin(MASTER_URL, link), "raw_text": cell["text"],
                    })
    return {
        "detail_distance_text": length.group(1) if length else None,
        "detail_distance_m": metric_distance(length.group(1)) if length else None,
        "detail_leadin_text": leadin.group(1) if leadin else None,
        "detail_leadin_m": metric_distance(leadin.group(1)) if leadin else None,
        "basics_text": basics, "leaderboard_section_present": section_present,
        "segment_section_kind": section_kind,
        "segments": segments,
    }


def source_member(url):
    return "master.html" if url == MASTER_URL else urlparse(url).path.strip("/") + ".html"


def collect_pages(output_dir, limit=50, workers=3):
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "sources.zip"
    if not archive_path.exists():
        response = requests.get(MASTER_URL, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("master.html", response.text)
    with zipfile.ZipFile(archive_path) as archive:
        routes = parse_master_routes(archive.read("master.html").decode("utf-8"))
        existing = set(archive.namelist())
    pending = [route for route in routes if source_member(route["url"]) not in existing]
    selected = pending[:limit] if limit else pending
    error_path = output_dir / "fetch_errors.json"
    errors = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else {}

    def fetch(route):
        response = requests.get(route["url"], timeout=30)
        response.raise_for_status()
        details = parse_route_page(response.text)
        if details["detail_distance_m"] is None:
            raise ValueError("Route Basics metric length was not found")
        return response.text

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, route): route for route in selected}
        with zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED) as archive:
            for future in as_completed(futures):
                route = futures[future]
                try:
                    html = future.result()
                except Exception as error:
                    errors[route["url"]] = str(error)
                    continue
                archive.writestr(source_member(route["url"]), html)
                existing.add(source_member(route["url"]))
                errors.pop(route["url"], None)
    error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    complete = sum(source_member(route["url"]) in existing for route in routes)
    print(f"Detail pages: {complete}/{len(routes)} cached; {len(routes) - complete} remaining; "
          f"{len(errors)} fetch/parser errors", flush=True)
    return routes, complete, errors


def read_sources(output_dir):
    with zipfile.ZipFile(output_dir / "sources.zip") as archive:
        routes = parse_master_routes(archive.read("master.html").decode("utf-8"))
        existing = set(archive.namelist())
        for route in routes:
            member = source_member(route["url"])
            if member not in existing:
                route["page_status"] = "not fetched"
                continue
            html = archive.read(member)
            route.update(parse_route_page(html.decode("utf-8")))
            route["page_status"] = "fetched"
            route["source_sha256"] = hashlib.sha256(html).hexdigest()
    return routes


def match_routes(published, routes_dir):
    aliases = {
        (6, "Time Trial Lap"): "Bologna Time Trial",
        (3, "London Classique"): "Classique",
        (4, "Shuman Trail Loop Reverse"): "Shuman Trail Reverse",
        (1, "Bambino Fondo"): "Zwift Bambino Fondo",
        (1, "Medio Fondo"): "Zwift Medio Fondo",
        (1, "Gran Fondo"): "Zwift Gran Fondo",
    }
    cache = defaultdict(list)
    for path in routes_dir.glob("world_[0-9]*.json"):
        world_id = int(path.stem.removeprefix("world_"))
        for route in json.loads(path.read_text(encoding="utf-8"))["routes"]:
            cache[(world_id, normalize_name(route["name"]))].append(route)
    for entry in published:
        alias = aliases.get((entry["map_id"], entry["name"]), entry["name"])
        candidates = cache[(entry["map_id"], normalize_name(alias))]
        if "run only" in entry["restriction"].casefold():
            candidates = [route for route in candidates if route["sport_type"] == 2]
        else:
            cycling = [route for route in candidates if route["sport_type"] != 2]
            candidates = cycling or candidates
        entry["cache_match"] = candidates[0] if len(candidates) == 1 else None
        entry["cache_match_status"] = "matched" if len(candidates) == 1 else "ambiguous" if candidates else "not cached"
        entry["cache_match_method"] = "verified native name/identifier alias" if alias != entry["name"] else "world and normalized name"
    return published


def segment_family(name):
    name = re.sub(r"\s*@\s*[\d.]+\s*km.*$", "", name)
    name = re.sub(r"\b(reverse|ccw|forward)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(Watopia|Zwift)\s+", "", name, flags=re.IGNORECASE)
    aliases = {
        "volcanocircuit": "volcano", "hillyroute": "hillyloop",
        "parkperimeterloop": "centralparkloop", "timessquarecircuit": "timessquare",
        "champselyseesloop": "champselysees", "critcityloop": "critcity",
        "redrockloop": "redrocklap", "luteceexpress": "champselysees",
        "downtowndolphin": "critcity", "thebelllap": "critcity",
        "libbyhillkom": "libbyhill", "23rdstkom": "23rdst", "yorkshirekom": "kom",
    }
    normalized = normalize_name(name)
    return aliases.get(normalized, normalized)


def resolve_segment_catalog(published, root):
    catalog = {}
    by_name = {}
    for filename in ("zwift_sprint_accuracy_audit.csv", "zwift_kom_accuracy_audit.csv"):
        path = root / "race_data" / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("hash") and row.get("world") in WORLD_IDS:
                    catalog[(WORLD_IDS[row["world"]], row["detail_url"].rstrip("/"))] = {
                        "hash": int(row["hash"]), "method": "prior verified sprint/KOM identity",
                    }
                    by_name[(WORLD_IDS[row["world"]], normalize_name(row["name"]))] = {
                        "hash": int(row["hash"]), "method": "same world/public name as prior verified sprint/KOM identity",
                    }
    public = defaultdict(list)
    native = {}
    loop_path = root / "race_data" / "zwift_loop_accuracy_audit.csv"
    if loop_path.exists():
        with loop_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                native[(int(row["map_id"]), int(row["hash"]))] = {
                    "hash": int(row["hash"]), "name": row["source_name"],
                }
    for entry in published:
        for segment in entry.get("segments", []):
            public[(entry["map_id"], segment["url"].rstrip("/"))].append((entry, segment))
        route = entry.get("cache_match")
        if route:
            for segment in route.get("segments", []):
                native[(entry["map_id"], segment["hash"])] = segment
    for key, occurrences in public.items():
        if key in catalog:
            continue
        public_name = re.sub(r"\s*@\s*[\d.]+\s*km.*$", "", occurrences[0][1]["name"])
        prior = by_name.get((key[0], normalize_name(public_name)))
        if prior:
            catalog[key] = prior
            continue
        family = segment_family(public_name)
        candidates = [segment for (world_id, _), segment in native.items()
                      if world_id == key[0] and segment_family(segment["name"]) == family]
        scores = {}
        for candidate in candidates:
            score = 0
            for entry, _ in occurrences:
                route = entry.get("cache_match")
                if route and candidate["hash"] in {
                    segment["hash"] for segment in route.get("segments", []) + route.get("native_segments", [])
                }:
                    score += 1
            scores[candidate["hash"]] = score
        maximum = max(scores.values(), default=0)
        winners = [segment_hash for segment_hash, score in scores.items() if score == maximum]
        if maximum and len(winners) == 1:
            catalog[key] = {"hash": winners[0], "method": "unique source-name family and route association overlap"}
        elif len(candidates) == 1:
            catalog[key] = {"hash": candidates[0]["hash"], "method": "unique world/source-name identity; no complete route passage"}
        elif candidates and not maximum:
            exact = [segment for segment in candidates if normalize_name(segment["name"]) == normalize_name(public_name)]
            if len(exact) == 1:
                catalog[key] = {"hash": exact[0]["hash"], "method": "exact world/source name; no complete route passage; direction not independently verified"}
    return catalog


def profile_length(profile):
    return profile["d"][-1] if profile and profile.get("d") else None


def leadin_lengths(route):
    leadin, main = route.get("leadin"), route.get("route")
    length = profile_length(leadin)
    if length is None:
        return None, None
    if main and main.get("x") and main.get("z"):
        length += math.hypot(main["x"][0] - leadin["x"][-1], main["z"][0] - leadin["z"][-1])
    displayed = length - (route.get("leadin_view") or {}).get("paddock_distance_m", 0)
    return length, displayed


def rounding_half_width(text):
    match = re.search(r"([\d,]+(?:\.(\d+))?)\s*(km|m)\b", text or "", re.IGNORECASE)
    if not match:
        return 0
    if float(match[1].replace(",", "")) == 0:
        return 0
    places = len(match[2] or "")
    return 0.5 * 10 ** -places * (1000 if match[3].lower() == "km" else 1)


def distance_difference(actual, reference):
    return actual - reference if actual is not None and reference is not None else None


def needs_review(difference, text, threshold):
    return difference is not None and abs(difference) > max(threshold, rounding_half_width(text)) + 1e-8


def compare_routes(published, catalog, distance_threshold=100, leadin_threshold=50):
    output = []
    for entry in published:
        row = {key: value for key, value in entry.items() if key not in ("cache_match", "segments", "basics_text")}
        route = entry.get("cache_match")
        row["published_segments"] = entry.get("segments", [])
        row["issues"], row["coverage_notes"], row["name_differences"] = [], [], []
        published_leadin = entry.get("detail_leadin_m")
        if published_leadin is None:
            published_leadin = entry["master_leadin_m"]
        row["published_leadin_reference_m"] = published_leadin
        row["published_leadin_reference_source"] = "detail" if entry.get("detail_leadin_m") is not None else "master (detail omits lead-in)"
        if not route:
            row["issues"].append("Route not present in our cache")
            row["coverage_notes"].append("No distance or native segment comparison available; legacy Gravel Mountain is not assumed to be the modern Red Rock Loop")
            output.append(row)
            continue
        row.update({"own_name": route["name"], "own_hash": str(route["nameHash"]), "own_sport_type": route["sport_type"],
                    "own_header_distance_m": route.get("distance_m"), "own_header_leadin_m": route.get("leadin_distance_m"),
                    "own_measured_distance_m": profile_length(route.get("route"))})
        raw_leadin, displayed_leadin = leadin_lengths(route)
        row["own_raw_leadin_with_connector_m"] = raw_leadin
        row["own_displayed_leadin_m"] = displayed_leadin
        row["own_omitted_paddock_m"] = (route.get("leadin_view") or {}).get("paddock_distance_m", 0)
        if entry["page_status"] != "fetched":
            row["coverage_notes"].append("Detail page not fetched; no published comparison available")
            output.append(row)
            continue
        leadin_text = entry["detail_leadin_text"] or entry["master_leadin_text"]
        comparisons = [
            ("distance_header", route.get("distance_m"), entry["detail_distance_m"], entry["detail_distance_text"], distance_threshold),
            ("distance_geometry", row["own_measured_distance_m"], entry["detail_distance_m"], entry["detail_distance_text"], distance_threshold),
            ("leadin_header", route.get("leadin_distance_m"), published_leadin, leadin_text, leadin_threshold),
            ("leadin_geometry", displayed_leadin, published_leadin, leadin_text, leadin_threshold),
            ("master_detail_distance", entry["master_distance_m"], entry["detail_distance_m"], entry["detail_distance_text"], distance_threshold),
            ("master_detail_leadin", entry["master_leadin_m"], entry["detail_leadin_m"], entry["detail_leadin_text"], leadin_threshold),
        ]
        for label, actual, reference, text, threshold in comparisons:
            difference = distance_difference(actual, reference)
            row[label + "_delta_m"] = difference
            if needs_review(difference, text, threshold):
                row["issues"].append(f"{label}: {difference:+.2f} m")
        if raw_leadin is None:
            row["coverage_notes"].append("Lead-in geometry unavailable (not treated as zero)")
        if row["own_measured_distance_m"] is None:
            row["coverage_notes"].append("Main geometry unavailable")
        if not entry["leaderboard_section_present"]:
            row["coverage_notes"].append("No segment-list section found; segment comparison unavailable")
            output.append(row)
            continue
        raw = {segment["hash"]: segment for segment in route.get("segments", [])}
        displayed = {segment["hash"]: segment for segment in route.get("segments", [])
                     if segment.get("distance_m") is None or segment.get("max_distance_m") is None
                     or round(segment["max_distance_m"] - segment["distance_m"], 2) <= 25}
        native_associations = {segment["hash"] for segment in route.get("native_segments", [])}
        row["own_raw_segments"] = [{"hash": str(segment_hash), "name": segment["name"]} for segment_hash, segment in raw.items()]
        row["own_displayed_segments"] = [{"hash": str(segment_hash), "name": segment["name"]} for segment_hash, segment in displayed.items()]
        row["segment_identities"], row["missing_segments"], row["extra_segments"], row["unresolved_segments"] = [], [], [], []
        expected = set()
        for segment in entry["segments"]:
            identity = catalog.get((entry["map_id"], segment["url"].rstrip("/")))
            if not identity:
                row["unresolved_segments"].append(segment["name"])
                continue
            segment_hash = identity["hash"]
            expected.add(segment_hash)
            row["segment_identities"].append({"public_name": segment["name"], "url": segment["url"],
                                              "hash": str(segment_hash), "method": identity["method"]})
            if segment_hash not in displayed:
                reason = "partial passage hidden by existing 25 m display filter" if segment_hash in raw else "native association without complete passage" if segment_hash in native_associations else "no exported passage or native association"
                row["missing_segments"].append(f"{segment['name']} ({reason})")
            elif normalize_name(segment["name"]) != normalize_name(displayed[segment_hash]["name"]):
                own_name = displayed[segment_hash]["name"]
                row["name_differences"].append(f"{segment['name']} = cache {own_name}")
                public_reverse = bool(re.search(r"\b(reverse|ccw)\b", segment["name"], re.IGNORECASE))
                own_reverse = bool(re.search(r"\b(reverse|ccw)\b", own_name, re.IGNORECASE))
                if public_reverse != own_reverse:
                    row["issues"].append(f"Direction label: {segment['name']} = cache {own_name}")
        if not row["unresolved_segments"]:
            for segment_hash, segment in displayed.items():
                if segment_hash in expected:
                    continue
                if entry["segment_section_kind"] == "Sprint & KQOM Segments" and not any(
                    word in segment["name"].casefold() for word in ("sprint", "kom")
                ):
                    continue
                passages = [item for item in route.get("segments", []) if item["hash"] == segment_hash]
                leadin_only = all(item.get("end_checkpoint", float("inf")) < route.get("leadin_checkpoint_count", 0)
                                  for item in passages)
                row["extra_segments"].append(segment["name"] + (" (lead-in only)" if leadin_only else ""))
        for label in ("missing_segments", "extra_segments"):
            row[label] = sorted(set(row[label]))
            if row[label]:
                row["issues"].append(label + ": " + "; ".join(row[label]))
        if row["unresolved_segments"]:
            row["coverage_notes"].append("Unresolved public segment identities: " + "; ".join(sorted(set(row["unresolved_segments"]))))
        row["issues"] = list(dict.fromkeys(row["issues"]))
        row["name_differences"] = list(dict.fromkeys(row["name_differences"]))
        output.append(row)
    return output


def write_reports(rows, catalog, root, output_dir, distance_threshold, leadin_threshold):
    csv_path = root / "race_data" / "zwift_route_accuracy_audit.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    (output_dir / "audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    identity_rows = [{"map_id": key[0], "url": key[1], "hash": str(value["hash"]), "method": value["method"]} for key, value in sorted(catalog.items())]
    (output_dir / "segment_identities.json").write_text(json.dumps(identity_rows, indent=2), encoding="utf-8")
    significant = [row for row in rows if row["issues"]]
    gaps = [row for row in rows if row["coverage_notes"]]
    categories = Counter(label for row in rows for label in {issue.split(":")[0] for issue in row["issues"]})
    omitted_leadins = sum(row.get("detail_leadin_m") is None and row["master_leadin_m"] == 0 for row in rows)
    lines = ["# Zwift Insider Master Route Audit", "", f"Audit generated: {datetime.now(timezone.utc).isoformat()}", "",
             f"Master list: {MASTER_URL}", f"- Master routes: {len(rows)}; detail pages fetched: {sum(row['page_status'] == 'fetched' for row in rows)}.",
             f"- Cached route matches: {sum(row['cache_match_status'] == 'matched' for row in rows)}; significant-difference rows: {len(significant)}; rows with coverage caveats: {len(gaps)}.",
             "- These are review flags, not a count of proven bad routes. Category row counts overlap: " + "; ".join(f"{key}={value}" for key, value in sorted(categories.items())) + ".",
             f"- Route-distance review: >{distance_threshold:g} m. Lead-in review: >{leadin_threshold:g} m. Differences within half the displayed metric precision are not flagged; an explicit zero/no lead-in is treated as exact.",
             "- This is a native-cache comparison, not independent in-game/Strava certification. Lead-in headers, event pens, free-ride starts and displayed paddock omission may describe different starts.",
             "- Header and sampled planar main-route lengths are separate. Lead-in geometry includes the last-sample/main-start connector; displayed length subtracts only an existing paddock view prefix. Missing geometry is blank, never zero.",
             "- Segment matching uses prior verified hashes or world/source-name and route-association evidence, never length. All raw/public names and identity methods are retained in the CSV/JSON; uncertain identities remain explicit.",
             "- Segment sets ignore repeated crossings. Existing partial-passage display filtering (>25 m below cached maximum) is reported separately from absent raw extraction. Older Sprint & KQOM tables do not claim lap coverage. Extra lead-in-only passages are labeled; omissions may reflect published main-route scope rather than wrong geometry.",
             f"- {omitted_leadins} detail pages omit a lead-in and have explicit zero in the master list; that master value is the comparison reference, not an invented detail-page zero.",
             "- ZIP source snapshot and structured evidence: zwift_route_master_audit_sources/. Full unrounded comparisons: zwift_route_accuracy_audit.csv.",
             "- Own-cache SHA256 checksums are stored in zwift_route_master_audit_sources/route_cache_sha256.json; the generator verifies that cached route files remain unchanged.",
             "", "## Significant Differences", "", "All values below are kilometres. Own values are header / measured geometry; lead-in geometry is the displayed prefix-adjusted length.", "",
             "| Route | World | ZI Distance | Own Distance (Header / Geometry) | ZI Lead-In | Own Lead-In (Header / Geometry) | Findings |",
             "|---|---|---:|---:|---:|---:|---|"]

    def km(value):
        return "n/a" if value is None else f"{value / 1000:.3f}"

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for row in significant:
        findings = "; ".join(row["issues"] + row["coverage_notes"])
        lines.append(f"| [{escape(row['name'])}]({row['url']}) | {row['world']} | {km(row.get('detail_distance_m'))} | "
                     f"{km(row.get('own_header_distance_m'))} / {km(row.get('own_measured_distance_m'))} | {km(row.get('published_leadin_reference_m'))} | "
                     f"{km(row.get('own_header_leadin_m'))} / {km(row.get('own_displayed_leadin_m'))} | {escape(findings)} |")
    lines += ["", "## Coverage Caveats", "", "| Route | World | Remaining Evidence Gap |", "|---|---|---|"]
    for row in gaps:
        lines.append(f"| [{escape(row['name'])}]({row['url']}) | {row['world']} | {escape('; '.join(row['coverage_notes']))} |")
    (root / "race_data" / "zwift_route_accuracy_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports: {len(rows)} routes; {len(significant)} significant rows; {len(gaps)} coverage caveats", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Audit every linked Zwift Insider master-list route against native cached data.")
    parser.add_argument("--output-dir", type=Path, default=Path("race_data/zwift_route_master_audit_sources"))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=50, help="Maximum new pages to fetch; 0 fetches all remaining pages.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--distance-threshold-m", type=float, default=100)
    parser.add_argument("--leadin-threshold-m", type=float, default=50)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache_files = list((root / "zwift_routes").glob("*.json"))
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cache_files}
    if arguments.collect:
        collect_pages(arguments.output_dir, arguments.limit, arguments.workers)
    published = match_routes(read_sources(arguments.output_dir), root / "zwift_routes")
    catalog = resolve_segment_catalog(published, root)
    rows = compare_routes(published, catalog, arguments.distance_threshold_m, arguments.leadin_threshold_m)
    write_reports(rows, catalog, root, arguments.output_dir, arguments.distance_threshold_m, arguments.leadin_threshold_m)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cache_files}
    if checksums != after:
        raise RuntimeError("Own route cache changed during the audit; rerun against a stable cache")
    (arguments.output_dir / "route_cache_sha256.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()