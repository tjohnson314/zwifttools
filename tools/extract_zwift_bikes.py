"""Extract authoritative bike frame/fork/handlebar and wheel aerodynamic and
weight data directly from the Zwift game client.

Zwift models every bike component as a weight (``<WeightGrams>``) plus an
aerodynamic drag-area delta (``<CdABias>``, in m^2) applied on top of a baseline
rider+bike CdA. Newer physics values are stored alongside as ``<WeightGramsV2>``
and ``<CdABiasV2>``. This is the raw data ZwifterBikes measures empirically.

The component XML payloads inside ``bikes_config.wad`` and each
``Frames/<Frame>/frame.wad`` / ``Wheels/data.wad`` are obfuscated with a
repeating 182-byte XOR key. Each entry is XORed with the key at a per-file phase
(the key offset that yields readable XML). This tool decompresses the WADs
(reusing the ZWF! decoder), auto-detects each file's phase, decrypts, parses the
component values, and writes:

    zwiftdata/game_frames.json   full bike (frame+fork+handlebars+drivetrain)
                                 weight + CdA bias, whole-bike price/level, and
                                 the assigned upgrade path (definitions exported separately)
    zwiftdata/game_wheels.json   front/rear wheel weight + CdA, price/level,
                                 TT-specific CdA, and per-surface Crr

Wheel fronts and rears are paired via the game's ``BikeComponents`` registry
(``pairingHashes``), so composite disc sets whose front and rear live in
different folders -- e.g. the Zipp 858/Super9 (858 front + Super9 disc rear) --
are emitted as a single wheelset rather than two halves.

Usage:
    python tools/extract_zwift_bikes.py [--zwift-dir "C:\\Program Files (x86)\\Zwift"] [--out zwiftdata]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries  # noqa: E402

# Repeating 182-byte XOR key used to obfuscate Zwift bike component XML.
KEY = bytes.fromhex(
    "eae40810eb13ebf1eb0f17d800d8ded8fc04df07dfe5df030be60ee6ece60a12"
    "ed15edf3edf7ffda02dae0dafe06e109e1e7e1050de810e8eee80c14ef17efdb"
    "d5f901dc04dce2dc0008e30be3e9e3070fea12eaf0ea0e16f1ffd7ddd7fb03de"
    "06dee4de020ae50de5ebe50911ec14ecf2ec10fed901d9dfd9fd05e008e0e6e0"
    "040ce70fe7ede70b13ee16eef4d4f800db03dbe1dbff07e20ae2e8e2060ee911"
    "e9efe90d15f018d6dcd6fa02dd05dde3dd0109e40ce4"
)
KEYLEN = len(KEY)
_PRINT = set([9, 10, 13]) | set(range(0x20, 0x7f))


def decrypt(data: bytes) -> str:
    """Auto-detect the per-file key phase and return the decrypted text."""
    probe = data[:500] or data
    best_phase, best_score = 0, -1.0
    for phase in range(KEYLEN):
        good = sum(
            1 for i, b in enumerate(probe)
            if (b ^ KEY[(i + phase) % KEYLEN]) in _PRINT
        )
        if good > best_score:
            best_score, best_phase = good, phase
    dec = bytes(data[i] ^ KEY[(i + best_phase) % KEYLEN] for i in range(len(data)))
    return dec.decode("utf-8", "replace")


def _text(el: ET.Element | None):
    return el.text.strip() if el is not None and el.text else None


def _fnum(el: ET.Element, tag: str):
    t = _text(el.find(tag))
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _inum(el: ET.Element, tag: str):
    v = _fnum(el, tag)
    return int(v) if v is not None else None


def parse_component(xml_text: str) -> dict | None:
    """Parse a <Component> payload into weight/CdA fields (with V2 variants)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    if root.tag != "Component":
        return None
    return {
        "name": _text(root.find("Name")),
        "weight_g": _inum(root, "WeightGrams"),
        "cda_bias": _fnum(root, "CdABias"),
        "weight_g_v2": _inum(root, "WeightGramsV2"),
        "cda_bias_v2": _fnum(root, "CdABiasV2"),
        "price": _inum(root, "Price"),
        "level": _inum(root, "Level"),
    }


def _looks_xml(data: bytes) -> bool:
    """True if the raw bytes are already plaintext XML (Components files)."""
    head = data[:64].lstrip(b"\xef\xbb\xbf").lstrip()
    return head[:1] == b"<"


def load_component(data: bytes) -> dict | None:
    """Parse a component entry, decrypting only XOR-obfuscated payloads.

    Frame/fork/wheel components (packed in the Frames/Wheels WADs and
    ``bikes_config.wad``) are XOR-obfuscated, but drivetrain components under
    ``Components/`` are stored as plaintext XML and must not be decrypted.
    """
    text = data.decode("utf-8", "replace") if _looks_xml(data) else decrypt(data)
    return parse_component(text)


# Surface types Zwift models a per-wheel rolling resistance / control factor for.
_SURFACES = ["Pavement", "Wood", "Grass", "Brick", "Cobble", "Dirt", "Snow", "Gravel"]


def parse_wheel_component(data: bytes) -> dict | None:
    """Parse a wheel component, adding wheel-only fields on top of the base:

    * ``cda_bias_tt``  -- CdA bias applied when the wheel is on a TT bike (only a
      handful of deep/disc wheels define it; often rear-only).
    * ``surface_crr``  -- absolute rolling resistance (Crr) per surface type, for
      the few wheels that override it (all-around / off-road wheels).
    * ``surface_cf``   -- per-surface control factor (grip/handling), rarer still.
    """
    text = data.decode("utf-8", "replace") if _looks_xml(data) else decrypt(data)
    comp = parse_component(text)
    if comp is None:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return comp
    comp["cda_bias_tt"] = _fnum(root, "CdABiasTT")
    crr = {s.lower(): _fnum(root, f"{s}RR") for s in _SURFACES}
    comp["surface_crr"] = {k: v for k, v in crr.items() if v is not None} or None
    cf = {s.lower(): _fnum(root, f"{s}CF") for s in _SURFACES}
    comp["surface_cf"] = {k: v for k, v in cf.items() if v is not None} or None
    return comp


# Bike-config component references -> logical component keys. A Zwift bike's
# total (minus wheels) is the sum of these components, resolved through the
# component index below.
_REF_FIELDS = {
    "frame": "Frame",
    "fork": "Fork",
    "handlebars": "Handlebars",
    "crank": "Crank",
    "front_derailleur": "FrontDerailleur",
    "rear_derailleur": "RearDerailleur",
    "brake": "Brake",
}

# Order used when composing the per-bike component breakdown.
_COMPONENT_ORDER = [
    "frame", "fork", "handlebars", "crank",
    "front_derailleur", "rear_derailleur", "brake", "cassette",
]


def _norm_ref(path: str) -> str:
    """Normalise a component path to a lowercase ``bikes/...`` index key."""
    p = path.replace("\\", "/").lower()
    if p.startswith("data/"):
        p = p[5:]
    return p


_UPGRADE_PATHS = {"distance": "road", "elevation": "climbing", "time": "tt"}


def _upgrade_path(level_up: str | None) -> str | None:
    """Derive the upgrade path (``road``/``climbing``/``tt``) from ``<LevelUp>``.

    ``<LevelUp>`` is formatted ``{path}_{tier}`` e.g. ``elevation_highend``.
    """
    if not level_up:
        return None
    return _UPGRADE_PATHS.get(level_up.split("_", 1)[0].lower())


def build_bike_configs(base: str) -> dict[str, dict]:
    """Map stable bike IDs to configs, retaining variants sharing components."""
    localization = read_wad_entries(os.path.join(base, "global.wad"))
    text = localization["Localization/Localization.xml"]
    # The client places a VERSION element before the localization root.
    root = ET.fromstring(text[text.index(b"<ZL>"):])
    names = {
        entry.attrib["LOC_ID"]: english.attrib["value"]
        for entry in root.findall("Entry")
        if (english := entry.find("EN")) is not None and english.get("value")
    }
    cfg = read_wad_entries(os.path.join(base, "Bikes", "bikes_config.wad"))
    by_folder: dict[str, list[tuple[str, dict]]] = {}
    configs: dict[str, dict] = {}
    unresolved: list[str] = []
    for name, data in cfg.items():
        if not (name.startswith("bikes/") and name.endswith("Config.xml")):
            continue
        if "/Wheels/" in name or "/Components/" in name:
            continue
        try:
            root = ET.fromstring(decrypt(data))
        except ET.ParseError:
            continue
        if root.tag != "Bike":
            continue
        frame_path = _text(root.find("Frame"))
        if not frame_path:
            continue
        folder = _folder_from_path(frame_path)
        if folder is None:
            continue
        refs = {}
        for key, tag in _REF_FIELDS.items():
            t = _text(root.find(tag))
            if t:
                refs[key] = _norm_ref(t)
        cassette = (_text(root.find("Cassette")) or "").lower() == "true"
        name_loc = _text(root.find("NameLoc"))
        display_name = names.get(name_loc)
        if name_loc and not name_loc.startswith("LOC_"):
            display_name = name_loc
        elif name_loc and not display_name:
            unresolved.append(name_loc)
        config_id = os.path.basename(name)[:-len("Config.xml")]
        record = {
            "source_frame_folder": folder,
            "make": _text(root.find("Make")),
            "name": _text(root.find("Name")),
            "name_loc": name_loc,
            "display_name": display_name,
            "type": _text(root.find("Type")),
            "year": _inum(root, "Year"),
            "class": _text(root.find("BikeClass")),
            "level_up": _text(root.find("LevelUp")),
            "price": _inum(root, "Price"),
            "level": _inum(root, "Level"),
            "refs": refs,
            "cassette": cassette,
        }
        by_folder.setdefault(folder, []).append((config_id, record))
    for folder, variants in by_folder.items():
        # Keep the historic folder ID for the base bike, not whichever variant
        # happens to be encountered last in the archive.
        normalized_folder = folder.replace("_", "").lower()
        primary_id = next(
            (bike_id for bike_id, _ in variants
             if bike_id.replace("_", "").lower() == normalized_folder),
            variants[0][0],
        )
        for bike_id, record in variants:
            frame_id = folder if bike_id == primary_id else bike_id
            if frame_id in configs:
                raise ValueError(f"Duplicate bike ID: {frame_id}")
            configs[frame_id] = record
    if unresolved:
        print(f"  unresolved bike name localization keys: {sorted(set(unresolved))}; "
              "using internal names for these bikes", file=sys.stderr)
    return configs


def build_component_index(base: str) -> dict[str, bytes]:
    """Index every component XML entry across all bike WADs by normalised path.

    Spans ``bikes_config.wad``, ``Frames/data.wad``, each legacy
    ``Frames/<folder>/frame.wad``, and every ``Components/**/*.wad`` so that any
    component a bike config references can be resolved regardless of which
    archive packs it.
    """
    bikes = os.path.join(base, "Bikes")
    wads = [
        os.path.join(bikes, "bikes_config.wad"),
        os.path.join(bikes, "Frames", "data.wad"),
    ]
    wads += glob.glob(os.path.join(bikes, "Frames", "*", "frame.wad"))
    wads += glob.glob(os.path.join(bikes, "Components", "**", "*.wad"), recursive=True)
    index: dict[str, bytes] = {}
    for wad in wads:
        if not os.path.isfile(wad):
            continue
        try:
            entries = read_wad_entries(wad)
        except Exception as exc:  # noqa: BLE001
            print(f"  index {os.path.basename(wad)}: FAILED ({exc})", file=sys.stderr)
            continue
        for ename, data in entries.items():
            if ename.lower().endswith(".xml"):
                index.setdefault(_norm_ref(ename), data)
    return index


def build_wheel_pairings(base: str) -> dict[str, list[str]]:
    """Map each front-wheel component path -> the rear component path(s) it is
    paired with, per the game's authoritative ``BikeComponents`` registry.

    Zwift stores some wheelsets as a dedicated front in one folder paired with a
    disc rear in another (e.g. the Zipp 858/Super9 = a 858 front + a Super9 disc
    rear), so grouping raw component files by folder splits them into halves.
    Each ``FRONT_WHEEL`` entry's ``pairingHashes`` attribute names the rear(s) it
    mounts with; ``overrideHashName`` lets us resolve the hashed ``.gde``
    references back to real component files.
    """
    try:
        cfg = read_wad_entries(os.path.join(base, "Bikes", "bikes_config.wad"))
    except Exception as exc:  # noqa: BLE001
        print(f"  wheel pairings: FAILED ({exc})", file=sys.stderr)
        return {}
    hash_to_path: dict[str, str] = {}
    fronts: list[tuple[str, str]] = []
    for name, data in cfg.items():
        try:
            root = ET.fromstring(decrypt(data))
        except ET.ParseError:
            continue
        if root.tag != "bikeComponents":
            continue
        for c in root:
            ctype = c.attrib.get("type", "")
            if "WHEEL" not in ctype:
                continue
            fp = _norm_ref(c.attrib.get("filePath", ""))
            oh = c.attrib.get("overrideHashName", "")
            if oh:
                hash_to_path[_norm_ref(oh)] = fp
            if ctype == "FRONT_WHEEL":
                fronts.append((fp, c.attrib.get("pairingHashes", "")))
        break  # a single bikeComponents registry covers every wheel

    pairings: dict[str, list[str]] = {}
    for fp, hashes in fronts:
        rears: list[str] = []
        for tok in hashes.split(","):
            tok = _norm_ref(tok.strip())
            if not tok:
                continue
            rear = tok if tok.endswith(".xml") else hash_to_path.get(tok)
            if rear and rear not in rears:
                rears.append(rear)
        pairings[fp] = rears
    return pairings


def _setname_from_loc(name: str | None, brand: str) -> str | None:
    """Derive a readable wheelset model from a component's ``LOC_WHEELNAME_*``
    localisation key, dropping the brand prefix and the front/rear suffix.

    Used to name composite sets (e.g. ``LOC_WHEELNAME_ZIPP_858_SUPER9_REAR`` ->
    ``858 SUPER9``) that have no single source folder of their own.
    """
    if not name:
        return None
    s = name
    for pre in ("LOC_WHEELNAME_", "LOC_WHEELS_", "LOC_WHEEL_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    for suf in ("_FRONT", "_REAR", "_NAME"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    b = brand.upper()
    if b and s.upper().startswith(b + "_"):
        s = s[len(b) + 1:]
    s = " ".join(p.capitalize() for p in s.split("_") if p).strip()
    return s or None


def _sum(values):
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _eff_weight(comp: dict):
    """Current-physics weight: V2 value if present, else V1."""
    return comp["weight_g_v2"] if comp.get("weight_g_v2") is not None else comp.get("weight_g")


def _eff_cda(comp: dict):
    """Current-physics CdA bias: V2 value if present, else V1."""
    return comp["cda_bias_v2"] if comp.get("cda_bias_v2") is not None else comp.get("cda_bias")


def _eff_cda_tt(comp: dict):
    """CdA bias on a TT bike: the TT-specific value if the wheel defines one,
    otherwise the wheel's normal effective CdA bias."""
    tt = comp.get("cda_bias_tt")
    return tt if tt is not None else _eff_cda(comp)


def _merge_surface(front: dict | None, rear: dict | None, field: str):
    """Merge a per-surface dict defined on either wheel (Zwift stores the
    wheelset's surface values on just one component). Front takes precedence."""
    merged: dict = {}
    for side in (front, rear):
        if side and side.get(field):
            for k, v in side[field].items():
                merged.setdefault(k, v)
    return merged or None


def _folder_from_path(name: str) -> str | None:
    parts = name.replace("\\", "/").split("/")
    try:
        return parts[parts.index("Frames") + 1]
    except (ValueError, IndexError):
        return None


def _handlebars_fallback_map(index: dict[str, bytes]) -> dict[str, str]:
    """Map a lowercase frame folder -> its integrated handlebars index key.

    Used for bikes whose config leaves ``<Handlebars>`` empty but whose frame
    folder ships its own handlebars component.
    """
    fallback: dict[str, str] = {}
    for key in index:
        if "/frames/" in key and "handlebar" in key:
            folder = key.split("/frames/")[1].split("/")[0]
            fallback.setdefault(folder, key)
    return fallback


def extract_frames(base: str, configs: dict[str, dict], index: dict[str, bytes]) -> list[dict]:
    """Compose each bike's complete non-wheel component set (frame + fork +
    handlebars + full drivetrain) from its config and the component index."""
    hb_fallback = _handlebars_fallback_map(index)
    default_cassette = next(
        (k for k in index if k.endswith("components/cassette/cassette.xml")), None
    )

    out: list[dict] = []
    unresolved: list[tuple] = []
    for folder in sorted(configs):
        cfg = configs[folder]
        refs = dict(cfg["refs"])
        # Fall back to the frame's integrated handlebars when the config omits it.
        if "handlebars" not in refs:
            fb = hb_fallback.get(folder.lower())
            if fb:
                refs["handlebars"] = fb
        # A <Cassette>true</Cassette> flag adds the default (0 g) cassette.
        if cfg.get("cassette") and default_cassette:
            refs.setdefault("cassette", default_cassette)

        components: dict[str, dict] = {}
        for kind in _COMPONENT_ORDER:
            ref = refs.get(kind)
            if not ref:
                continue
            data = index.get(ref)
            if data is None:
                unresolved.append((folder, kind, ref))
                continue
            comp = load_component(data)
            if comp:
                components[kind] = comp
        if not components:
            continue

        rec = {
            "folder": folder,
            "source_frame_folder": cfg.get("source_frame_folder", folder),
            "make": cfg.get("make"),
            "name": cfg.get("name"),
            "name_loc": cfg.get("name_loc"),
            "display_name": cfg.get("display_name"),
            "type": cfg.get("type"),
            "year": cfg.get("year"),
            "class": cfg.get("class"),
            "level_up": cfg.get("level_up"),
            "upgrade_path": _upgrade_path(cfg.get("level_up")),
            "price": cfg.get("price"),
            "level": cfg.get("level"),
            "components": components,
            "frameset_weight_g": _sum(c.get("weight_g") for c in components.values()),
            "frameset_cda_bias": _sum(c.get("cda_bias") for c in components.values()),
            "frameset_weight_g_v2": _sum(c.get("weight_g_v2") for c in components.values()),
            "frameset_cda_bias_v2": _sum(c.get("cda_bias_v2") for c in components.values()),
            "frameset_weight_g_effective": _sum(_eff_weight(c) for c in components.values()),
            "frameset_cda_bias_effective": _sum(_eff_cda(c) for c in components.values()),
        }
        out.append(rec)

    if unresolved:
        print(
            f"  {len(unresolved)} unresolved component refs (first 5): {unresolved[:5]}",
            file=sys.stderr,
        )
    return out


def _pair_composite_wheels(wheels: dict[str, dict], pairings: dict[str, list[str]]) -> None:
    """Fold split front/rear halves into complete sets using ``pairingHashes``.

    Folder grouping leaves a few wheelsets as half-entries because the game
    packs their front and rear in different folders. Using the authoritative
    front->rear pairings we either (a) merge a lone rear into its dedicated
    front-only group (e.g. Zipp 858/Super9), deleting the redundant rear group,
    or (b) clone a shared front into a lone disc-rear group (e.g. Zipp
    808/Super9, whose disc rear reuses the 808 Firecrest front). Complete
    folder-grouped sets are left untouched.
    """
    rear_to_fronts: dict[str, list[str]] = {}
    for fp, rears in pairings.items():
        for rp in rears:
            rear_to_fronts.setdefault(rp, []).append(fp)
    front_index = {
        g["front_path"]: (k, g)
        for k, g in wheels.items()
        if g.get("front") and g.get("front_path")
    }
    to_delete: list[str] = []
    for key, g in list(wheels.items()):
        if not g.get("rear") or g.get("front"):
            continue  # only act on lone-rear groups
        rp = g.get("rear_path")
        for fp in rear_to_fronts.get(rp, []):
            fg = front_index.get(fp)
            if not fg:
                continue
            _fk, fgroup = fg
            set_name = _setname_from_loc(g["rear"].get("name"), g["brand"])
            if not fgroup.get("rear"):
                # Dedicated composite: merge this rear into the front-only group.
                fgroup["rear"] = g["rear"]
                fgroup["rear_path"] = rp
                merged_name = _setname_from_loc(g["rear"].get("name"), fgroup["brand"])
                if merged_name:
                    fgroup["model"] = merged_name
                to_delete.append(key)
            else:
                # Shared front (already a complete set): clone it into this
                # lone-rear group so the disc variant becomes a full set.
                g["front"] = fgroup["front"]
                g["front_path"] = fp
                if set_name:
                    g["model"] = set_name
            break
    for key in to_delete:
        wheels.pop(key, None)


def extract_wheels(base: str) -> list[dict]:
    try:
        entries = read_wad_entries(os.path.join(base, "Bikes", "Wheels", "data.wad"))
    except Exception as exc:  # noqa: BLE001
        print(f"  wheels: FAILED ({exc})", file=sys.stderr)
        return []
    wheels: dict[str, dict] = {}
    for ename, data in entries.items():
        low = ename.lower()
        if low.endswith("front.xml"):
            side = "front"
        elif low.endswith("rear.xml"):
            side = "rear"
        else:
            continue
        parts = ename.replace("\\", "/").split("/")
        # .../Wheels/<Brand>/<Model>/Front.xml
        try:
            wi = parts.index("Wheels")
            brand, model = parts[wi + 1], parts[wi + 2]
        except (ValueError, IndexError):
            brand, model = "", os.path.dirname(ename)
        comp = parse_wheel_component(data)
        if not comp:
            continue
        key = f"{brand}/{model}"
        group = wheels.setdefault(key, {"brand": brand, "model": model})
        group[side] = comp
        group[f"{side}_path"] = _norm_ref(ename)

    # Combine cross-folder front/rear halves (disc wheelsets) into single sets.
    _pair_composite_wheels(wheels, build_wheel_pairings(base))

    out = []
    for key in sorted(wheels):
        w = wheels[key]
        front, rear = w.get("front"), w.get("rear")
        parts = [p for p in (front, rear) if p]
        src = front or rear or {}
        rec = {
            "brand": w["brand"],
            "model": w["model"],
            "name": src.get("name"),
            "price": src.get("price"),
            "level": src.get("level"),
            "front": front,
            "rear": rear,
            "pair_weight_g": _sum(p.get("weight_g") for p in parts),
            "pair_cda_bias": _sum(p.get("cda_bias") for p in parts),
            "pair_weight_g_v2": _sum(p.get("weight_g_v2") for p in parts),
            "pair_cda_bias_v2": _sum(p.get("cda_bias_v2") for p in parts),
            "pair_weight_g_effective": _sum(_eff_weight(p) for p in parts),
            "pair_cda_bias_effective": _sum(_eff_cda(p) for p in parts),
            # CdA bias when mounted on a TT bike (equals the normal effective
            # value for wheels without a TT-specific override).
            "pair_cda_bias_tt": _sum(_eff_cda_tt(p) for p in parts),
            # Per-surface rolling resistance (absolute Crr) / control factor,
            # for the few wheels that override the defaults. None otherwise.
            "surface_crr": _merge_surface(front, rear, "surface_crr"),
            "surface_cf": _merge_surface(front, rear, "surface_cf"),
        }
        out.append(rec)
    return out


def read_zwift_version(zwift_dir: str) -> str | None:
    """Human-readable Zwift client version (the ``sversion`` string, e.g.
    ``1.119.0 (164079)``) from the install's ``Zwift_ver_cur.<build>.xml``, or
    None if it can't be read. A sibling ``Zwift_ver_cur_filename.txt`` names the
    current file; fall back to globbing if it's absent."""
    try:
        names: list[str] = []
        ptr = os.path.join(zwift_dir, "Zwift_ver_cur_filename.txt")
        if os.path.isfile(ptr):
            with open(ptr, encoding="utf-8") as f:
                names.append(f.read().strip())
        names += [os.path.basename(p)
                  for p in glob.glob(os.path.join(zwift_dir, "Zwift_ver_cur.*.xml"))]
        for name in names:
            path = os.path.join(zwift_dir, name) if name else ""
            if not path or not os.path.isfile(path):
                continue
            attrib = ET.parse(path).getroot().attrib
            sv = attrib.get("sversion") or attrib.get("version")
            if sv:
                return sv.strip()
    except Exception:  # noqa: BLE001 -- version is best-effort metadata
        return None
    return None


def _dataset_meta(zwift_version: str | None) -> dict:
    """Leading metadata element written at the top of each dataset. JSON has no
    comments, so this record (skipped by every loader, which key on ``folder`` /
    ``brand``) stamps the extracting Zwift version and date at the top of file."""
    return {
        "_comment": (
            "Auto-generated by tools/extract_zwift_bikes.py from the Zwift game "
            "client; do not edit by hand."
        ),
        "zwift_version": zwift_version,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift")
    ap.add_argument("--out", default="zwiftdata")
    args = ap.parse_args()

    base = os.path.join(args.zwift_dir, "assets")
    if not os.path.isdir(os.path.join(base, "Bikes")):
        print(f"error: {base}\\Bikes not found", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    version = read_zwift_version(args.zwift_dir)
    print(f"zwift version: {version or 'unknown'}")
    meta = _dataset_meta(version)
    configs = build_bike_configs(base)
    index = build_component_index(base)

    frames = extract_frames(base, configs, index)
    with open(os.path.join(args.out, "game_frames.json"), "w", encoding="utf-8") as f:
        json.dump([meta, *frames], f, ensure_ascii=False, indent=1)
    print(f"frames: {len(frames)} -> {args.out}/game_frames.json")

    wheels = extract_wheels(base)
    with open(os.path.join(args.out, "game_wheels.json"), "w", encoding="utf-8") as f:
        json.dump([meta, *wheels], f, ensure_ascii=False, indent=1)
    print(f"wheels: {len(wheels)} -> {args.out}/game_wheels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
