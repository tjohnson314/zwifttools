"""
Bike data module backed by authoritative Zwift game files.

Frame and wheel stats are sourced from data extracted directly from the Zwift
game install, stored in:

  * zwiftdata/game_frames.json  -- per-frameset weight (grams) and CdA bias (m^2)
  * zwiftdata/game_wheels.json  -- per-wheelset weight (grams) and CdA bias (m^2)

The game encodes aerodynamics as a *CdA bias*: a delta (m^2) applied to the
rider's baseline CdA.  Absolute CdA at the reference rider is therefore

    CdA = BASE_CDA + frame_cda_bias + wheel_cda_bias

Downstream physics multiplies a per-bike ``cd`` by the rider's frontal area, so
we expose an equivalent ``cd`` such that ``cd * REF_FRONTAL_AREA`` reproduces the
absolute CdA above.  This keeps every existing physics call-site working while
sourcing the numbers from the game.  The bike-vs-bike comparison and best-bike
search are differential (evaluated against recorded power), so BASE_CDA cancels
there; it only anchors absolute solo / ride-simulator predictions.

Note: the extracted game data represents base (un-upgraded, "stage 0") bikes.
Zwift's 5-stage bike upgrades are computed at runtime from bike type/tier and are
not present in the game files, so upgrade levels are not modelled here.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Reference rider frontal area (183 cm / 75 kg), matching physics.frontal_area_from_rider.
REF_FRONTAL_AREA = 0.3449
# Baseline absolute CdA (m^2) for a rider on a zero-bias bike at the reference
# rider.  This anchors absolute predictions; it cancels out of all differential
# (vs recorded power) comparisons, so its exact value only affects solo /
# ride-sim absolute watts.  Chosen so aero TT setups land near ~0.22 m^2 and
# neutral road bikes near ~0.27 m^2.
BASE_CDA = 0.2680

# Map Zwift game frame "type" tokens to the frametype strings used by
# shared.surface_lookup (which drives per-surface CRR selection).
_GAME_TYPE_TO_FRAME_TYPE = {
    'STANDARD': 'Standard',
    'TT': 'TT',
    'GRAVEL': 'Gravel',
    'MOUNTAIN': 'MTB',
    'CRUISER': 'Standard',
    'CONCEPT': 'Tron',
    'TRICYCLE': 'Standard',
    'HANDCYCLE': 'Hand',
    'RECUMBENT': 'Standard',
}


def _humanize(text: str, strip_prefix: Optional[str] = None) -> str:
    """Turn a CamelCase/PackedDigits identifier into spaced, readable text."""
    if not text:
        return ''
    text = text.replace('_', ' ')
    if strip_prefix:
        prefix = strip_prefix.replace('_', ' ').strip()
        if prefix and text.lower().lstrip().startswith(prefix.lower()):
            text = text.lstrip()[len(prefix):]
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    s = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', s)
    s = re.sub(r'(?<=\d)(?=[A-Z][a-z])', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


@dataclass
class BikeSetup:
    """Complete bike setup with computed stats (from authoritative game data)."""
    frame_id: str
    frame_name: str
    wheel_id: str
    wheel_name: str
    upgrade_level: int
    cd: float  # Equivalent drag coefficient (cd * frontal_area = absolute CdA)
    weight_kg: float  # Total bike weight (frame + wheels) in kg
    frame_type: str  # 'Standard', 'TT', 'Gravel', 'MTB', 'Tron', 'Hand'
    # Authoritative game values (base / stage 0)
    cda_bias: float = 0.0        # Total CdA bias (frame + wheels), m^2
    frame_weight_g: float = 0.0
    frame_cda_bias: float = 0.0
    wheel_weight_g: float = 0.0
    wheel_cda_bias: float = 0.0

    def __str__(self):
        return f"{self.frame_name} + {self.wheel_name}"


class BikeDatabase:
    """Frames, wheels, and combined stats sourced from the Zwift game files."""

    DATA_DIR = Path(__file__).parent.parent / 'zwiftdata'
    GAME_FRAMES = 'game_frames.json'
    GAME_WHEELS = 'game_wheels.json'

    def __init__(self):
        self.frames: Dict[str, dict] = {}
        self.wheels: Dict[str, dict] = {}
        self.bikes: Dict[Tuple[str, str], dict] = {}  # (frame_id, wheel_id) -> {'cd':[...], 'weight':[...]}
        self._load_data()

    def _read_json(self, name: str):
        with open(self.DATA_DIR / name, encoding='utf-8-sig') as f:
            return json.load(f)

    def _load_data(self):
        self._load_frames()
        self._load_wheels()
        self._build_combos()

    def _load_frames(self):
        raw = self._read_json(self.GAME_FRAMES)
        self.frames = {}
        for fr in raw:
            if not isinstance(fr, dict):
                continue
            folder = fr.get('folder')
            if not folder:
                continue
            weight_g = fr.get('frameset_weight_g_effective')
            cda_bias = fr.get('frameset_cda_bias_effective')
            if weight_g is None or cda_bias is None:
                # Incomplete frameset (e.g. missing frame or fork) -- skip.
                continue
            make = fr.get('make') or ''
            model = _humanize(fr.get('name') or folder, strip_prefix=make)
            if not model:
                model = _humanize(folder)
            frame_type = _GAME_TYPE_TO_FRAME_TYPE.get(fr.get('type'), 'Standard')
            level = fr.get('level')
            self.frames[folder] = {
                'frameid': folder,
                'framemake': make,
                'framemodel': model,
                'frametype': frame_type,
                'framewheeltype': 'any',
                'framelevel': int(level) if isinstance(level, (int, float)) else 0,
                'frameprice': fr.get('price'),      # whole-bike Drops price
                'frameupgradelevel': '',
                'frameweight_g': weight_g,
                'framecda_bias': cda_bias,
                'frameyear': fr.get('year'),
                'frameclass': fr.get('class'),
            }

    def _load_wheels(self):
        raw = self._read_json(self.GAME_WHEELS)
        self.wheels = {}
        seen = set()
        for wh in raw:
            if not isinstance(wh, dict):
                continue
            weight_g = wh.get('pair_weight_g_effective')
            cda_bias = wh.get('pair_cda_bias_effective')
            if weight_g is None or cda_bias is None:
                continue
            brand = wh.get('brand') or ''
            model = wh.get('model') or ''
            wheel_id = _slug(f'{brand}{model}')
            if not wheel_id or wheel_id in seen:
                # Ensure uniqueness for the rare duplicate slug.
                base = wheel_id or 'wheel'
                i = 2
                while f'{base}{i}' in seen:
                    i += 1
                wheel_id = f'{base}{i}'
            seen.add(wheel_id)
            level = wh.get('level')
            self.wheels[wheel_id] = {
                'wheelid': wheel_id,
                'wheelmake': brand,
                'wheelmodel': _humanize(model, strip_prefix=brand),
                'wheelfitsframe': 'Standard,TT,Gravel,MTB,Tron,Hand',
                'wheellevel': int(level) if isinstance(level, (int, float)) else 0,
                'wheelprice': wh.get('price'),
                'wheelweight_g': weight_g,
                'wheelcda_bias': cda_bias,
            }

    def _combo_cd_weight(self, frame: dict, wheel: Optional[dict]) -> Tuple[float, float]:
        frame_bias = float(frame.get('framecda_bias') or 0.0)
        frame_wt = float(frame.get('frameweight_g') or 0.0)
        wheel_bias = float(wheel.get('wheelcda_bias') or 0.0) if wheel else 0.0
        wheel_wt = float(wheel.get('wheelweight_g') or 0.0) if wheel else 0.0
        cda = BASE_CDA + frame_bias + wheel_bias
        cd = cda / REF_FRONTAL_AREA
        weight_kg = (frame_wt + wheel_wt) / 1000.0
        return cd, weight_kg

    def _build_combos(self):
        """Precompute (frame, wheel) combos with cd/weight arrays (6 identical stages)."""
        self.bikes = {}
        for fid, frame in self.frames.items():
            # Built-in / no separate wheelset option (wheel_id == '')
            cd, wt = self._combo_cd_weight(frame, None)
            self.bikes[(fid, '')] = {'cd': [cd] * 6, 'weight': [wt] * 6}
            for wid, wheel in self.wheels.items():
                cd, wt = self._combo_cd_weight(frame, wheel)
                self.bikes[(fid, wid)] = {'cd': [cd] * 6, 'weight': [wt] * 6}

    def get_bike_stats(self, frame_id: str, wheel_id: Optional[str] = None, upgrade_level: int = 0) -> Optional[BikeSetup]:
        """Get complete bike stats for a frame/wheel combination (game data).

        Args:
            frame_id: Frame folder id (e.g. 'CanyonAeroad2024').
            wheel_id: Wheel id, or None/'' for the frame's built-in wheels.
            upgrade_level: Retained for API compatibility; game data has no
                upgrade stages, so this does not change the result.
        """
        frame = self.frames.get(frame_id)
        if not frame:
            return None

        lookup_wheel_id = wheel_id if wheel_id else ''
        wheel = self.wheels.get(lookup_wheel_id) if lookup_wheel_id else None
        if lookup_wheel_id and wheel is None:
            return None

        cd, weight_kg = self._combo_cd_weight(frame, wheel)
        frame_bias = float(frame.get('framecda_bias') or 0.0)
        wheel_bias = float(wheel.get('wheelcda_bias') or 0.0) if wheel else 0.0

        return BikeSetup(
            frame_id=frame_id,
            frame_name=f"{frame['framemake']} {frame['framemodel']}".strip() or frame_id,
            wheel_id=lookup_wheel_id,
            wheel_name=(f"{wheel['wheelmake']} {wheel['wheelmodel']}".strip() if wheel else '(Built-in wheels)'),
            upgrade_level=upgrade_level,
            cd=cd,
            weight_kg=weight_kg,
            frame_type=frame.get('frametype', 'Standard'),
            cda_bias=frame_bias + wheel_bias,
            frame_weight_g=float(frame.get('frameweight_g') or 0.0),
            frame_cda_bias=frame_bias,
            wheel_weight_g=(float(wheel.get('wheelweight_g') or 0.0) if wheel else 0.0),
            wheel_cda_bias=wheel_bias,
        )

    def list_frames(self, frame_type: Optional[str] = None) -> List[dict]:
        """List all frames, optionally filtered by type."""
        frames = list(self.frames.values())
        if frame_type:
            frames = [f for f in frames if f.get('frametype', '').lower() == frame_type.lower()]
        return sorted(frames, key=lambda f: f"{f['framemake']} {f['framemodel']}")

    def list_wheels(self) -> List[dict]:
        """List all wheels."""
        return sorted(self.wheels.values(), key=lambda w: f"{w['wheelmake']} {w['wheelmodel']}")

    def get_all_upgrade_levels(self, frame_id: str, wheel_id: str) -> List[BikeSetup]:
        """Compatibility shim: game data has no upgrade stages, so all 6 are identical."""
        return [self.get_bike_stats(frame_id, wheel_id, level) for level in range(6)]


# Singleton instance
_db: Optional[BikeDatabase] = None


def get_bike_database() -> BikeDatabase:
    """Get the singleton bike database instance."""
    global _db
    if _db is None:
        _db = BikeDatabase()
    return _db


def get_bike_stats(frame_id: str, wheel_id: str, upgrade_level: int = 0) -> Optional[BikeSetup]:
    """Convenience function to get bike stats."""
    return get_bike_database().get_bike_stats(frame_id, wheel_id, upgrade_level)


if __name__ == "__main__":
    db = get_bike_database()

    print(f"=== Frames: {len(db.frames)} ===")
    for frame in db.list_frames()[:10]:
        print(f"  {frame['frameid']}: {frame['framemake']} {frame['framemodel']} "
              f"({frame['frametype']}, {frame['frameweight_g']}g, bias {frame['framecda_bias']})")

    print(f"\n=== Wheels: {len(db.wheels)} ===")
    for wheel in db.list_wheels()[:10]:
        print(f"  {wheel['wheelid']}: {wheel['wheelmake']} {wheel['wheelmodel']} "
              f"({wheel['wheelweight_g']}g, bias {wheel['wheelcda_bias']})")

    print("\n=== Example combo ===")
    fid = next(iter(db.frames))
    wid = next(iter(db.wheels))
    setup = db.get_bike_stats(fid, wid)
    if setup:
        print(f"  {setup}")
        print(f"  Frame:  {setup.frame_weight_g}g, CdA bias {setup.frame_cda_bias}")
        print(f"  Wheels: {setup.wheel_weight_g}g, CdA bias {setup.wheel_cda_bias}")
        print(f"  Total:  {setup.weight_kg:.3f} kg, CdA bias {setup.cda_bias:+.4f}")
        print(f"  cd={setup.cd:.4f}  (abs CdA @ref {setup.cd * REF_FRONTAL_AREA:.4f})")
