"""
Bike data module for ZwifterBikes equipment stats.

Provides bike weight and Cd values for frame/wheel combinations at different upgrade levels.
Data sourced from https://zwifterbikes.web.app/
"""

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BikeSetup:
    """Complete bike setup with computed stats."""
    frame_id: str
    frame_name: str
    wheel_id: str
    wheel_name: str
    upgrade_level: int
    cd: float  # Drag coefficient Cd (NOT CdA - must multiply by frontal area)
    weight_kg: float  # Bike weight in kg
    
    def __str__(self):
        return f"{self.frame_name} + {self.wheel_name} (Level {self.upgrade_level})"


class BikeDatabase:
    """Database of Zwift bike frames, wheels, and their combined stats."""
    
    DATA_DIR = Path(__file__).parent / 'zwiftdata'
    URLS = {
        'bikes': 'https://zwifterbikes.web.app/assets/bikes.json',
        'frames': 'https://zwifterbikes.web.app/assets/frames.json',
        'wheels': 'https://zwifterbikes.web.app/assets/wheels.json'
    }
    
    def __init__(self):
        self.frames: Dict[str, dict] = {}
        self.wheels: Dict[str, dict] = {}
        self.bikes: Dict[Tuple[str, str], dict] = {}  # (frame_id, wheel_id) -> stats
        self._load_data()
    
    def _load_data(self):
        """Load bike data from local cache or download from ZwifterBikes."""
        for name in ['frames', 'wheels', 'bikes']:
            file_path = self.DATA_DIR / f'{name}.json'
            
            if not file_path.exists():
                self._download_data()
                break
        
        # Load frames
        with open(self.DATA_DIR / 'frames.json') as f:
            frames_list = json.load(f)
            self.frames = {fr['frameid']: fr for fr in frames_list}
        
        # Load wheels
        with open(self.DATA_DIR / 'wheels.json') as f:
            wheels_list = json.load(f)
            self.wheels = {wh['wheelid']: wh for wh in wheels_list}
        
        # Load bike combos
        with open(self.DATA_DIR / 'bikes.json') as f:
            bikes_list = json.load(f)
            self.bikes = {(b['frameid'], b['wheelid']): b for b in bikes_list}
    
    def _download_data(self):
        """Download fresh data from ZwifterBikes."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        for name, url in self.URLS.items():
            with urllib.request.urlopen(url) as resp:
                content = resp.read().decode('utf-8-sig')  # Handle BOM
                data = json.loads(content)
            
            with open(self.DATA_DIR / f'{name}.json', 'w') as f:
                json.dump(data, f, indent=2)
    
    def get_bike_stats(self, frame_id: str, wheel_id: Optional[str] = None, upgrade_level: int = 0) -> Optional[BikeSetup]:
        """
        Get complete bike stats for a frame/wheel combination.
        
        Args:
            frame_id: Frame ID (e.g., 'F115' for Canyon Aeroad 2024)
            wheel_id: Wheel ID (e.g., 'W057' for Enve SES 4.5 Pro).
                      Use None or '' for special bikes like Tron that have built-in wheels.
            upgrade_level: Upgrade level 0-5
            
        Returns:
            BikeSetup with cd and weight, or None if not found
        """
        if upgrade_level < 0 or upgrade_level > 5:
            raise ValueError("Upgrade level must be 0-5")
        
        # Handle None or empty string for built-in wheel bikes (like Tron)
        lookup_wheel_id = wheel_id if wheel_id else ''
        
        combo = self.bikes.get((frame_id, lookup_wheel_id))
        if not combo:
            return None
        
        frame = self.frames.get(frame_id)
        
        # For bikes with built-in wheels (wheel_id is empty string)
        if lookup_wheel_id == '':
            return BikeSetup(
                frame_id=frame_id,
                frame_name=f"{frame['framemake']} {frame['framemodel']}" if frame else frame_id,
                wheel_id='',
                wheel_name='(Built-in)',
                upgrade_level=upgrade_level,
                cd=combo['cd'][upgrade_level],  # Drag coefficient (not CdA!)
                weight_kg=combo['weight'][upgrade_level]
            )
        
        wheel = self.wheels.get(wheel_id)
        
        if not frame or not wheel:
            return None
        
        return BikeSetup(
            frame_id=frame_id,
            frame_name=f"{frame['framemake']} {frame['framemodel']}",
            wheel_id=wheel_id,
            wheel_name=f"{wheel['wheelmake']} {wheel['wheelmodel']}",
            upgrade_level=upgrade_level,
            cd=combo['cd'][upgrade_level],  # Drag coefficient (not CdA!)
            weight_kg=combo['weight'][upgrade_level]
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
        """Get stats for all upgrade levels of a bike combo."""
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
    # Demo usage
    db = get_bike_database()
    
    print("=== Available Frames (sample) ===")
    for frame in db.list_frames()[:10]:
        print(f"  {frame['frameid']}: {frame['framemake']} {frame['framemodel']}")
    
    print(f"\n... and {len(db.frames) - 10} more frames")
    
    print("\n=== Available Wheels ===")
    for wheel in db.list_wheels():
        print(f"  {wheel['wheelid']}: {wheel['wheelmake']} {wheel['wheelmodel']}")
    
    print("\n=== Example: Canyon Aeroad 2024 + Enve SES 4.5 Pro ===")
    setup = get_bike_stats('F115', 'W057', upgrade_level=0)
    if setup:
        print(f"  {setup}")
        print(f"  Cd: {setup.cd:.4f}")
        print(f"  Weight: {setup.weight_kg:.3f} kg")
    
    print("\n=== All Upgrade Levels ===")
    for level, setup in enumerate(db.get_all_upgrade_levels('F115', 'W057')):
        print(f"  Level {level}: Cd={setup.cd:.4f}, Weight={setup.weight_kg:.3f} kg")
