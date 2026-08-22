"""Verify option 2: geometry distances after re-extraction, for Hilly Route."""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from shared import surface_map  # noqa: E402
from bike_comparison.ride_simulator import _find_wad_route, load_route_profile  # noqa: E402

entry = _find_wad_route("Hilly Route", "WATOPIA", "2737483381")
data = surface_map.get_route(entry["mapID"], entry["nameHash"])
route_d = data["route"]["d"][-1]
lead_d = data["leadin"]["d"][-1]
print("=== stored geometry d (after re-extraction) ===")
print(f"  route   d[-1] = {route_d:8.1f} m   (metadata distance_m      = {data['distance_m']:.1f})")
print(f"  leadin  d[-1] = {lead_d:8.1f} m   (metadata leadin_distance_m= {data['leadin_distance_m']:.1f})")
print(f"  total geometry = {route_d + lead_d:8.1f} m")

for inc in (True, False):
    rp = load_route_profile("2737483381", "Hilly Route", "WATOPIA", include_leadin=inc, laps=1)
    print(f"\n=== load_route_profile(include_leadin={inc}) ===")
    print(f"  distance_m[-1]      = {rp.distance_m[-1]:8.1f} m")
    print(f"  total_distance_km   = {rp.total_distance_km:6.3f}")
    print(f"  display_distance_km = {rp.display_distance_km:6.3f}")
    print(f"  total_ascent_m      = {rp.total_ascent_m:6.1f}")
    print(f"  display_ascent_m    = {rp.display_ascent_m:6.1f}")
