"""Survey: summed checkpoint geometry vs Zwift's reported distance, Watopia."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from shared import surface_map  # noqa: E402


def leg_geom_m(leg):
    if not leg or not leg.get("x"):
        return 0.0
    xs, zs = leg["x"], leg["z"]
    return sum(math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1])
               for i in range(1, len(xs)))


# Resolve Watopia's map id.
wat_id = next(mid for mid, nm in surface_map.WORLD_NAMES.items() if nm == "Watopia")
world = surface_map._load_route_world(wat_id)
routes = world.get("routes", [])

rows = []
for r in routes:
    g_route = leg_geom_m(r.get("route"))
    g_lead = leg_geom_m(r.get("leadin"))
    m_route = float(r.get("distance_m") or 0.0)
    m_lead = float(r.get("leadin_distance_m") or 0.0)
    if m_route <= 0 or g_route <= 0:
        continue
    rows.append((r.get("name", "?"), g_route, m_route, g_route / m_route,
                 g_lead, m_lead, (g_lead / m_lead) if m_lead else float("nan")))

rows.sort(key=lambda x: x[3])
print(f"Watopia (map {wat_id}): {len(rows)} routes with geometry\n")
hdr = f"{'route':40s} {'geom_m':>9s} {'zwift_m':>9s} {'ratio':>6s}  {'lead_geom':>9s} {'lead_zwift':>10s} {'l_ratio':>7s}"
print(hdr)
print("-" * len(hdr))
for nm, gr, mr, rat, gl, ml, lrat in rows:
    print(f"{nm[:40]:40s} {gr:9.1f} {mr:9.1f} {rat:6.3f}  {gl:9.1f} {ml:10.1f} {lrat:7.3f}")

ratios = [x[3] for x in rows]
print(f"\nroute ratio  geom/zwift: min {min(ratios):.3f}  max {max(ratios):.3f}  "
      f"mean {sum(ratios)/len(ratios):.3f}")
lratios = [x[6] for x in rows if x[6] == x[6] and x[5] > 0]
if lratios:
    print(f"leadin ratio geom/zwift: min {min(lratios):.3f}  max {max(lratios):.3f}  "
          f"mean {sum(lratios)/len(lratios):.3f}")
