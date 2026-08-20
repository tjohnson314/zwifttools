import numpy as np
from bike_comparison.ride_simulator import list_routes, load_route_profile
from bike_comparison.bike_data import BASE_CDA, REF_FRONTAL_AREA
from bike_comparison.physics import frontal_area_from_rider
from bike_comparison.pacing_planner import plan_tt_pacing

routes = list_routes()
w_kg, h, bike = 75.0, 1.75, 8.0
fa = frontal_area_from_rider(h, w_kg)
cda = BASE_CDA * (fa / REF_FRONTAL_AREA)

def check(name):
    r = [x for x in routes if x['name'] == name][0]
    route = load_route_profile(r['id'], r['name'], world=r['world'], include_leadin=True)
    a = np.asarray(route.altitude_m, float)
    raw = np.diff(a); raw = raw[raw > 0].sum()
    res = plan_tt_pacing(route, w_kg, h, bike, cda, 250.0)
    print(f"\n=== {name} ({r['world']}) disp {route.display_distance_km}km asc(disp) {route.display_ascent_m:.0f} raw_asc {raw:.0f} ===")
    print(f"  pacing @250W: time {res.total_time_formatted}  avgspd {res.avg_speed_kph}kph  avgP {res.avg_power_w}  maxP {res.max_power_w}")

check('Road to Sky')            # Watopia big climb (was 2:09, should be ~1h)
check('Tempus Fugit')           # Watopia flat sanity
check('Ven-Top')                # France big climb (ratio ~1.0, should be unchanged/correct)
