import time
from bike_comparison.ride_simulator import list_routes, load_route_profile
from bike_comparison.bike_data import BASE_CDA, REF_FRONTAL_AREA
from bike_comparison.physics import frontal_area_from_rider
from bike_comparison.pacing_planner import plan_tt_pacing

routes = list_routes()
print(f"{len(routes)} routes available")
# Pick a moderate hilly route (8-20 km) for a representative test.
cand = [r for r in routes if 8 <= r['distance_km'] <= 20 and r['ascent_m'] > 100]
cand.sort(key=lambda r: -r['ascent_m'])
r = cand[0] if cand else routes[0]
print("Testing route:", r['name'], r['world'], r['distance_km'], "km", r['ascent_m'], "m")

route = load_route_profile(r['id'], r['name'], world=r['world'], include_leadin=True)

h_m, w_kg = 1.75, 75.0
fa = frontal_area_from_rider(h_m, w_kg)
cda = (BASE_CDA + 0.0) * (fa / REF_FRONTAL_AREA)

t0 = time.time()
res = plan_tt_pacing(route, w_kg, h_m, 8.0, cda, 250.0)
dt = time.time() - t0

print(f"planning took {dt:.2f}s")
print("total time:", res.total_time_formatted, "avg speed", res.avg_speed_kph)
print("avg power", res.avg_power_w, "min", res.min_power_w, "max", res.max_power_w)
print("n sections:", len(res.sections))
# Show correlation: higher power on steeper sections
import statistics
climbs = [s for s in res.sections if s['avg_gradient_pct'] > 2]
descents = [s for s in res.sections if s['avg_gradient_pct'] < -2]
if climbs:
    print("avg power on climbs(>2%):", round(statistics.mean(s['power_w'] for s in climbs)))
if descents:
    print("avg power on descents(<-2%):", round(statistics.mean(s['power_w'] for s in descents)))
print("sample sections:")
for s in res.sections[:6]:
    print(s)
