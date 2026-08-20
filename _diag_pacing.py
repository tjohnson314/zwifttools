import numpy as np
from bike_comparison.ride_simulator import list_routes, load_route_profile
from bike_comparison.bike_data import BASE_CDA, REF_FRONTAL_AREA
from bike_comparison.physics import frontal_area_from_rider
import bike_comparison.pacing_planner as pp

routes = list_routes()
cand = [r for r in routes if 8 <= r['distance_km'] <= 20 and r['ascent_m'] > 100]
cand.sort(key=lambda r: -r['ascent_m'])
r = cand[0]
route = load_route_profile(r['id'], r['name'], world=r['world'], include_leadin=True)
print("route", r['name'], r['distance_km'], "km")

h_m, w_kg, bike_kg = 1.75, 75.0, 8.0
fa = frontal_area_from_rider(h_m, w_kg)
cda = BASE_CDA * (fa / REF_FRONTAL_AREA)

length, grad, mid, end = pp._build_chunks(route, 10.0, 0.004)
n = len(length)
print("n chunks", n, "total km", end[-1]/1000)
print("grad range", grad.min()*100, grad.max()*100)
print("frac uphill>2%", np.mean(grad>0.02), "downhill<-2%", np.mean(grad<-0.02))

# Replicate init to probe initial gains
avg_power = 250.0
total_mass = w_kg + bike_kg
inv_mass = 1/total_mass
aero_k = 0.5*1.225*cda
cos_s = np.cos(np.arctan(grad))
f_grav = total_mass*9.8067*grad
f_roll = 0.004*total_mass*9.8067*cos_s
window=10; nsub=2
units = min(15*n, 30000)
delta = avg_power*n/units
drive_step = delta*(1-0.025)
print("delta", round(delta,2), "units", units)

drive = np.zeros(n)
v_entry = np.empty(n+1)
from bike_comparison.physics import speed_from_power
v_entry[0] = max(speed_from_power(avg_power, float(grad[0]), w_kg, bike_kg, cda, 0.004), pp.V_FLOOR)

# forward at zero power
v = v_entry[0]
for c in range(n):
    v,_ = pp._traverse(v, drive[c], f_grav[c], f_roll[c], aero_k, inv_mass, length[c], nsub)
    v_entry[c+1]=v

def window_time(start, extra):
    v = v_entry[start]; total=0.0; first=True
    for c in range(start, min(start+window,n)):
        d = drive[c] + (extra if first else 0.0)
        v,dt = pp._traverse(v, d, f_grav[c], f_roll[c], aero_k, inv_mass, length[c], nsub)
        total += dt; first=False
    return total

gains = np.array([window_time(c,0.0)-window_time(c,drive_step) for c in range(n)])
print("gain stats: min", gains.min(), "max", gains.max(), "mean", gains.mean())
# top 10 gain chunks: their gradient and entry speed
order = np.argsort(-gains)[:10]
print("TOP gain chunks (gain, grad%, v_entry m/s):")
for c in order:
    print(f"  c={c} gain={gains[c]:.2f} grad={grad[c]*100:.1f}% ventry={v_entry[c]:.2f}")
# correlation gain vs gradient
print("mean gain uphill>2%:", gains[grad>0.02].mean())
print("mean gain downhill<-2%:", gains[grad<-0.02].mean())
print("mean gain flat:", gains[np.abs(grad)<=0.02].mean())
# entry speed distribution
print("v_entry: min", v_entry.min(), "max", v_entry.max(), "mean", v_entry.mean())
print("mean v_entry uphill>2%:", v_entry[:-1][grad>0.02].mean())
print("mean v_entry downhill<-2%:", v_entry[:-1][grad<-0.02].mean())
