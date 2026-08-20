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
h_m, w_kg, bike_kg = 1.75, 75.0, 8.0
fa = frontal_area_from_rider(h_m, w_kg)
cda = BASE_CDA * (fa / REF_FRONTAL_AREA)
length, grad, mid, end = pp._build_chunks(route, 10.0, 0.004)
total_mass=w_kg+bike_kg; inv_mass=1/total_mass; aero_k=0.5*1.225*cda
cos_s=np.cos(np.arctan(grad)); f_grav=total_mass*9.8067*grad; f_roll=0.004*total_mass*9.8067*cos_s
drive_step = (250.0*len(length)/min(15*len(length),30000))*(1-0.025)

# Isolated single chunk: descent -3.6%, entry speed 15 m/s. Sweep own power.
grd=-0.036; L=10.0; cs=np.cos(np.arctan(grd)); fg=total_mass*9.8067*grd; fr=0.004*total_mass*9.8067*cs
v_entry=15.0
print("Isolated single-chunk gain vs accumulated power (descent -3.6%, entry 15 m/s):")
drv=0.0
for k in range(0,40):
    # gain of adding one more step
    _,t0=pp._traverse(v_entry, drv, fg, fr, aero_k, inv_mass, L, 2)
    _,t1=pp._traverse(v_entry, drv+drive_step, fg, fr, aero_k, inv_mass, L, 2)
    g=t0-t1
    if k%5==0:
        print(f"  power={drv/0.975:.0f}W gain={g:.4f}s t0={t0:.3f}")
    drv+=drive_step
print()
# Now the same but with entry speed also allowed to grow? No, entry fixed. Check climb chunk.
grd=0.08; cs=np.cos(np.arctan(grd)); fg=total_mass*9.8067*grd; fr=0.004*total_mass*9.8067*cs
v_entry=0.1
print("Isolated single-chunk gain vs power (climb 8%, entry 0.1 m/s):")
drv=0.0
for k in range(0,40):
    _,t0=pp._traverse(v_entry, drv, fg, fr, aero_k, inv_mass, L, 2)
    _,t1=pp._traverse(v_entry, drv+drive_step, fg, fr, aero_k, inv_mass, L, 2)
    g=t0-t1
    if k%5==0:
        print(f"  power={drv/0.975:.0f}W gain={g:.4f}s t0={t0:.3f}")
    drv+=drive_step
