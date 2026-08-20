import numpy as np
from bike_comparison.bike_data import BASE_CDA, REF_FRONTAL_AREA
from bike_comparison.physics import frontal_area_from_rider
import bike_comparison.pacing_planner as pp

h_m, w_kg, bike_kg = 1.75, 75.0, 8.0
fa = frontal_area_from_rider(h_m, w_kg); cda = BASE_CDA*(fa/REF_FRONTAL_AREA)
total_mass=w_kg+bike_kg; inv_mass=1/total_mass; aero_k=0.5*1.225*cda
drive_step = (250.0*2000/30000)*(1-0.025)  # ~16.2

def mk(grd):
    cs=np.cos(np.arctan(grd)); return total_mass*9.8067*grd, 0.004*total_mass*9.8067*cs

# window: chunk0 flat run-up, chunks1..9 climb 8% at power 0
fg=[]; fr=[]; L=10.0
g0,r0=mk(0.0); fg.append(g0); fr.append(r0)
for _ in range(9):
    g,r=mk(0.08); fg.append(g); fr.append(r)
drive=[0.0]*10

def window_time(extra):
    v=15.0; total=0.0
    for c in range(10):
        d=drive[c]+(extra if c==0 else 0.0)
        v,dt=pp._traverse(v,d,fg[c],fr[c],aero_k,inv_mass,L,2); total+=dt
    return total

print("Run-up chunk (flat, entry 15) before 9 stalled 8% climb chunks:")
print("sweep run-up power, windowed gain of one more step:")
for k in range(0,60):
    g = window_time(0.0)-window_time(drive_step)
    if k%5==0:
        print(f"  runup_power={drive[0]/0.975:.0f}W gain={g:.4f}s  window_base={window_time(0.0):.2f}s")
    drive[0]+=drive_step

# Compare: gain of putting the step on climb chunk1 instead (at its stalled state)
drive=[0.0]*10
def window_time_from(idx, extra):
    v=15.0 if idx==0 else 0.1
    # simulate from chunk idx only, window of 10
    total=0.0
    for j,c in enumerate(range(idx, idx+10)):
        gg,rr=mk(0.08); d=(extra if j==0 else 0.0)
        v,dt=pp._traverse(v,d,gg,rr,aero_k,inv_mass,L,2); total+=dt
    return total
g_climb = window_time_from(1,0.0)-window_time_from(1,drive_step)
print(f"\nGain of first step on a climb chunk (entry 0.1): {g_climb:.4f}s")
