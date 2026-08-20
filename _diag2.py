import heapq
import numpy as np
from bike_comparison.ride_simulator import list_routes, load_route_profile
from bike_comparison.bike_data import BASE_CDA, REF_FRONTAL_AREA
from bike_comparison.physics import frontal_area_from_rider, speed_from_power
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
n = len(length)
avg_power = 250.0
total_mass = w_kg + bike_kg
inv_mass = 1/total_mass
aero_k = 0.5*1.225*cda
cos_s = np.cos(np.arctan(grad))
f_grav = total_mass*9.8067*grad
f_roll = 0.004*total_mass*9.8067*cos_s
window=10; nsub=2
units = min(15*n, 30000); units=max(units,n)
delta = avg_power*n/units
drive_step = delta*(1-0.025)

power = np.zeros(n); drive = np.zeros(n); v_entry = np.empty(n+1)
v_entry[0] = max(speed_from_power(avg_power, float(grad[0]), w_kg, bike_kg, cda, 0.004), pp.V_FLOOR)

def forward(start,count,v0):
    v=v0; endc=min(start+count,n)
    for c in range(start,endc):
        v,_=pp._traverse(v,drive[c],f_grav[c],f_roll[c],aero_k,inv_mass,length[c],nsub)
        v_entry[c+1]=v
    return v
forward(0,n,v_entry[0])

def window_time(start,extra):
    v=v_entry[start]; total=0.0; first=True
    for c in range(start,min(start+window,n)):
        d=drive[c]+(extra if first else 0.0)
        v,dt=pp._traverse(v,d,f_grav[c],f_roll[c],aero_k,inv_mass,length[c],nsub)
        total+=dt; first=False
    return total
def gain(c):
    return window_time(c,0.0)-window_time(c,drive_step)

last_modified=np.zeros(n,dtype=np.int64); step=0; heap=[]
for c in range(n):
    heapq.heappush(heap,(-gain(c),c,step))
placed=0
uphill_units=0; downhill_units=0; flat_units=0
while placed<units and heap:
    neg_g,c,g_step=heapq.heappop(heap)
    if g_step<last_modified[c]:
        heapq.heappush(heap,(-gain(c),c,step)); continue
    power[c]+=delta; drive[c]+=drive_step; placed+=1; step+=1
    if grad[c]>0.02: uphill_units+=1
    elif grad[c]<-0.02: downhill_units+=1
    else: flat_units+=1
    forward(c,window,v_entry[c])
    lo=max(0,c-window+1); hi=min(n,c+window+1); last_modified[lo:hi]=step
    heapq.heappush(heap,(-gain(c),c,step))

print("placed", placed, "of", units)
print("units by class -> uphill:", uphill_units, "downhill:", downhill_units, "flat:", flat_units)
print("mean power uphill>2%:", power[grad>0.02].mean())
print("mean power downhill<-2%:", power[grad<-0.02].mean())
print("mean power flat:", power[np.abs(grad)<=0.02].mean())
print("max power", power.max(), "at grad", grad[power.argmax()]*100)
# final forward for time
v=v_entry[0]; T=0.0
for c in range(n):
    v,dt=pp._traverse(v,drive[c],f_grav[c],f_roll[c],aero_k,inv_mass,length[c],nsub); T+=dt
print("total time h:", T/3600, "avg kmh", end[-1]/1000/(T/3600))
