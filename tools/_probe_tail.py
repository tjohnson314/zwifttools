import numpy as np
from bike_comparison.ride_simulator import load_route_profile

r = load_route_profile('2737483381', 'Watopia Hilly Route', world='WATOPIA', include_leadin=True)
d = np.asarray(r.distance_m, dtype=float)
a = np.asarray(r.altitude_m, dtype=float)
d = d - d[0]

# cumulative positive ascent along the path
dalt = np.diff(a)
cum_asc = np.concatenate([[0.0], np.cumsum(np.where(dalt > 0, dalt, 0.0))])

total = d[-1]
print(f'total stitched distance {total:.0f} m, total ascent {cum_asc[-1]:.1f} m')
print('\ndist_m   cum_ascent_m')
for target in [500, 3000, 6000, 8000, 8300, 8500, 8700, 9000, 9193, 9400, 9695]:
    i = int(np.argmin(np.abs(d - target)))
    print(f'{d[i]:7.0f}   {cum_asc[i]:8.1f}')

# find where cumulative ascent first reaches 108 m (climb essentially complete)
idx = int(np.argmax(cum_asc >= 108.0)) if np.any(cum_asc >= 108.0) else -1
if idx >= 0:
    print(f'\ncum ascent reaches 108 m at distance {d[idx]:.0f} m')
    print(f'flat tail after that = {total - d[idx]:.0f} m')

# where along the path is "9.0 km traveled"? what remains after it in OUR model?
i9 = int(np.argmin(np.abs(d - 9000)))
print(f'\nat 9000 m: cum ascent {cum_asc[i9]:.1f} m, remaining path {total - d[i9]:.0f} m')
