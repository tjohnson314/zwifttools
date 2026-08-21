import json, glob

out = []
targets = {'hilly route', 'flat route', 'volcano circuit', 'watopia flat route',
           'tempus fugit', '5k loop', 'greater london flat'}
for fp in glob.glob('zwift_routes/world_*.json'):
    with open(fp, encoding='utf-8') as f:
        data = json.load(f)
    for r in data.get('routes', []):
        if str(r.get('name', '')).lower() not in targets:
            continue
        leg = r.get('route') or {}
        xs = leg.get('x', []); zs = leg.get('z', [])
        d = leg.get('d', []); alt = leg.get('alt', [])
        rec = {
            'name': r.get('name'),
            'world': data.get('world'),
            'supported_laps': r.get('supported_laps'),
            'header_distance_m': r.get('distance_m'),
            'geom_d_last_m': d[-1] if d else None,
            'geom_npts': len(xs),
        }
        if len(xs) >= 2:
            gap = ((xs[0]-xs[-1])**2 + (zs[0]-zs[-1])**2) ** 0.5
            rec['xy_gap_m'] = round(gap, 1)
            rec['alt_start'] = round(alt[0], 2) if alt else None
            rec['alt_end'] = round(alt[-1], 2) if alt else None
            rec['alt_step_at_seam_m'] = round(alt[0]-alt[-1], 2) if alt else None
            rec['start_xz'] = [round(xs[0],1), round(zs[0],1)]
            rec['end_xz'] = [round(xs[-1],1), round(zs[-1],1)]
        out.append(rec)

with open('_loopdiag_out.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
