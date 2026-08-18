import logging
logging.basicConfig(level=logging.WARNING)
from pathlib import Path
from race_replay.data_cleaner import clean_race_data

for rid in ['race_data_6924708', 'race_data_7028414']:
    print('=== ', rid)
    try:
        rd = clean_race_data(Path('race_data')/rid, cache=False)
        riders = sorted(rd.riders, key=lambda r: r.rank)
        rows = [(r.rank, r.name[:18], round(r.finish_time_sec,1) if r.finish_time_sec is not None else None,
                 round(float(r.ttt_time_offset),1) if r.ttt_time_offset is not None else None) for r in riders]
        for x in rows[:12]:
            print('  rank=%s name=%-18s finish=%s offset=%s' % x)
        fins = [r.finish_time_sec for r in riders if r.finish_time_sec is not None]
        print('  finish_count=%d min=%.1f max=%.1f' % (len(fins), min(fins), max(fins)))
    except Exception as e:
        import traceback; traceback.print_exc()
