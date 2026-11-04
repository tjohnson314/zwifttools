# Race Pacer Mod

Pace against a reference ride on the same course: a past race, or a ZwiftTools TT
pacing plan. The overlay shows your time gap (ahead or behind) at your current
distance, the reference power and speed at that point, and, for bucketed TT plans,
your average power in the current bucket against its target.

## Getting a reference CSV

- **Race Replay:** load a race and click **↓** on a rider's row. This exports their
  real race telemetry (`race_pacer_<name>_<date>.csv`).
- **TT Pacing:** build a plan, choose the smart-bucket count, and click
  **↓ Export for Sauce** (`tt_plan_<route>.csv`). It exports the plan as shown on
  screen: times come from the planned speed, and power is the target power. With
  buckets, the power is constant within each bucket, and a row sits exactly at each
  bucket boundary.

## How to use

1. Copy this folder into Sauce's mods folder (`Documents/SauceMods`; Sauce's
   Settings → Mods shows the exact path), restart Sauce and enable the mod.
2. Add a **Race Pacer** window.
3. Click **Load reference ride CSV** and pick your file.
4. Ride. **Reset** restarts timing from your current position. Outside events, use it
   at the start line. **Change** loads another file.

## What the overlay shows

```
        WATOPIA                 route (Race Replay slug) or world name
         +12s                   time gap
        AHEAD
   275W · 3.82 W/kg             reference / target power at your distance
      34.6 km/h                 reference speed at your distance
  avg 268 W / 275 W (−7)        your average in this bucket vs target
  BUCKET 2/4 · 1.4 KM LEFT      bucket progress
      Plan name                 ride name
```

- **Gap:** green (ahead), red (behind) or blue (even). See below for how it's calculated.
- **Reference power** is held from the last row at or before your distance, not
  interpolated, so bucket targets change cleanly. W/kg needs `weight_kg`.
- **Reference speed** is interpolated.
- **Bucket average** appears only when the CSV has a `bucket` column, which means
  bucketed TT plans. It's your power over the current bucket, weighted by time.
  - It restarts at each bucket and when you press **Reset**.
  - Readings without power, and gaps over 5 s (pauses), are not counted.
  - Colour: green within ±max(2%, 3 W) of the target, red when under, amber when over.
- **Wrong course:** if the CSV's world doesn't match the world you're riding, the gap
  is replaced by `wrong course — plan is <world>`.

### Hiding details

The gear button in the titlebar opens **Race Pacer Settings**. Each line can be
switched off, and changes apply live:

| Setting | Hides |
|---|---|
| Target / reference power | power line |
| Reference speed | speed line |
| Bucket average vs target | `avg … W / … W` line |
| Bucket number and distance left | `BUCKET n/m · … KM LEFT` line |
| Route and ride name | top and bottom name lines |

The gap and its ahead/behind label are always shown. Hidden lines keep updating, so
the bucket average is correct when you turn it back on.

## CSV format

```
course_id,route_id,name,date,weight_kg
1,1337403830,Tempus Fugit,2026-11-04,72.0
time_sec,distance_m,lat,lng,power_watts,speed_kmh,bucket
0.000,0.000,,,295,41.20,1
6.112,70.000,-11.636,166.972,295,41.30,1
...
```

### Metadata (lines 1–2)

| Column | Description |
|---|---|
| `course_id` | Zwift **world ID** (1 Watopia, 2 Richmond, 3 London, 4 New York, 5 Innsbruck, 6 Bologna, 7 Yorkshire, 8 Crit City, 9 Makuri Islands, 10 France, 11 Paris, 12 Gravel Mountain, 13 Scotland). The overlay converts it to Sauce's `courseId`, which uses different numbers, before comparing. Blank turns off the course check. |
| `route_id` | Route slug (Race Replay, e.g. `tempus-fugit`, shown title-cased) or numeric route ID (TT plans; the world name is shown instead). Optional. |
| `name` | Shown at the bottom of the overlay. |
| `date` | ISO date. Informational. |
| `weight_kg` | Reference rider weight, used for W/kg. Optional. |

### Data (line 3 = header, lines 4+ = rows)

| Column | Required | Description |
|---|---|---|
| `time_sec` | yes | Seconds from the start, increasing. |
| `distance_m` | yes | Metres from the start, increasing. Shifted so the first row is 0. |
| `lat`, `lng` | no | Not used by the overlay. |
| `power_watts` | no | Reference or target power. |
| `speed_kmh` | no | Reference speed. |
| `bucket` | no | 1-based bucket number. Turns on the bucket average. |

Columns are matched by header name, so their order doesn't matter. Rows with a blank
`time_sec` or `distance_m` (telemetry gaps) are skipped; any other non-numeric value
rejects the file.

## Gap calculation

```
gap = reference time at your current distance − your elapsed time
```

The reference time is interpolated between the two nearest rows. A positive gap means
you're ahead.

- **In events:** elapsed time and distance come from Zwift (`state.time`, and
  `routeDistance` falling back to `eventDistance`).
- **Outside events:** the overlay starts its own clock and distance the first time you
  move, or when you press **Reset**.

The overlay follows Sauce's watched athlete, so it also works while spectating.

## Files

| File | Purpose |
|---|---|
| `display.html` / `.css` / `.mjs` | The overlay window |
| `display-settings.html` / `.mjs` | Settings window (detail toggles) |
| `reference.mjs` | CSV parsing, world/course mapping, route label |
| `bucket-tracker.mjs` | Bucket spans, running average, pace status |
| `details.mjs` | Detail-line defaults and hide classes |

The `.mjs` modules other than `display*.mjs` have no Sauce imports and are tested in
`tests/race_pacer_*.test.cjs`.
