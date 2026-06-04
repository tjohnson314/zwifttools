# Race Pacer Mod

Pace against a past ride on the same route. Shows your time gap (ahead or behind) at every point on the course.

## How to Use

1. Export a reference ride CSV from ZwiftTools (see format below)
2. In Zwift, open the **Race Pacer Setup** window from the Sauce overlay
3. Click **Select CSV file** and pick your exported file
4. Start riding — the **Race Pacer** overlay shows your gap in real time

## CSV Format

The export tool must produce a file with this exact structure:

```
course_id,route_id,name,date
12,1337403830,My Watopia Race,2026-06-01
time_sec,distance_m,lat,lng
0,0.0,11.649702,-165.397491
1,12.5,11.649810,-165.397600
2,25.1,11.649920,-165.397710
...
```

### Metadata rows (lines 1–2)

| Column      | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `course_id` | integer | Zwift course/world ID (from `state.courseId`) |
| `route_id`  | integer | Zwift route ID (from `state.routeId`); 0 or blank if free-riding |
| `name`      | string  | Human-readable ride name (shown in overlay)  |
| `date`      | string  | ISO date of the ride, e.g. `2026-06-01`      |

### Data rows (line 3 = header, lines 4+ = data)

| Column       | Type  | Description                                    |
|--------------|-------|------------------------------------------------|
| `time_sec`   | float | Elapsed time from race/event start (seconds)   |
| `distance_m` | float | Cumulative distance from race/event start (metres) |
| `lat`        | float | Latitude (optional, not used by mod)           |
| `lng`        | float | Longitude (optional, not used by mod)          |

**Important:**
- `time_sec` must start at or near 0 and be monotonically increasing
- `distance_m` must be monotonically increasing  
- One row per second is recommended; higher resolution is fine
- `distance_m` should correspond to Zwift's `eventDistance` field (resets to 0 at race start)

## Gap Calculation

At any point in the current ride:

```
gap = ref_time_at(current_distance) − current_elapsed_time
```

- **Positive gap** (green) → you are **ahead** of the reference pace
- **Negative gap** (red) → you are **behind** the reference pace

The reference time at a given distance is linearly interpolated between the two nearest data points.

## Course Matching

The mod checks that `course_id` in the CSV matches the Zwift world you are currently riding in. If they don't match, the overlay shows "wrong course" instead of a gap.
