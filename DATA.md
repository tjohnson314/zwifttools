# Data Pipeline Documentation

This document describes the complete data pipeline for every page and API endpoint in the Zwift Tools web app.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Authentication](#authentication)
- [Landing Page](#landing-page)
- [Surface Map](#surface-map)
- [My Activities](#my-activities)
- [Workout Analysis](#workout-analysis)
- [Bike Comparison](#bike-comparison)
- [Race Replay](#race-replay)
- [Ride Compare](#ride-compare)
- [TTT Analysis](#ttt-analysis)
- [Shared Infrastructure](#shared-infrastructure)

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Browser UI  │◄───►│  Flask App   │◄───►│  Zwift REST API   │
│  (Jinja2 +   │     │  (app.py)    │     │  us-or-rly101     │
│   JS/CSS)    │     │              │     │  .zwift.com/api    │
└──────────────┘     └──────┬───────┘     └───────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌──────────┐  ┌────────────┐  ┌──────────────┐
      │  Local   │  │   Azure    │  │  ZwiftMap     │
      │  Disk    │  │   Blob     │  │  (route data) │
      │ race_data│  │  Storage   │  │  zwiftmap.com │
      └──────────┘  └────────────┘  └──────────────┘
```

All data originates from the **Zwift REST API** (`https://us-or-rly101.zwift.com/api`). Route geometry comes from **ZwiftMap** (`zwiftmap.com`). Processed race data is cached to **local disk** (parquet + JSON) and optionally persisted to **Azure Blob Storage**.

---

## Authentication

### `GET /auth/login`

Renders the login form (`login.html`).

### `POST /auth/login`

**Flow:** Email + password → `get_token_with_password()` → Zwift OAuth token pair → stored in Flask `session['tokens']` as a `ZwiftTokens` dict (access_token, refresh_token, expires_at).

### `GET /auth/callback`

OAuth code exchange flow (CSRF state check → `exchange_code_for_tokens()`). Stores tokens in session.

### `GET /auth/logout`

Clears `session['tokens']`.

### `GET /auth/status`

Returns `{ authenticated: bool, method: 'oauth', expires_at: int }`.

**Token refresh:** `get_session_tokens()` checks `expires_at` on every authenticated request and calls `refresh_access_token()` if expired. All Zwift API calls use `get_headers()` which returns `{ Authorization: 'Bearer <access_token>' }`.

---

## Landing Page

### `GET /`

Renders `landing.html`. Passes `logged_in = 'tokens' in session` to the template so it can show login/logout links. No data pipeline — purely static.

---

## Surface Map

### Road Splines

The main map's surface road layer and the local-only road inspector sample
cubic Bezier road geometry. When a `ROADNODE` omits its tangent handles, both
derive uniform Catmull-Rom
handles from neighboring nodes instead of drawing straight chords. Looped
roads use wrapped neighbors; open roads keep their endpoint positions.
Explicit handles (including zero) and `m_bStraight` segments are preserved.
The dev overview, selected road, node segments, and spawn road anchors share
this geometry. The main map's generated surface exports use the same handles,
resampled at the existing 8 m spacing. Raw XML, route exports, and GPS calibration
are unchanged. Regenerate the surface exports with
`python -m tools.extract_zwift_surfaces`.

### Road-to-Route Selection

The main map snaps the click to a native road and matches routes at that position,
not elsewhere on the same road. Per-road checkpoint paths use a small fixed
geometry tolerance that does not grow when zooming out. A point with no known
traversing routes leaves the view unchanged; one match loads that route; multiple
matches open a searchable route
picker. The picker includes all matching routes independently of sidebar filters,
supports keyboard dismissal, and keeps loading/error feedback beside the choice.
Dragging still pans rather than selecting.

Surface segments retain `road_id`. Route versions retain `road_ids` for the course
and that version's native approach, plus per-road `road_paths` derived from native
checkpoint geometry. World API summaries project these paths by `race` / `free_ride`
mode and expose `road_match_tolerance`. Free-ride paths preserve the interpolated
spawn-range midpoint. Inferred paddock fallback roads are not treated as observed
traversals. Legacy video routes use their existing road membership and route paths.

### Dev Timing-Arch Inspector

The local-only `/surface-map-dev` page has a **Timing Arches** tab and a
toggleable start/finish overlay. Search by name, exact hash, or road ID;
select a list entry or marker to inspect its directional definition, raw
road-time values, declared length, coordinates, and serialized source entity
XML. A direct link can specify `?world=9&arch=44010445334`.

Definitions come from `ENTITY_TYPE_TIMINGARCH` entries in installed world
`entities.xml`. Positions interpolate the narrowest bracketing native route
checkpoint interval on the same road, with the source route exposed for each
boundary. They are not independent in-game observations. Missing coverage
remains explicitly unlocated rather than clamped or guessed; hashes remain
strings to preserve 64-bit precision, and declared lengths are not corrected.
Shared F/R finishes are drawn once without a direction suffix; start markers
remain directional. The inspector itself remains read-only.

### Dev Spawn Markers

The local-only map also has a **Spawn Markers** tab, searchable by source name,
label, or road ID, and a toggleable blue marker/name overlay. Selecting a
marker shows its raw road time, position source, and entity XML, with a link
to inspect its associated road. A direct link can specify
`?world=1&spawn=AdZ_Paddock`.

Spawn entities come from `ENTITY_TYPE_SPAWNMARKER` in installed `entities.xml`.
Where native checkpoints bracket their road time, the same interpolation as
the timing inspector supplies their position. Otherwise a hollow marker and
name label use the road geometry midpoint, explicitly identified as a **road
anchor**, not an exact spawn position. Missing roads stay unlocated. Raw names
and road times are preserved; no paddock IDs or race start lines are inferred.
`AdZ_Paddock` belongs to road 130 near the Alpe summit, not Road to Sky's start.

### Route Start Versions

The route browser offers **Free Ride** and **Race**, preserved in shared links
with `mode=free_ride` or `mode=race`. The route API accepts the same parameter;
omitting it preserves the previous display behavior for other consumers.

Race mode restores the raw native lead-in, including recorded paddock sections
that the legacy display can omit. Free Ride uses the midpoint of the source
`spawn_area` road-time range, followed along native checkpoints to the main
route start. A same-world reference path supplies missing approach coverage
only if it reaches the same road/time start within 2 m. These representative
spawn positions are not observations of an individual rider's actual spawn.
Declared mode distances stay separate from measured geometry; no rescaling
or plausible-range clipping is applied. Course segment distances are rebased
to the selected measured start offset; race-only approach segments are omitted
from free-ride views. Missing start geometry is explicitly indicated, and
routes without a free-ride definition disable that option.

All 389 numeric-world definitions retain their existing course and raw lead-in
data. There are 307 available free-ride definitions, with native approach
coverage for 300; seven have unlocated approach geometry. Race metadata also
records the longest sampled road in the associated paddock group as an
inferred fallback, breaking exact length ties by lower road ID. It does not
replace a native reference approach or invent unverified paddock connections.

Refresh only start metadata and profiles without regenerating existing routes:

```powershell
python -m tools.extract_zwift_routes --refresh-route-versions
```

An optional `--source-cache-dir` accepts trusted local per-world WAD pickle
snapshots. Normal extraction also produces the mode metadata and profiles.

### Route Sampling

Regenerated virtual-route profiles use `--max-points` (default 600) as an
initial sampling target, not a hard cap. Additional samples preserve segment
boundaries, surface transitions, and bends or elevation changes until the
three-dimensional interpolation error against original checkpoints is at most
1 m before coordinate rounding. Published distances and full-resolution
cumulative distances are not rescaled or recomputed from simplified chords.

Segment boundaries are inserted into both physical-distance profiles and
`segment_path`. The latter's `d` array stores original checkpoint-index
positions, which are nonuniform after refinement; consumers must interpolate
through these values rather than treating array indices as route percentages.
All 389 cached virtual routes across 13 worlds have been refreshed with this
sampling. Video routes already retain every original checkpoint and are unchanged.

The route profile's main leg uses the measured `route_offset_m`, including
the native lead-in/main connector and subtracting any omitted paddock prefix.
Elevation traces, hover/click positions, and the lead-in marker share this
axis with timing-derived segment distances. Published `leadin_distance_m`
remains metadata, not a profile offset; absent lead-in geometry adds no offset.

Segment names retain the source label for direction `F`. Direction `R` appends
`Reverse` unless the source label already contains that word. Direction comes
from the timing entity's explicit F/R hash fields, not the hash sign; hashes,
segment boundaries, and geometry are unchanged by this naming rule.
These are XML road directions, not guaranteed public forward/reverse names:
Alley Sprint's XML `F` traversal corresponds to the publicly named reverse
segment. Its raw timing-to-timing lengths are about 386.5 m (`F`) and 414.6 m
(`R`), consistent with the user-checked Strava figures of 0.38 km reverse and
0.41 km forward. The entity's declared 482.8020036 m for both directions is
retained separately rather than used to stretch the geometry.

### Sprint List Accuracy

[Zwift Insider's sprint list](https://zwiftinsider.com/sprints/) mostly uses
in-game values and explicitly warns that some are incorrect. Its public
forward/reverse labels must be matched to route associations, not assumed
to follow XML `F`/`R`. A length discrepancy is a review finding, not permission
to rescale geometry or overwrite source declarations.

Alley Sprint's listed 480 m disagrees with the approximately 414.6 m raw
passage and the user-checked 0.41 km Strava segment. Alley Sprint Reverse's
listed 380 m is consistent with the approximately 386.5 m raw passage and
0.38 km Strava display. The source entity declares 482.8020036 m for both.

The complete 71-entry audit is recorded in
[`race_data/zwift_sprint_accuracy_audit.csv`](race_data/zwift_sprint_accuracy_audit.csv).
It compares the public list and all 71 linked detail pages with complete
timing-to-timing passages in the cached native route data. World, source names,
and route associations determine identity, never distance agreement. All 71
entries have distinct directional hashes. Bell Lap's Prime is distinguished
from the Crit City lap entity by its friendly name; Manhattan Reverse and
Monument Ave Reverse are paired with the same-road entity of their uniquely
matched opposite public direction, using different hashes.

A review flag requires a median difference greater than both 10 m and 5% of
the listed length. Eleven entries meet that heuristic; the remaining 60 are
within it, not certified accurate. The CSV retains the full observed minimum,
maximum, median, uncorrected declared distance, signed difference, public
gradient, matching evidence, and correction notes for every entry. Hashes
must be imported as text in spreadsheet software to retain 64-bit precision.

| Public Sprint | Listed Length (m) | Native Passage Median (m) | Assessment |
| --- | ---: | ---: | --- |
| Alley Sprint | 480 | 414.56 | Discrepancy corroborated by user-checked Strava 0.41 km |
| Castle Park Sprint | 220 | 207.09 | Approximate published estimate; review, not a proven error |
| Innsbruck Sprint | 300 | 275.12 | Geometry-based review candidate |
| Innsbruck Sprint Reverse | 200 | 154.06 | Geometry-based review candidate |
| Lutece Reverse Sprint | 220 | 231.51 | Geometry-based review candidate |
| Railway Sprint | 640 | 491.20 | Geometry-based review candidate |
| Shisa Sprint Reverse | 280 | 265.02 | Geometry-based review candidate |
| Sprinteur Sprint Reverse | 250 | 234.72 | Geometry-based review candidate |
| Sunset Sprint | 4000 | 257.73 | Strong discrepancy; source entity declares 250 m |
| Sunset Sprint Reverse | 4000 | 257.57 | Strong discrepancy; source entity declares 250 m |
| Tower Sprint Reverse | 320 | 287.36 | Geometry-based review candidate |

Both Sunset detail pages also state 4 km, so this is not just a table-parsing
error. Repeated route measurements are consistent, but they share native
source geometry: they do not independently validate in-game line placement.
Except for the user's Alley Strava check, these measurements are not a
separate rider-odometer or Strava distance audit.

The detail articles explicitly document these **in-game errors already
corrected in the master list**:

- Alley Sprint Reverse: 480 m in-game versus about 380 m actual.
- Castle Park Sprint: 320 m in-game versus an estimated 220 m actual.
- Castle Park Sprint Reverse: 320 m in-game versus an estimated 210 m actual.
- Sprinteur Sprint Reverse: -1.8% in-game versus a flat 0% actual gradient.

Castle Park's articles expressly say their lengths are approximate because
Strava's minimum segment length prevents precise measurement. Other listed
gradients were captured but not independently audited. No source declarations,
raw lengths, or route geometry were changed by this audit.

### KOM List Accuracy

[Zwift Insider's KOM list](https://zwiftinsider.com/koms/) contains 49 entries.
Its master table omits the metric length for Sgurr Summit South, showing only
0.6 miles, and leaves length, elevation gain, and gradient blank for The Hill
KOM Reverse. These are missing-data findings, not numerical comparisons;
detail-page values and native measurements must remain separate sources.
As with sprints, public direction names are not necessarily XML `F`/`R`.

The full 49-entry audit, including published elevation and gradient fields,
is in
[`race_data/zwift_kom_accuracy_audit.csv`](race_data/zwift_kom_accuracy_audit.csv).
World, public/in-game name families, and unique route-name overlap identify
48 distinct native hashes without using distance to select a match. Mostly
Ventoux remains unmatched. Of the 48 matched entries, 44 are within the sprint
audit's review threshold, two have length discrepancies, and two lack a listed
metric length. Within the threshold does not mean independently verified.

| Public KOM | Listed Length (m) | Native Passage Median (m) | Native Range (m) | Declared Length (m) |
| --- | ---: | ---: | --- | ---: |
| Castle KOM | 2500 | 2688.04 | 2688.00-2688.05 | 2494.48 |
| 23rd St. KOM | 260 | 282.64 | 282.60-282.65 | 275.00 |

Both differences exceed 10 m and 5% of the listed length. Their detail pages
repeat the master values, with no explicit correction note. These are
**geometry-based review candidates**, not independently proven public-list
errors; repeated native measurements share the same underlying source.
Full-precision measurements and declarations remain in the CSV.

The incomplete or unverified cases are kept separate:

- **Sgurr Summit South:** the detail page supplies 1 km, consistent with the
   993.20 m native median. The missing master metric value remains blank in
   the CSV rather than being filled from the detail page or rounded miles.
- **The Hill KOM Reverse:** both master and detail omit length, elevation,
   and gradient; the detail still says more information is coming on
   October 27, 2025. Its raw native passages measure 11.89-11.92 m, median
   11.90 m, with a declared 9.9898148 m. These surprisingly short values are
   preserved, not treated as independently verified replacement figures.
- **Mostly Ventoux:** no matching complete cached native passage was found.
   The article describes a hidden start at marker #2, distinct from the full
   Ventoux KOM start, and the same summit finish. The live France timing-arch
   lookup timed out; the listed 17.1 km remains unverified, not marked wrong
   or compared to the full Ventoux measurement.

Apart from Sgurr Summit South's missing metric length, the extracted numeric
values agree between the master and detail pages, including missing fields.
Published elevation gain and gradients were captured and cross-checked
between pages, not independently measured. Article remarks about steeper
uphill portions do not contradict averages over a full segment containing
flat or downhill sections. No raw geometry or declared lengths were changed.
Import the CSV's hashes as text to preserve exact 64-bit identifiers.

### Master Route Accuracy

The complete [Zwift Insider master route list](https://zwiftinsider.com/routes/)
audit visits all 339 linked detail pages, including running and event-only
routes. Both modern Leaderboard Segments and older Sprint & KQOM Segments
tables are parsed. Six verified native name/identifier aliases produce 337
cached route matches; the two older Gravel Mountain routes remain uncached
and are not conflated with modern Red Rock Loop definitions.

The full comparison is in
[race_data/zwift_route_accuracy_audit.csv](race_data/zwift_route_accuracy_audit.csv),
with every significant row tabulated in
[race_data/zwift_route_accuracy_audit.md](race_data/zwift_route_accuracy_audit.md).
The reproducible reader is
[tools/audit_zwiftinsider_routes.py](tools/audit_zwiftinsider_routes.py).
Run `python tools/audit_zwiftinsider_routes.py` to rebuild from its local
snapshot, or add `--collect --limit 0` to fetch all remaining detail pages.
Sources, exact segment identities, structured results and cache checksums
are stored under `race_data/zwift_route_master_audit_sources/`.

Review thresholds are greater than 100 m for route length and 50 m for
lead-in length, also requiring a difference outside the published metric
rounding interval; an explicit zero/no lead-in is exact. Header distances and
measured planar geometry are separate;
lead-in geometry includes its final connector and subtracts only an existing
display paddock prefix. Missing geometry remains blank, never zero. On 53
pages with no detail lead-in value, the master's explicit zero is retained
as a separately identified reference rather than invented detail metadata.

The snapshot has 276 rows with at least one review flag, **not 276 proven
incorrect routes**: 63 have header-distance flags, 86 measured-distance flags,
118 header lead-in flags, 165 measured lead-in flags, 90 direction-label flags,
27 missing displayed-segment sets and 37 extra sets. Categories overlap.
Event pens, free-ride starts and public main-route versus lead-in scope can
explain some differences. Four public segment identities remain explicitly
unresolved; missing native passages are not silently inferred or extended.
Segment matching reuses verified sprint/KOM hashes and source-name/route
associations, never distance. Existing partial-passage display filtering is
distinguished from missing extraction, and lead-in-only extras are labeled.

The largest numeric cases include Red Rock Loop variants (published 2.6 km,
cached headers about 5.183 km and measured profiles about 5.171 km), That's
Amore Reverse (7.4 km versus 5.829 km header / 5.554 km geometry), and Epic
Run's lead-in (master zero versus 4.121 km header / 4.234 km geometry). These
are comparison findings, not independent in-game accuracy certification.
Jungle Loop's public labels remain inverted relative to the cache's XML-derived
labels, while traversal and hash identity remain correct. No route geometry,
metadata, names or app behavior are changed by this audit.

### Segment Percentages

Virtual-world segment boundaries now come from directional timing arches,
not native route percentages. The extractor finds start and finish road-time
crossings on original same-road, same-direction checkpoint edges and
interpolates their exact checkpoint positions and physical distances. Ordinary
passages require both crossings. Partial traversals, unmatched directions, and
missing boundary coverage do not produce a complete passage. Road wraps and
lead-in/main joins are not interpolated as ordinary edges, but exact native
endpoints remain usable, including a finish at a road wrap. Exactly closed,
lap-enabled main paths may establish cyclic completion as described below.

Literal-crossing passages record `boundary_source: "timing_arch"`, `start_checkpoint`,
`end_checkpoint`, direction, road ID, source road times, and the uncorrected
`declared_distance_m`. Display geometry prefers explicit checkpoint positions;
sampling preserves these interpolated boundaries. Declared length is not
used to reject or resize an otherwise complete passage. Derived
`percent_start` and `percent_end` remain as compatibility fields only.
Original native entries are preserved separately in `native_segments`.
The existing named leaderboard-definition filter is retained; the dev
inspector still exposes unnamed or incomplete timing entities as raw data.

#### Near-Coincident Lap Arches

Near-coincident start/finish fields are treated as a lap only when the source
marks the entity as a lap arch (`m_telemetryArchType: 4`, a LapArch model, or
`m_bStopwatchRequiresAllCheckpoints: 1`). Proximity is a circular road-time
difference of at most 0.0001, not a minimum physical segment length. Nearby
non-lap fields retain ordinary segment matching.

A lap starts at an observed arch crossing, must subsequently cross its
directional `m_beginTimerSplineTime`, and finishes on a later return to the
arch. Without a timer field, a direction-compatible road wrap arms the lap.
Either near-coincident source boundary can identify the observed arch; raw
`start_time`, `end_time`, and declared distance are retained alongside
`effective_start_time`, `effective_end_time`, `is_loop`,
`begin_timer_time`, and `lap_armed_checkpoint`. A different road or direction
interrupts the matching sequence; it is not bridged into an inferred lap.

If no literal return crossing exists, an exactly closed, lap-enabled main
path can establish completion: its first and last checkpoints have identical
coordinates, road, direction, and road time; every main edge matches the lap's
road and direction; its first edge crosses the arch and a later edge arms the
lap. These passages record `boundary_source: "closed_route_lap"` and use the
full measured native main extent for their profile span. The interpolated
arch is retained separately as `timing_arch_checkpoint` and
`timing_arch_distance_m`; the map overlay is rotated around that checkpoint
so both pins remain on the actual timing arch. No route coordinates are
changed or extrapolated, and declared distance does not determine lap length.
This handles cycling Volcano routes whose closed native lap starts just
before the arch, whereas the running variants start exactly on it.

The installed-source audit across all 13 cached worlds is recorded in
[`race_data/zwift_loop_accuracy_audit.csv`](race_data/zwift_loop_accuracy_audit.csv).
All 40 near-coincident named definitions were checked: 36 laps and four
non-lap reverse markers (Itza KOM, Mayan Mountainside KOM, Rooftop KOM, and
Railway Sprint). Of the laps, 27 have complete measured passages, totaling
130 exported occurrences: 122 literal crossings and eight closed native
laps. Nine lack a complete matching sequence; this
does not certify that the corresponding leaderboard or course is invalid.
London Loop and Central Park Loop have original native associations but
unresolved timing coverage. Missing
start/finish samples and interrupted road coverage are not repaired with
percentage estimates, extrapolation, or geometry stretching.

The initial refresh touched 120 routes in seven world caches. Hilly Loop Reverse's
false 0.08-0.09 m passages, including Volcano Circuit Run, are removed;
complete passages on Hilly Route Reverse and its running variant measure
9165.79-9165.88 m. The closed-lap follow-up adds eight passages in five world
caches, including Volcano Circuit (4095.31 m) and Volcano Circuit CCW
(4095.29 m). These measured values are not forced to the source declarations. Non-lap
segment entries, original native associations, route metadata, index, video,
and altitude-scale files are unchanged by these refreshes. The follow-up also
preserves every native profile and pre-existing measured passage, while
refreshing per-hash maxima to include the new observations. Import audit hashes
as text to preserve their exact 64-bit values; missing distances are blank.

XML `percentStart` and `percentEnd` describe positions in the concatenated
lead-in and main-route high-resolution checkpoint sequence. They are not
fractions of physical distance or of an individual road's `time` parameter.
Verified native examples encode checkpoint index divided by checkpoint count;
compatibility percentages derived from timing crossings use checkpoint
position divided by count minus one. Legacy sources without supplied timing
definitions retain the previous percentage interpretation.

For that legacy path, a lap-enabled route with an exactly closed main path and a positive segment
start followed by `percent_end: 0` resolves to the final checkpoint. The raw
percentage is retained, with additive `resolved_percent_end: 1`; calculated
end distance and map geometry use the resolved endpoint. Other zero values
are left unchanged. Timing-derived passages instead use their actual crossings.

Each exported segment also records `distance_m` (end minus start distance,
rounded to 0.01 m) and `max_distance_m` (the longest observed passage with that
exact hash across the cached routes). Every extraction refreshes these maxima
across the output directory, including after a single-world refresh. Forward
and reverse hashes remain separate; names are not used to group passages.

Surface Map and Race Replay omit segment overlays whose recorded distance is
more than 25 m below that maximum. Shortfalls up to and including 25 m remain
visible, compared at the recorded 0.01 m precision. This is a display heuristic,
not proof of leaderboard eligibility. Raw segment entries, names, percentages,
and route geometry remain available unchanged in the exports. Legacy entries
without the distance metadata retain their previous display behavior.

### Route Paddocks

`tools/extract_zwift_routes.py` retains `event_paddocks` (the ordered numeric
references from `eventPaddocks`) and `alternate_event_ramp` (the numeric
`useAlternateEventRamp` value, or null when absent) in each virtual route and
the route index. An empty paddock list means the route header declares none;
it does not imply the route cannot be used for events.

Each virtual-world route file also contains `paddock_roads`, extracted from
roads explicitly marked `roadIsPaddock` in the client's road XML. Each record
retains its `road_id`, name, optional `paddock_exit_road_time`, and ordered
`nodes` with local-metre `x`, `alt`, `z`, `tangent_in`, and `tangent_out`.
Missing spline handles remain null. These are source spline control nodes,
not an interpolated elevation profile; altitude retains the raw game datum.
Associated `spawn_markers` retain their source names and native `road_time`.
A spawn marker is not assumed to be a race start line.

Route paddock references and road IDs are separate namespaces. The extraction
does not guess their mapping from proximity or ordering, and does not prepend
pen roads to the existing lead-in checkpoints. For example, Elevation
Evaluation references paddocks 21–25 and alternate-ramp mode 2, while its dense
lead-in checkpoints still measure 2,599.65 m versus the 3,089.79 m header.

Routes optionally contain `leadin_view` when removing a marked paddock prefix
reduces the absolute lead-in discrepancy from strictly **over 50 m** to strictly
**under 10 m**. The comparison uses the original high-resolution planar
checkpoint intervals plus the final gap to the main-route start. The interval
from a paddock checkpoint to the first ordinary-road checkpoint belongs to the
paddock. No non-prefix sections are joined, and no distances are stretched.

`leadin_view.profile` contains the remaining checkpoints, rebased to distance
zero before the usual resampling. Its `full_distance_m`, `distance_m`,
`paddock_distance_m`, `error_before_m`, and `error_after_m` retain the unrounded
measurements; `paddock_road_ids` and `excluded_checkpoint_count` identify the
omitted prefix. The raw `leadin`, published distances, main route, and index
remain unchanged. The Surface Map uses this optional profile, preserves the
original altitude anchor, and shifts checkpoint-based segment distances and
lead-in escalators by the omitted distance. All other routes keep their view.

Run `python -m tools.analyze_route_paddock_leadins` to regenerate the complete
analysis and `race_data/paddock_leadin_adjustments.md`, which lists every
qualifying route with its before/after distances and absolute discrepancies.

#### All-World Paddock Groups

[tools/analyze_paddock_groups.py](tools/analyze_paddock_groups.py) independently
audits physical paddock-road groups without changing the extraction or app.
Run `python -m tools.analyze_paddock_groups` against the installed world WADs;
`--source-cache-dir` optionally reuses trusted per-world pickle snapshots from
an earlier local WAD scan. The default spatial grouping threshold is 100 m.

The current audit covers all 389 definitions across 13 worlds and all 170
marked paddock roads, producing 33 groups. Every repeated or overlapping
world-local paddock ID matches one nearest group: 383 routes declare IDs,
197 have directly observed paddock-road starts with IDs, and none has an
inconsistent ID/group assignment. The other matches remain spatial inference.

Groups use connected components of nearby road control points. Route matching
uses the first original lead-in checkpoint, falling back to coarse lead-in
or main start only when necessary; distances are planar point-to-control-
polyline distances, not travelled road-network lengths or sampled Bezier
curves. Animated TT/BOOST gate models provide additional road evidence.
Scotland's two previously verified complexes are reproduced. This associates
IDs with groups, not individual IDs with specific roads or spawn lines.

The full group and exception table is in
[race_data/paddock_group_analysis.md](race_data/paddock_group_analysis.md).
[race_data/paddock_groups.csv](race_data/paddock_groups.csv) retains every
physical group; [race_data/paddock_group_routes.csv](race_data/paddock_group_routes.csv)
retains all 389 route matches, exact distances, source restrictions and caveats.
[race_data/paddock_group_analysis.json](race_data/paddock_group_analysis.json)
also retains per-ID reference counts, gate models, source hashes, cache
checksums and 50/100/150/200 m sensitivity partitions.

Six route headers omit IDs: five Watopia Two Bridges/Beach Island/Ocean Lava
variants and Yorkshire's Harrogate Circuit Reverse. Watopia road 107, Watopia
roads 129/130/131 (AdZ_Paddock), and France roads 115/116/117 have no route-ID
references; no missing IDs are guessed. Watopia road 101 is separate from
roads 7-12 at 100 m but merges with them at 150 m, so those two memberships
are threshold-sensitive. All other group memberships are stable across the
tested thresholds. Richmond, Bologna, France and Watopia's special 19/20/116
references have nearest-only ID evidence. Seventy-seven definitions lack
lead-in checkpoints; Mayan 8, Mayan Mash, Tides and Temples, and Navig8 have
main-start fallback anchors over 500 m from their nearest group. These are
coverage caveats, not inconsistent mappings. No route cache fields change.

### Page: `GET /surface-map`

The world selector includes **Video Routes**, a surface-map-only group with
ID `1000`. Its data is published in `zwift_routes/video_routes.json` and
`zwift_surfaces/video_routes.json`. These checked-in files work on production
without a local Zwift install. The existing worlds, routes, and their IDs are
unchanged; video routes retain their original IDs and source metadata.

Video routes come from the downloaded Mochi cache, not the virtual-world WADs.
Refresh the published files on a machine with downloaded video-route data:

```powershell
python tools/extract_zwift_routes.py --video-cache "$env:LOCALAPPDATA\Zwift\mochi"
```

The main `tools/regenerate_zwift_data.py` pipeline also refreshes video routes
and their road network from `%LOCALAPPDATA%\Zwift\mochi`, honoring both
`--out-routes` and `--out-surfaces`. Use `--video-cache PATH` to select another
cache. A missing default cache is reported and skipped, leaving existing video
exports untouched; an explicitly supplied missing directory is an error.
Video extraction shares the normal client-version check, so use `--force` when
only the downloaded cache has changed. No video assets are downloaded by this
pipeline; errors extracting an available cache stop regeneration.

The extractor retains every high-resolution route checkpoint, converts game
centimetres to metres, and joins named surface markers by road ID and road time.
Native road nodes retain latitude, longitude, elevation, video timestamps, and
video distance. No physical-range clamping or virtual-world altitude scaling is
applied. Readable route labels are separate from the original `source_name`.

The existing `/api/surface_map/worlds`, `/api/surface_map/world/1000`, and
`/api/surface_map/route/1000/<route_id>` endpoints serve the new group. Route
detail includes native GPS/video data in `road_nodes`. The canvas uses the
client's local route geometry without a virtual-world background image.
Restart Flask after refreshing data to clear the cached world files.

---

## My Activities

### Page: `GET /my-activities`

Renders `my_activities.html`. Redirects to login if not authenticated.

### `GET /api/my_activities`

**Purpose:** Fetch the logged-in user's recent Zwift activities.

**Pipeline:**

1. **Get profile ID** — `GET /api/profiles/me` → extract `id`
2. **Fetch activities** — `GET /api/profiles/{id}/activities?start={start}&limit={limit}`
3. **Transform each activity:**
   - `distanceInMeters` → `distance_km` (÷ 1000)
   - `movingTimeInMs` → `duration_sec` (÷ 1000)
   - Extract `sport`, `name`, `avgWatts`, `calories`, `startDate`
   - Initialize enrichment fields as null: `route_name`, `avg_hr`, `normalized_power`, `is_race`, `race_name`, `event_subgroup_id`
4. **Return** `{ activities: [...], start, limit }`

### `GET /api/activity_enrichment/<activity_id>`

**Purpose:** Enrich a single activity with route name, NP, HR, and race info. Called lazily by the frontend for each visible activity.

**Pipeline:**

1. **Activity details** — `get_activity_details(activity_id, headers)` → `GET /api/activities/{id}`
   - Extract `routeId`, `eventInfo` (subgroup ID, event ID, race name)
   - If no `routeId` but has `eventId`: fetch `GET /api/events/{eventId}` → scan `eventSubgroups` for matching subgroup → get `routeId`
2. **Route lookup** — `get_route_info(routeId)` from the WAD route index (`zwift_routes/index.json`) → `route_name`
3. **Telemetry** — `fetch_rider_telemetry(activity_id, headers)`:
   - `GET /api/activities/{id}` → extract `fitnessData` URL
   - Fetch raw telemetry JSON (contains `powerInWatts`, `heartRateInBpm`, etc.)
   - `convert_telemetry_to_dataframe()` → DataFrame with `power_watts`, `hr_bpm`, etc.
4. **Compute:**
   - **Normalized Power** — `calculate_normalized_power(power_watts)`: 30-sec rolling avg → 4th power → mean → 4th root
   - **Average HR** — Mean of non-zero `hr_bpm` values
5. **Return** `{ route_name, normalized_power, avg_hr, is_race, race_name, event_subgroup_id, event_id }`

---

## Workout Analysis

### Page: `GET /workout-analysis`

Requires login and preserves `?activity_id=` across the login redirect. Linked
from the homepage and each row in My Activities.

### Catalog: `GET /api/workouts` and `GET /api/workouts/<workout_id>`

`tools/extract_zwift_workouts.py` reads the installed client's
`assets/Workouts/workouts.wad` with the existing WAD reader. It exports cycling
workout names, source paths, numeric segments, and interpretation warnings to
`zwiftdata/workouts.json`. Coaching text is not exported. Stable IDs are hashes
of source paths. Running/swimming plans and untimed rest days are excluded.
Distance-based dynamic workouts are also excluded rather than misinterpreted
as elapsed-time plans. The checked-in catalog works without a local game installation.

The index returns names, IDs, source paths, and durations. Detail adds timed
segments with FTP-relative targets. Repeats are expanded; warmup/cooldown/ramp
endpoints remain directional. Interval low/high values are target ranges.
Free-ride, max-effort, and unresolved zone-only blocks have null targets.
Quantized ramps are shown as continuous approximations, with a warning.
An explicit `ftpOverride` is retained as `ftp_override`; fixed-reference FTP
tests scale by that reference rather than the rider's profile FTP. The page
disables the FTP field for these plans and explains the fixed reference.
Missing or invalid catalog data returns an explicit 503 error.

### Custom plans: `POST /api/workouts/upload`

Requires a logged-in session. Accepts a multipart `file` containing a `.zwo`
workout, at most 1 MiB, and returns the same plan shape as catalog details.
Uses the shared `parse_workout()` parser, so repeats, ramps, ranges, fixed FTP
references, and unscored blocks behave identically. Missing sport type defaults
to cycling; other sports, distance-based plans, untimed/empty plans, malformed
XML, and unknown or invalid blocks return explicit 400 errors. Oversized uploads
return 413. DOCTYPE declarations are rejected and expanded plans are limited to
10,000 blocks to bound parsing work. Total duration must remain finite.

Uploaded files and parsed plans are not saved, added to the catalog, or sent to
Zwift. The browser keeps the parsed custom plan in memory for this page only.
Switching sources preserves the selected built-in plan and the last successfully
loaded custom plan, with stale requests prevented from changing the active plan.
Both sources share the same FTP/intensity/offset controls, nearest-5-W target
rounding, raw-power chart, and coverage statistics.

### Recorded power: `GET /api/workout_activity/<activity_id>`

Requires authentication. Uses `fetch_rider_telemetry()` and returns raw
`timeInSec` / `powerInWatts` as `time_sec` / `power_watts`, plus activity name/ID.
No cooldown trimming, smoothing, or power clipping is applied. Null power and
timestamp gaps are retained. Mismatched arrays, non-finite values, and
non-increasing timestamps return 422; upstream failures return 502.

### FTP and comparison

`GET /api/my_profile` exposes finite positive `ftp` as `ftp_w` (otherwise null),
alongside the existing name/height/weight fields. The page prefills this current
profile value without overwriting manual edits. This is not historical FTP.
Missing FTP or lookup failures prompt manual entry without blocking analysis.

The browser scales targets by editable FTP and whole-workout intensity, then
aligns the schedule using an editable elapsed-time offset. Statistics weight
each recorded power sample until the next timestamp, capped at the median
positive recording interval (lower median for an even count; one second for a
single sample). Gaps and nulls are excluded, not interpreted as zero.
Targets are rounded to the nearest 5 W after FTP/intensity scaling (half-way
values round toward the higher target). Ramps step at rounding thresholds.
Target integration uses the exact same covered time, splitting at those
thresholds; target ranges use the midpoint of their rounded bounds for averages
and signed errors. Time/range averages need not be multiples of 5 W. The chart
shows the full raw activity and the planned bounds, including unscored gaps.
There is no automatic workout matching, pause/skip reconstruction, or
mid-workout intensity reconstruction.

---

## Bike Comparison

### Page: `GET /bike-comparison`

Renders `bike_comparison.html`.

### `GET /api/frames`

**Source:** `BikeDatabase` loaded from `zwiftdata/frames.json` (originally sourced from ZwifterBikes).

**Transform per frame:**
- `frameid` → `id`
- `framemake + framemodel` → `name`
- `frameaero` → `aero`, `frameweight` → `weight`
- `framewheeltype == 'fixed'` → `hasBuiltInWheels`
- `frametype` → `isTT` (bool) + `frameType` (string)
- `framelevel` → `level`

### `GET /api/wheels`

**Source:** `zwiftdata/wheels.json` via `BikeDatabase`.

**Transform per wheel:**
- `wheelid` → `id`
- `wheelmake + wheelmodel` → `name`
- `wheelaero`, `wheelweight`, `wheelfitsframe`, `wheellevel`
- Prepends a synthetic `(Built-in wheels)` entry for fixed-wheel frames

### `GET /api/my_profile`

**Source:** `GET /api/profiles/me`

**Transform:**
- `height` (mm) → `height_cm` (÷ 10)
- `weight` (g) → `weight_kg` (÷ 1000)
- `firstName + lastName` → `name`
- Finite positive `ftp` → `ftp_w`; missing/invalid FTP → null (editable in Workout Analysis)

### `GET /api/bike_stats`

**Params:** `frame_id`, `wheel_id`, `upgrade_level`

**Source:** `BikeDatabase.get_bike_stats()` → looks up `(frame_id, wheel_id)` in pre-computed combo table from `zwiftdata/bikes.json`. Each combo has arrays of 6 Cd and weight values (upgrade levels 0–5).

**Returns:** `{ frame_id, frame_name, wheel_id, wheel_name, upgrade_level, cd, weight_kg }`

### `POST /api/fetch_activity`

**Purpose:** Fetch full telemetry for an activity to use in bike comparison.

**Pipeline:**

1. **Fetch telemetry** — same as enrichment: `fetch_rider_telemetry()` → `convert_telemetry_to_dataframe()`
2. **Cooldown detection** — `detect_cooldown_start(df)`:
   - Compute race average power (excluding near-zero)
   - Threshold = 40% of average power
   - 5-second rolling average smoothing
   - Scan backwards: find last block of ≥10 consecutive seconds above threshold
   - Return the index where sustained power drops off
3. **Gradient calculation:**
   - `delta_alt / delta_dist` (altitude change over distance change)
   - 5-point uniform filter smoothing
   - Clamped to ±25%
4. **Surface type computation** (GPS-based):
   - `detect_world_from_coords()` → determine Zwift world from mid-ride GPS
   - `compute_surface_types_array(lats, lngs, world)` → ray-cast each GPS point against surface polygons from `surface_data.json`
   - `surface_types_to_crr(surface_types, bike_type)` → convert to rolling resistance coefficients (Tarmac=0.004, Cobbles=0.0065, Dirt=0.025, etc.)
5. **Rider profile** — fetch `GET /api/profiles/{profileId}` for height/weight
6. **Return** `{ telemetry: { time_sec, distance_km, speed_mps, gradient, power_watts, altitude_m, surface_type?, crr? }, cooldown_start_sec, rider_profile?, ... }`

### `POST /api/upload_fit`

**Purpose:** Parse a `.fit file` upload as an alternative to fetching from Zwift API.

**Pipeline:**

1. **Decode FIT file** — `garmin-fit-sdk` decoder with scale/offset/datetime conversion
2. **Auto-decompress** gzip if detected (magic bytes `\x1f\x8b`)
3. **Extract records** — iterate `record_mesgs`, extract power, HR, cadence, speed, distance, altitude, GPS
4. **GPS conversion** — FIT semicircles → degrees (`× 180 / 2^31`)
5. **Same processing as fetch_activity:** cooldown detection, gradient calc, surface types
6. **Return** same telemetry structure

### `POST /api/compare`

**Purpose:** Compare actual vs. hypothetical bike setup using loaded telemetry.

**Pipeline:**

1. **Resolve bike setups** — `get_bike_stats()` for both actual and alternative
2. **Frontal area** — `frontal_area_from_rider(height_m, weight_kg)`, matched to Zwift's speed-test data: `FA = 0.3449 × (H/1.83)^0.66 × (M/75)^0.44` (anchored at the 183 cm / 75 kg reference rider)
3. **Surface-aware CRR** — if `surface_type` column present, compute per-bike-type CRR arrays (road bike vs MTB vs gravel have different rolling resistance on the same surface)
4. **Physics comparison** — `compare_bike_setups()`:
   - For each timestep, compute the power needed on the **alternative** bike to maintain the **same speed** as recorded
   - **With recorded power (energy-balance method):**
     - Split recorded power into: rolling resistance, aerodynamic drag, gravitational force, and a "draft" component (difference between recorded and theoretically required solo power)
     - Apply mass difference to kinetic/potential energy terms
     - Apply Cd difference to aero drag (preserving estimated draft fraction)
   - **Power equation:** `P = (F_gravity + F_rolling + F_aero) × v / (1 - η)` with η = 2.5% drivetrain loss
   - Compute NP for both actual and alternative power series
5. **Return** `{ actual: {name, cd, weight}, alternative: {name, cd, weight}, summary: {total_kj, diff_kj, diff_pct, avg_watts_diff, np_diff}, chart_data: {time_min, distance_km, actual_watts, alternative_watts, watts_difference, draft_watts, gradient} }`

### `POST /api/combo_count`

**Purpose:** Quick count of how many bike combos match the current filters.

**Filters applied by `_filter_bike_combos()`:**
- **Exclude TT frames** — filter by `frametype != 'TT'`
- **Exclude special unlocks** — frames with price=0 (except defaults) or Halo tier; wheels with price=0 (except defaults)
- **Excluded frames list** — explicit blacklist
- **Max rider level** — filter by `framelevel` and `wheellevel` ≤ rider level
- **Pareto frontier** — sort by (Cd, weight), keep only non-dominated combos where lower Cd also means lower weight

### `POST /api/best_bikes`

**Purpose:** Find the top N bike setups that minimize Normalized Power for the loaded telemetry.

**Pipeline:**

1. **Filter combos** — same `_filter_bike_combos()` as above
2. **Always include actual bike** — even if not on Pareto frontier
3. **For the actual bike:** use recorded NP directly (no simulation)
4. **For each alternative combo:**
   - Same physics pipeline as `/api/compare` — `compare_bike_setups()` with surface-aware CRR
   - Record alternative NP, avg power, total kJ
5. **Cost calculation** — `_get_combo_drops_cost()`:
   - Frame price + wheel price + upgrade cost per tier
   - Tier costs: Entry-Level (0–400k), Mid-Range (0–750k), High-End (0–1.9M), Halo (0–10M)
   - Flag `non_shop` for special-acquisition items
6. **Sort by NP** ascending, return top N
7. **Return** `{ best_bikes: [{name, cd, weight_kg, np, avg_power, total_kj, drops_cost, non_shop, is_actual}], combos_searched, actual_np, actual_bike_name }`

---

## Race Replay

### Page: `GET /race-replay`

Renders `race_replay.html`.

### `GET /api/race/fetch_stream`

**Purpose:** Fetch all race data from Zwift API given an activity URL or ID, streaming progress via **Server-Sent Events (SSE)**.

**Params:** `activity_id` (required, URL or ID), `all_subgroups` (optional, `1` for all categories), `force_refresh` (optional, `1` to re-download individual activities too)

1. **Extract activity ID** from URL or raw ID
2. **Get activity details** — `GET /api/activities/{id}` → extract `eventInfo.eventSubGroupId`
3. **Index cached individual activities** in `race_data/race_data_{subgroup_id}/` by activity ID, not rank
4. **Check blob storage** — download if unavailable locally, using it only to reuse individual activity data
5. **Fetch event subgroup** — `GET /api/events/subgroups/{subgroupId}` → race name, start time, route ID, segment distance
6. **Always fetch fresh race results** — `get_race_entries(subgroup_id, headers)` (also for each category in all-subgroup mode):
   - Paginated `GET /api/race-results/entries?event_subgroup_id={id}&start={n}&limit=50`
   - Extract rank, name, activity_id, weight, player_id per participant
   - `deduplicate_ranks()` to fix duplicate rank numbers
7. **Fetch all rider telemetry** — 5 concurrent threads, each calls:
   - Reuse successful cached activity data when available; retry missing or previously failed downloads
   - Use current results for rider membership, rank, and profile information; regenerate rank-keyed files if rank changed
   - `fetch_rider_telemetry(activity_id, headers)` → raw telemetry JSON
   - `convert_telemetry_to_dataframe(telem_data)` → DataFrame
   - `save_rider_data()` → writes `rank{n}_{activity_id}.csv` + `rank{n}_{activity_id}_raw.json`
8. **Save metadata:**
   - `complete_race_summary.csv` — rank, activity_id, name, team, weight, player_id, start_date
   - `race_meta.json` — race_name, start_time, event_id, subgroup_id, route_id, segment_distance
9. **Upload to blob storage** — `upload_race_dir()`
10. **Return** `{ race_id: 'race_data_{subgroup_id}' }` (as final SSE message)

Uses a background thread + `queue.Queue` to bridge `progress_callback` → SSE generator. Progress messages: `{ progress: true, current, total, name }`. Final message: `{ success, race_id }` or `{ error }`.

### `GET /api/race/list`

**Purpose:** List all available races (local + blob storage).

**Pipeline:**

1. **In-memory cache** — returns cached result if < 120 seconds old
2. **Scan local `race_data/`** — for each `race_data_*` directory:
   - Read `race_meta.json` → `race_name`
   - Count lines in `complete_race_summary.csv` → `rider_count` (line count - 1 for header)
3. **Merge blob storage** — `blob_storage.list_races()` for any races not found locally
4. **Cache result** for 120 seconds
5. **Return** `{ races: [{ race_id, race_name, rider_count, has_summary }] }`

### `POST /api/race/load`

**Purpose:** Load and clean a race data directory, returning metadata

Authenticated loads refresh the participant list from Zwift using the saved source
activity, while reusing successful individual activity downloads. The UI skips this
extra refresh immediately after a successful `fetch_stream` call. Logged-out users
can still open the saved replay snapshot without API access. Refresh failures are
reported rather than silently showing an old roster. Cleaned caches are invalidated
after fetching; old rider files outside the current successful results are ignored.

**Pipeline:**

1. **Check local disk** → if missing, download from blob storage
2. **`clean_race_data(data_path, cache=True)`** — the main cleaning pipeline (see below)
3. **Store in memory** — `_race_data_cache[race_id] = race_data`
4. **Upload cleaned cache to blob** — `upload_race_dir()`
5. **Return** `{ race_id, route_name, route_slug, finish_line_km, rider_count, min_time, max_time, riders: [{ rank, name, team, finish_time_sec }] }`

---

### The `clean_race_data()` Pipeline

This is the core data processing pipeline that transforms raw per-rider telemetry into a unified, time-aligned race dataset.

#### Step 1: Check Cache

If `cleaned_cache.json` exists, `load_from_cache()` reads it and restores all `RiderData` + elevation from parquet files. Returns immediately on cache hit.

#### Step 2: Load Summary

Read `complete_race_summary.csv` for rider metadata: name, team, weight_kg, player_id, activity_start_time.

#### Step 3: Load All Riders

For each `rank*_*.csv` file:
- Find matching `_raw.json`
- Parse rank from filename
- `load_raw_rider_data(csv_path, json_path)`:
  1. `pd.read_csv()` for telemetry
  2. `json.load()` for raw Zwift API response (contains `latlng`, `timeInSec`, `speedInCmPerSec`, `distanceInCm`)
  3. `fix_timestamp_offsets()` — detects server clock jumps in `time_sec` where telemetry was actually continuous (checks distance & speed continuity), corrects by subtracting excess time
  4. Interpolate `lat`/`lng` from JSON timestamps onto CSV timestamps via `scipy.interp1d(fill_value='extrapolate')`
- Attach summary metadata (name, team, weight, player_id, activity_start_time)

#### Step 4: Detect Route

`detect_route()` uses a three-method cascade:
1. **API route_id** — `route_id` from `race_meta.json` → `get_route_info()` from the WAD route index (`zwift_routes/index.json`)
2. **Name matching** — substring match race title against all known route names (sorted longest-first to avoid partial matches)
3. **GPS + distance matching** — `detect_world_from_coords()` → filter routes by world → compare median of top-10 riders' max distance against each route's total distance → pick closest within 3 km tolerance

#### Step 5: Detect World

`detect_world_from_coords()` — bounding-box lookup across 12 Zwift worlds using mid-ride GPS coordinates of the reference rider (the one with the most data points).

#### Step 6: Align Distances

**If the WAD main lap and ZwiftMap route are available** → `align_riders_to_game_route()` (preferred path):

- `load_game_route_profile()` resolves `race_meta.json.route_id` directly against the WAD route `nameHash` in `zwift_routes/index.json`
- Each WAD profile contains distance, altitude, and local-world `x/z` coordinates for the lead-in and main lap
- Lead-in checkpoints can omit the pen approach. Anchor the last checkpoint at the declared lead-in distance, preserving checkpoint spacing rather than stretching it. GPS projection, altitude calibration, and elevation profiles use this same axis, avoiding a distance jump at the main-lap join.
- `calibrate_game_route()` fits the WAD main lap to the existing ZwiftMap main-route data:
   - affine `x/z → lat/lng` transform
   - linear raw-altitude → exported-altitude transform
   - reject the calibration and use the old path if horizontal p95 error exceeds 50m
- When WAD lead-in checkpoints exist, the fitted transform is also applied to them, producing a complete GPS/elevation path from the pen through the finish

**Full-course projection:**

- Combine `[lead-in distance] + [leadin_distance + lap distance]` on an absolute race axis; if pen checkpoints are missing, the known geometry begins after 0
- Query the complete path's `cKDTree` for K=12 nearest route points
- KD-tree uses latitude-scaled coordinates: `[lat, lng × cos(median_lat)]`
- Repeat only main-lap candidates for multi-lap races; lead-in candidates can occur only once
- Candidate score = `GPS error × 10 + odometer error`, which disambiguates overlaps and intersections
- Estimate every rider's spatial course offset from the first 30 moving samples close to calibrated geometry
- Subtract that offset for on-time riders, so categories released from different pens each begin at race distance 0
- Keep true late joiners on the absolute course axis rather than rebasing them to 0
- While an alternate lead-in is off the known geometry, use raw Zwift distance without GPS correction; correction begins after it joins the route
- Smooth with the Zwift distance-delta approach:
- Uses Zwift's raw distance data (centimeter precision from `distanceInCm`)
- Walking forward: `predicted = smooth[i-1] + raw_delta`
- `smooth[i] = predicted + α × (route_distances[i] - predicted)` with **α = 0.05**
- Enforce non-decreasing derived course distance when GPS correction switches between nearby branches
- Write `distance_km` and `route_deviation_m` back to each rider

**If the WAD route has no lead-in checkpoints:**

- For on-time riders, use raw Zwift distance through the known lead-in distance, even if the pen approach overlaps a later pass of the main lap
- Only estimate the pen offset and apply GPS corrections after that lead-in; then rebase on-time riders to 0
- True late joiners can still match directly to their starting position on the main lap
- Retain the previous telemetry-altitude fallback because no authoritative lead-in elevation exists

**If WAD/ZwiftMap calibration fails** → `align_riders_to_route()` retains the previous heuristic alignment.

**If no route data** → fallback to `align_rider_distances()`:
- Pick reference rider (most data points)
- `find_course_landmarks()` — detect turn apexes via bearing angle change (sliding window)
- For each other rider, find passage time at each landmark
- Compute per-landmark distance offset vs. reference
- `scipy.interp1d` to interpolate offset across full route
- Apply offsets to correct `distance_km`

#### Step 7: Determine Finish Line

Finish-distance cascade:
1. `segment_distance_cm` from `race_meta.json` (÷ 100000 → km)
2. If segment distance is absent on a calibrated loop, `infer_loop_finish_distance()`:
   - Search near each rider's official `elapsed_ms` for a GPS pass through the lap finish
   - Convert the aligned crossing distance to a lap count
   - Use the modal lap count across riders and snap to `leadin + laps × lap_distance`
3. `route_id` → `get_total_race_distance()` from the WAD route index (`zwift_routes/index.json`)
4. `subgroupResults[0].segmentDistanceInCentimeters` from any rider's raw JSON
5. Median of top 5 ranked riders' maximum `distance_km`

#### Step 8: Compute Global Time Range

- `min_time` = min across all riders' first timestamp
- `max_time` = max across all riders' last timestamp
- (Zwift's full-data API provides telemetry at exactly 1-second intervals, so no interpolation is needed)

#### Step 9: Per-Rider Processing

For each rider:

**A. Align Replay Clock:**
- Zwift telemetry begins at the rider's start-line crossing, not necessarily the race gun
- Use the final GPS finish crossing and official result `elapsed_ms` to compute `telemetry_finish - official_elapsed`
- Subtract that offset from `time_sec`; rear-pen riders therefore appear only after their actual release/crossing delay
- Resample shifted telemetry to integer race-clock seconds
- The cached/API field remains named `ttt_time_offset` for backward compatibility, but it now applies to mass starts too

**B. Cut at Finish:**
- The Zwift race results API only returns finishers (DNF riders are silently excluded), so all riders in our data completed the race
- Detect finish crossing: first timestamp where `distance_km >= finish_line_km - 0.03 km` (30m tolerance for alignment noise)
- Prefer official `elapsed_ms` as `finish_time_sec` when replay-clock calibration succeeds
- Record `finish_time_sec` = that timestamp
- Keep data only up to `finish_time + 5 seconds`, drop everything after

**C. Compute Rolling Resistance (CRR):**
- If world detected: `compute_crr_array(lat, lng, world)` — ray-cast each GPS point against surface polygons
- Per-surface CRR values vary by bike type (road_bike, mtb, gravel_bike): Tarmac=0.004, Cobbles=0.0065, Dirt=0.025, etc.
- Default: 0.004 (tarmac)

**D. Index by Time:**
- Set `time_sec` as DataFrame index for O(1) lookups during playback

#### Step 10: Cap Max Time

Clamp `max_time` to last finisher's time + 10 seconds to exclude cooldown/post-race data.

#### Step 11: Build Elevation Profile

**If a calibrated WAD lead-in is available:**
- Use the calibrated WAD lead-in followed by the calibrated main lap
- Repeat only the main lap to the official finish distance
- Allow 100m tolerance when counting laps so tiny metadata/profile distance differences do not append an extra lap
- This removes dependence on one rider's telemetry altitude for long lead-ins such as Richmond Rollercoaster

**If only main-route data is available:**
- Retain the previous loop rebasing and reference-rider lead-in elevation logic

**Fallback:**
- `build_elevation_profile()` from reference rider's `altitude_m` column
- Linear interpolation at 10-meter intervals

#### Step 12: Assemble & Cache

- Read `source_activity_id` from `race_meta.json`
- Build `CleanedRaceData` dataclass
- `save_to_cache()`:
  - Per-rider DataFrames → `cleaned_data/rider_{rank}.parquet`
  - Elevation profile → `cleaned_data/elevation.parquet`
  - Metadata → `cleaned_cache.json` (JSON with rider info, route details, time ranges)

---

### `GET /api/race/data/<race_id>`

**Purpose:** Return the full cleaned race dataset as JSON for the frontend replay player.

**Pipeline:**

1. **Check `_race_data_cache`** — in-memory cache. If missing, re-run `clean_race_data()` or download from blob.
2. **Detect world** if not set (for older cached data)
3. **Build elevation profile JSON** — `{ distance_km: [...], altitude_m: [...] }`
4. **Load race metadata** — `race_meta.json` → `race_start_time`, `event_id`, `event_subgroup_id`
5. **Late joiner detection** — for each rider:
   - Parse race `race_start_time` as ISO 8601 datetime
   - Parse rider's `activity_start_time` as ISO 8601 datetime
   - `is_late_joiner = rider_start_dt > race_start_dt`
6. **Build per-rider JSON** — `{ rank, name, team, activity_id, player_id, weight_kg, is_late_joiner, finish_time_sec, time_sec: [...], distance_km: [...], altitude_m: [...], speed_kmh: [...], power_watts: [...], hr_bpm: [...], lat: [...], lng: [...] }`
7. **World map config** — `get_world_map_config(world)` → map image, GPS bounds, background color
8. **Route polyline** — `load_route_data(route_slug).latlng` → official route lat/lng array
9. **Developer warnings** — `dev_warnings` array (only shown on localhost):
   - Missing Strava segment ID for the route
   - `YOUTUBE_API_KEY` environment variable not set
10. **Return** `{ race_id, route_name, route_slug, source_activity_id, race_start_time, event_id, event_subgroup_id, finish_line_km, min_time, max_time, elevation_profile, riders: [...], map_config, route_latlng, dev_warnings }`

### `GET /api/race/streams/<race_id>`

**Purpose:** Find YouTube livestream VODs from known streamers who raced.

**Pipeline:**

1. **Check disk cache** — `stream_cache.json` in race directory (avoids YouTube API calls)
2. **Load race metadata** — `race_meta.json` → `race_start_time`
3. **Get rider player IDs** — from `_race_data_cache` or fall back to `cleaned_cache.json` / `complete_race_summary.csv`
4. **`find_matching_streams()`:**
   - Load `streamers.json` — maps Zwift player IDs → YouTube channel handles
   - Cross-reference race participants with known streamers
   - For each matching streamer:
     - `resolve_channel_id(handle)` — YouTube API `channels?forHandle=@handle`
     - **Fast path** — `_find_stream_for_race()`: get uploads playlist (`UU{channelId}`), filter by ±48h publish time, check `liveStreamingDetails.actualStartTime` / `actualEndTime` overlap
     - **Slow fallback** — `_find_stream_for_race_search()`: YouTube search API with `eventType=completed`
     - `_detect_trim_offset()` — compare broadcast duration vs current video duration to compute front-trim offset
   - Return `{ streamer_name, youtube_url, stream_title, offset_seconds }`
5. **Cache to disk** — `stream_cache.json` + upload to blob storage

---

## Ride Compare

### Page: `GET /ride-compare`

Renders `ride_compare.html`. Uses the same playback engine as Race Replay (elevation chart, map, rider table).

### `GET /api/ride_compare/fetch_stream`

**Purpose:** Fetch telemetry for 2+ activities and prepare them for side-by-side comparison. Streams progress via **SSE**.

**Params:** `activity_ids` (required, comma-separated activity IDs or URLs)

**Pipeline:**

1. **Parse input** — accepts comma, space, or newline-delimited IDs; extracts numeric ID from URLs via `extract_activity_id()`; requires minimum 2 activities
2. **Generate stable compare ID** — sort unique activity IDs → SHA-256 hash → `ride_compare_<first 16 chars>`
3. **Check cache** — if `complete_race_summary.csv` exists in `race_data/<compare_id>/`, return cached result
4. **Fetch each activity** (sequential, with SSE progress messages):
   - `fetch_rider_telemetry(activity_id, headers)` → raw telemetry + activity metadata
   - `convert_telemetry_to_dataframe()` → DataFrame
   - `save_rider_data()` → writes CSV + raw JSON per rider
   - Extract rider name, weight, player ID, route ID from profile/activity data
   - Compute NP, avg power, max power, avg HR, duration
5. **Name disambiguation** — if multiple riders share the same name, appends activity date (`YYYY-MM-DD`) or datetime (`YYYY-MM-DD HH:MM`) to distinguish them
6. **Save metadata:**
   - `complete_race_summary.csv` — rank, name, activity_id, weight, stats
   - `race_meta.json` — `is_ride_compare: true`, route_id, start times, source_activity_id
7. **Return** `{ success: true, race_id }` (final SSE message)

After fetching, the client calls `loadRaceById(race_id)` which hits the standard `/api/race/load` → `/api/race/data/<race_id>` pipeline to clean and render the comparison.

---

## TTT Analysis

### Page: `GET /ttt-analysis`

Renders `ttt_analysis.html`.

### `POST /api/ttt/fetch`

**Purpose:** Fetch and analyze a team time trial race. The main heavy-lifting endpoint.

**Params (JSON body):** `subgroup_id` (required), `team_tags` (optional, comma-separated)

**Pipeline:**

1. **Fetch race entries** — `get_race_entries(subgroup_id, headers)` via Zwift API
2. **Detect teams** — if `team_tags` not provided, auto-extracts tags from rider names (content inside `[...]` or `(...)`), keeps tags appearing on ≥2 riders
3. **Fetch all rider telemetry** — concurrent with 5 workers, each producing: time_sec, speed_kmh, distance_km, altitude_m, power_watts, lat/lng
4. **Compute draft estimates** per rider:
   - Uses a fixed reference bike (Cadex Tri frame + DT Swiss ARC 1100 DICUT 85 wheels, upgrade level 5)
   - `frontal_area_from_rider(height_m, weight_kg)` → CdA
   - Solo power = `(F_rolling + F_aero) × v / (1 - drivetrain_loss)`
   - Draft watts = solo power − actual power, smoothed over 5-point window
5. **Route alignment** (if route data available) — projects all riders onto shared route via `align_riders_to_route()`, corrects post-finish drift using raw Zwift odometer
6. **TTT time offset correction** — `compute_ttt_time_offset()` aligns staggered start times using finish line reference coordinates
7. **Resample** all riders to integer-second time grid
8. **Team aggregation** — groups riders by team tag, bins into 50 m distance intervals:
   - Lead speed (leader at each distance)
   - Lead power (leader at each distance, raw + smoothed)
   - Avg draft efficiency (mean draft_watts / total_power)
   - Lead rider name, elevation, lead time
   - **Pull statistics** per rider: pull count, avg duration, total front time, avg power (W and W/kg)
9. **Return** `{ event_name, ttt_bike, race_distance_km, teams: [...], unassigned: [...], team_tags, warnings }`

### `POST /api/ttt/reassign`

**Purpose:** Re-aggregate team charts after drag-and-drop rider reassignments, without re-fetching telemetry.

**Params (JSON body):** `subgroup_id`, `assignments` (dict of `activity_id → team_tag`, empty string = unassign)

Uses cached `all_processed` data from the original fetch. Calls `_build_ttt_team_results()` with new assignments and returns the same output structure as `/api/ttt/fetch`.

### `POST /api/ttt/rider`

**Purpose:** Load a single rider's telemetry for the drilldown chart.

**Params (JSON body):** `subgroup_id`, `activity_id`

Returns the rider's data downsampled to 50 m intervals: `{ name, activity_id, weight_kg, distance_km, altitude_m, power_watts, draft_watts }`.

### `POST /api/ttt/debug_bins`

**Purpose:** Debug endpoint — shows per-second leader data in a distance range for diagnostics.

**Params (JSON body):** `subgroup_id`, `team_tag`, `dist_start_km`, `dist_end_km`

---

## Shared Infrastructure

### Zwift API (`shared/data_fetcher.py`)

- **Base URL:** `https://us-or-rly101.zwift.com/api`
- **User-Agent headers:** Mimics the official Zwift mobile client (`Zwift/115 CFNetwork/758.0.2 Darwin/15.0.0`). Without these, the API may return 406.
- **`_request_with_retry()`** — HTTP wrapper with retry on 5xx/429 errors, auto-sets `Accept: application/json`
- **`extract_activity_id()`** — extracts numeric ID from Zwift activity URLs

### Route Lookup (`shared/route_lookup.py`)

- **`zwift_routes/index.json`** — WAD-extracted route index; all route metadata (distance, lead-in, world, ascent, eventOnly). No REST API dependency.
- `get_route_info(route_id)` → route dict by numeric ID
- `get_total_race_distance(route_id)` → `(distanceInMeters + leadinDistanceInMeters) / 1000` km
- `WORLD_ID_TO_MAP` / `MAP_TO_WORLD_ID` — int ↔ name mappings

### Route Geometry (`race_replay/data_cleaner.py`)

- **Source:** ZwiftMap (`zwiftmap.com/strava-segments/{id}`)
- `route_strava_segments.json` — maps route slugs to Strava segment IDs
- `fetch_route_from_zwiftmap()` — downloads `latlng.json`, `distance.json`, `altitude.json`
- Cached locally in `zwiftmap_surfaces/{route_slug}_route.json`
- `RouteData` — Nx2 latlng array, cumulative distance array, `cKDTree` for spatial queries

### Surface Lookup (`shared/surface_lookup.py`)

- **`surface_data.json`** — polygon definitions for non-tarmac surfaces per world
- `compute_surface_types_array(lats, lngs, world)` — for each GPS point, ray-cast against polygons to determine surface type
- `surface_types_to_crr(surface_types, bike_type)` — CRR by surface × bike type
- `compute_crr_array(lats, lngs, world, bike_type)` — combined helper: surface detection + CRR lookup in one call
- `get_bike_type_for_frame(frame_type)` — maps Zwift frame types (`Standard`, `TT`, `Gravel`, `MTB`) to CRR bike categories (`road_bike`, `mtb`, `gravel_bike`)

CRR values:

| Surface | Road Bike | MTB   | Gravel |
| ------- | --------- | ----- | ------ |
| Tarmac  | 0.004     | 0.004 | 0.004  |
| Cobbles | 0.0065    | 0.005 | 0.0055 |
| Dirt    | 0.025     | 0.006 | 0.008  |
| Gravel  | 0.020     | 0.007 | 0.009  |

### World Config (`shared/world_config.py`)

Static config for 12 Zwift worlds (Watopia, Richmond, London, New York, Innsbruck, Bologna, Yorkshire, Crit City, Makuri, France, Paris, Scotland): map image path, GPS bounding box (lat/lng min/max), background color. Used by the race replay map overlay.

The Watopia background (`static/maps/watopia.png`) is a label-free
4096 x 3072 composite of the installed Zwift game PNGs from
`data/Noesis/Blend_Data/Resources/Worlds/world1`, refreshed on 2026-10-28.
The main image and auxiliary tiles 01/02 were dated 2026-10-14.
At a grid size of 1024 pixels, the layout is:

```text
mapaux02 | map.png (top-left)    | map.png (top-right)    | mapaux03
mapaux01 | map.png (bottom-left) | map.png (bottom-right) | mapaux04
mapaux08 | mapaux07              | mapaux06              | mapaux05
```

Use all nine PNGs without cropping. Keep `map.png` and auxiliary tiles
01-05/08 at their original sizes; resize the 2048 x 2048 auxiliary tiles
06/07 to 1024 x 1024 with Lanczos resampling before placing them.
The composite covers the same geographic extent as the previous background,
so the existing Watopia GPS bounds remain unchanged. Both race replay and
the Surface Map / Route Explorer use this shared background and configuration.

### Blob Storage (`shared/blob_storage.py`)

Azure Blob Storage wrapper — **no-op when `AZURE_STORAGE_CONNECTION_STRING` is not set**.
- `upload_race_dir()` / `download_race_dir()` — persist/restore full race data directories
- `upload_file()` — upload a single file (e.g. `stream_cache.json`)
- `race_exists_in_blob()` — checks for `complete_race_summary.csv` sentinel
- `list_races()` — enumerate all stored races (race_id + name from `race_meta.json`)

### Bike Database (`bike_comparison/bike_data.py`)

- **Source:** `zwiftdata/frames.json`, `wheels.json`, `bikes.json` (from ZwifterBikes)
- `BikeDatabase` — singleton, auto-downloads missing JSON files
- `bikes.json` contains pre-computed combo table: `(frame_id, wheel_id)` → arrays of 6 Cd values and 6 weight values (upgrade levels 0–5)
- `BikeSetup` dataclass: `frame_id`, `frame_name`, `wheel_id`, `wheel_name`, `upgrade_level`, `cd`, `weight_kg`, `frame_type`

### Physics Engine (`bike_comparison/physics.py`)

- **Power equation:** `P = (F_gravity + F_rolling + F_aero) × v / (1 - η)` with η = 0.025 (2.5% drivetrain loss)
  - `F_gravity = total_mass × g × gradient`
  - `F_rolling = total_mass × g × CRR`
  - `F_aero = 0.5 × ρ × CdA × v²` where CdA = Cd × frontal_area
- **`compare_bike_setups()`** — energy-balance method that preserves the draft benefit from real race data when computing hypothetical alternative power

### Normalized Power (`shared/utils.py`)

Dr. Coggan's algorithm:
1. 30-second rolling average of power
2. Raise to 4th power
3. Take mean
4. Take 4th root

Window adapts to actual sample rate when `time_sec` is provided.

### Timestamp Correction (`race_replay/data_cleaner.py`)

- `fix_timestamp_offsets()` — detects Zwift server clock jumps where telemetry data was actually continuous (checks distance & speed continuity across the jump). Subtracts excess time to produce monotonic timestamps.

### Zwift Auth (`shared/zwift_auth.py`)

- Zwift OAuth 2.0 Authorization Code Flow
- `get_token_with_password()` — email + password → access/refresh token pair (used by Azure Function for headless auth)
- `exchange_code_for_tokens()` — OAuth callback code exchange (used by web app)
- `refresh_access_token()` — automatic token refresh on expiry
- Token storage via Flask session (`ZwiftTokens` dict: access_token, refresh_token, expires_at)

### YouTube Streams (`shared/youtube_streams.py`)

- `find_matching_streams()` — cross-references race participants with known streamers from `streamers.json`
- `resolve_channel_id(handle)` — YouTube API channel lookup by handle
- `_find_stream_for_race()` — fast path: scan uploads playlist for VODs overlapping race time (±48h)
- `_find_stream_for_race_search()` — slow fallback: YouTube search API with `eventType=completed`
- `_detect_trim_offset()` — computes front-trim offset by comparing broadcast vs. current video duration
