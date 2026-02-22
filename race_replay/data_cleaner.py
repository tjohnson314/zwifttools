"""
Data cleaning module for race telemetry.

Handles:
- Loading and parsing raw race data
- Aligning start times with race results
- Cutting off cooldown data after finish line
- Route-based distance alignment using ZwiftMap coordinates
- Caching cleaned data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


# Bump this version when data cleaning logic changes in a way that invalidates
# previously cached results.  The cache loader will discard stale caches.
CLEANING_VERSION = 2

# Map of Strava segment IDs for each route (sourced from ZwiftMap / ZwiftInsider)
# Loaded from route_strava_segments.json
def _load_route_strava_segments() -> Dict[str, int]:
    seg_file = Path(__file__).parent.parent / 'route_strava_segments.json'
    if seg_file.exists():
        with open(seg_file) as f:
            return json.load(f)
    return {}

ROUTE_STRAVA_SEGMENTS = _load_route_strava_segments()


@dataclass
class RiderData:
    """Cleaned and interpolated rider data."""
    rank: int
    activity_id: str
    name: str
    team: str
    data: pd.DataFrame  # Telemetry data (1-second intervals from Zwift API)
    finish_time_sec: Optional[float]  # Time when rider crossed finish line
    weight_kg: float = 75.0  # Rider weight in kg
    player_id: Optional[int] = None  # Numeric Zwift profile ID
    activity_start_time: Optional[str] = None  # ISO 8601 UTC when activity started


@dataclass
class CleanedRaceData:
    """Complete cleaned race data."""
    race_id: str
    route_name: str
    finish_line_km: float
    riders: List[RiderData]
    elevation_profile: pd.DataFrame  # distance_km, altitude_m
    min_time: float
    max_time: float
    source_activity_id: Optional[str] = None  # Activity ID used to fetch the race
    route_slug: Optional[str] = None  # Route slug for linking to Zwift Insider
    world: Optional[str] = None  # Detected Zwift world (WATOPIA, LONDON, etc.)


def haversine(lat1, lng1, lat2, lng2):
    """Calculate distance in meters between two lat/lng points (scalar or array)."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


@dataclass
class RouteData:
    """Official route path data from ZwiftMap."""
    strava_segment_id: int
    route_slug: str
    route_name: str
    latlng: np.ndarray  # Nx2 array of [lat, lng]
    distance: np.ndarray  # Distance along route in meters
    altitude: np.ndarray  # Altitude in meters
    total_distance_m: float
    kdtree: cKDTree  # For fast nearest-neighbor lookups


def load_route_data(route_slug: str) -> Optional[RouteData]:
    """
    Load official route coordinates from ZwiftMap data.
    
    Args:
        route_slug: Route identifier (e.g., 'london-8')
        
    Returns:
        RouteData object or None if not found
    """
    # Check for cached route file
    base_path = Path(__file__).parent.parent / 'zwiftmap_surfaces'
    route_file = base_path / f'{route_slug.replace("-", "_")}_route.json'
    
    if not route_file.exists():
        # Try to fetch from ZwiftMap
        strava_id = ROUTE_STRAVA_SEGMENTS.get(route_slug)
        if not strava_id:
            print(f"No Strava segment ID known for route: {route_slug}")
            return None
        
        route_data = fetch_route_from_zwiftmap(strava_id, route_slug)
        if route_data:
            # Save for future use
            with open(route_file, 'w') as f:
                json.dump(route_data, f, indent=2)
        else:
            return None
    else:
        with open(route_file) as f:
            route_data = json.load(f)
    
    # Parse the data
    latlng = np.array([[pt[0], pt[1]] if isinstance(pt, list) else [pt['value'][0], pt['value'][1]] 
                       for pt in route_data['latlng']])
    distance = np.array(route_data['distance'])
    altitude = np.array(route_data['altitude'])
    
    # Build KD-tree for fast nearest-neighbor lookups
    # Use scaled coordinates for better distance approximation
    lat_center = np.mean(latlng[:, 0])
    lng_scale = np.cos(np.radians(lat_center))
    scaled_coords = np.column_stack([latlng[:, 0], latlng[:, 1] * lng_scale])
    kdtree = cKDTree(scaled_coords)
    
    return RouteData(
        strava_segment_id=route_data.get('stravaSegmentId', 0),
        route_slug=route_slug,
        route_name=route_data.get('routeName', route_slug),
        latlng=latlng,
        distance=distance,
        altitude=altitude,
        total_distance_m=distance[-1],
        kdtree=kdtree
    )


def fetch_route_from_zwiftmap(strava_segment_id: int, route_slug: str) -> Optional[Dict]:
    """Fetch route data from ZwiftMap API."""
    import urllib.request
    
    base_url = f"https://zwiftmap.com/strava-segments/{strava_segment_id}"
    
    try:
        with urllib.request.urlopen(f"{base_url}/latlng.json", timeout=10) as resp:
            latlng = json.loads(resp.read().decode())
        with urllib.request.urlopen(f"{base_url}/distance.json", timeout=10) as resp:
            distance = json.loads(resp.read().decode())
        with urllib.request.urlopen(f"{base_url}/altitude.json", timeout=10) as resp:
            altitude = json.loads(resp.read().decode())
        
        return {
            'stravaSegmentId': strava_segment_id,
            'routeSlug': route_slug,
            'routeName': route_slug.replace('-', ' ').title(),
            'totalDistance': distance[-1],
            'pointCount': len(latlng),
            'latlng': latlng,
            'distance': distance,
            'altitude': altitude
        }
    except Exception as e:
        print(f"Failed to fetch route data from ZwiftMap: {e}")
        return None


def project_to_route(route: RouteData, lat: float, lng: float) -> Tuple[float, float]:
    """
    Project a GPS point onto the route path.
    
    Args:
        route: RouteData with KD-tree
        lat, lng: Point to project
        
    Returns:
        (distance_along_route_m, perpendicular_distance_m)
    """
    # Scale longitude for better distance approximation
    lat_center = np.mean(route.latlng[:, 0])
    lng_scale = np.cos(np.radians(lat_center))
    scaled_point = [lat, lng * lng_scale]
    
    # Find nearest point on route
    dist_scaled, idx = route.kdtree.query(scaled_point)
    
    # Get actual distance along route
    route_distance = route.distance[idx]
    
    # Calculate actual perpendicular distance in meters
    nearest_lat = route.latlng[idx, 0]
    nearest_lng = route.latlng[idx, 1]
    perp_distance = haversine(lat, lng, nearest_lat, nearest_lng)
    
    # Refine by interpolating between neighboring points
    if 0 < idx < len(route.distance) - 1:
        # Check if we're between this point and the next or previous
        for neighbor_idx in [idx - 1, idx + 1]:
            n_lat = route.latlng[neighbor_idx, 0]
            n_lng = route.latlng[neighbor_idx, 1]
            
            # Vector from neighbor to current route point
            v_route = np.array([route.latlng[idx, 0] - n_lat, 
                               (route.latlng[idx, 1] - n_lng) * lng_scale])
            # Vector from neighbor to query point
            v_point = np.array([lat - n_lat, (lng - n_lng) * lng_scale])
            
            # Project point onto route segment
            route_len_sq = np.dot(v_route, v_route)
            if route_len_sq > 0:
                t = np.clip(np.dot(v_point, v_route) / route_len_sq, 0, 1)
                if 0 < t < 1:
                    # Interpolate distance
                    d1 = route.distance[neighbor_idx]
                    d2 = route.distance[idx]
                    interpolated_dist = d1 + t * (d2 - d1)
                    
                    # Check if this gives a closer projection
                    proj_lat = n_lat + t * (route.latlng[idx, 0] - n_lat)
                    proj_lng = n_lng + t * (route.latlng[idx, 1] - n_lng)
                    new_perp = haversine(lat, lng, proj_lat, proj_lng)
                    
                    if new_perp < perp_distance:
                        route_distance = interpolated_dist
                        perp_distance = new_perp
    
    return route_distance, perp_distance


def align_riders_to_route(riders: List[Dict], route: RouteData, 
                          max_deviation_m: float = 50,
                          leadin_distance_m: Optional[float] = None) -> tuple:
    """
    Align all rider GPS coordinates to official route distances.
    Uses vectorized KD-tree queries for performance.
    
    Handles routes that cross over themselves (e.g. Neokyo All Nighter) by
    querying multiple KD-tree neighbors and choosing the one whose route
    distance best matches the expected position based on the rider's raw
    Zwift distance data. This avoids snapping to the wrong segment when
    the route overlaps itself geographically.
    
    On loop routes (start ≈ end), applies global offset unwrapping so all
    riders share the same distance reference frame.
    
    Args:
        riders: List of rider dicts with 'data' DataFrame containing lat/lng
        route: Official route data
        max_deviation_m: Maximum allowed deviation from route (for filtering bad GPS)
        
    Returns:
        Tuple of (riders, loop_start_offset_m) where loop_start_offset_m is the
        route distance (in meters) where the race starts on the Strava segment,
        or None for non-loop routes.
    """
    # Pre-compute scaling factor for longitude
    lat_center = np.mean(route.latlng[:, 0])
    lng_scale = np.cos(np.radians(lat_center))
    
    # Detect if this is a loop route (start ≈ end within 100m)
    start_ll = route.latlng[0]
    end_ll = route.latlng[-1]
    start_end_gap = haversine(start_ll[0], start_ll[1], end_ll[0], end_ll[1])
    is_loop = start_end_gap < 100  # meters
    route_total = route.total_distance_m
    loop_start_offset = None  # Will be set for loop routes
    all_start_offsets = []  # Collect per-rider start offsets for median
    
    # Query more neighbors so we can disambiguate overlapping route sections.
    # On routes that cross over themselves, the geographically closest point
    # may not be on the correct segment — we need alternatives to compare.
    K_NEIGHBORS = 8
    k_actual = min(K_NEIGHBORS, len(route.distance))
    
    rider_projections = []  # Store (route_distances, deviations, raw_dist) per rider

    for rider in riders:
        df = rider['data']
        if 'lat' not in df.columns or 'lng' not in df.columns:
            rider_projections.append(None)
            continue
        
        # Get all rider coordinates
        lats = df['lat'].values
        lngs = df['lng'].values
        
        # Scale coordinates for KD-tree query
        scaled_points = np.column_stack([lats, lngs * lng_scale])
        
        # Batch query - find K nearest route points to handle overlapping segments
        kd_dists, kd_indices = route.kdtree.query(scaled_points, k=k_actual)
        
        # Build expected distance along route from Zwift's raw distance data
        raw_dist_km = df['distance_km'].values if 'distance_km' in df.columns else None
        if raw_dist_km is not None:
            raw_traveled_m = (raw_dist_km - raw_dist_km[0]) * 1000
        else:
            raw_traveled_m = None
        
        # For each rider GPS point, pick the KD-tree neighbor whose route
        # distance best matches the expected position. This correctly
        # handles routes that cross over themselves.
        n_points = len(lats)
        route_distances = np.empty(n_points)
        deviations = np.empty(n_points)
        
        # Compute start offset: account for race lead-in distance.
        # During the lead-in, riders are NOT on the official route, so the
        # first GPS points may be far from any route point. We scan for the
        # first point that is close to the route (<50m) and use its
        # nearest-neighbor route distance vs raw traveled distance to
        # estimate the offset.
        #
        # Late joiners may start directly on the route (no lead-in at all).
        # We first check if the rider's initial GPS points are already on the
        # route; if so, skip the lead-in search entirely.
        #
        # If the known lead-in distance is available (from routes_cache.json),
        # we use it to skip ahead to approximately the right index and only
        # search a window around it. This avoids scanning thousands of
        # off-route GPS points on routes with long lead-ins (up to ~12 km).
        
        # Quick check: are the rider's first GPS points already on the route?
        # (Handles late joiners who start mid-race with no lead-in)
        n_quick = min(10, n_points)
        nn_quick = kd_indices[:n_quick, 0]
        quick_devs = haversine(
            lats[:n_quick], lngs[:n_quick],
            route.latlng[nn_quick, 0], route.latlng[nn_quick, 1]
        )
        rider_starts_on_route = np.median(quick_devs) < 50  # most of first 10 pts within 50m
        
        # Determine the search range for finding where the rider joins the route
        if rider_starts_on_route:
            # Rider is already on the route from the start (late joiner or no lead-in)
            search_start_idx = 0
            search_end_idx = min(50, n_points)
        elif leadin_distance_m is not None and leadin_distance_m > 50 and raw_traveled_m is not None:
            # Known lead-in: skip ahead to where raw_traveled_m ≈ leadin_distance_m
            # Search from 80% of lead-in to 150% (margin for GPS jitter / speed variation)
            search_start_m = leadin_distance_m * 0.8
            search_end_m = leadin_distance_m * 1.5
            search_start_idx = int(np.searchsorted(raw_traveled_m, search_start_m))
            search_end_idx = int(np.searchsorted(raw_traveled_m, search_end_m))
            # Clamp and ensure we check at least some points
            search_start_idx = max(0, min(search_start_idx, n_points - 1))
            search_end_idx = min(n_points, max(search_end_idx, search_start_idx + 50))
        else:
            # No known lead-in or very short: search from the beginning
            # Use a generous range to handle up to ~12 km lead-ins (~1200 points)
            search_start_idx = 0
            search_end_idx = min(1500, n_points)
        
        # Compute haversine deviations only for the search range (vectorized)
        search_slice = slice(search_start_idx, search_end_idx)
        nn_indices_search = kd_indices[search_slice, 0]
        geo_devs_search = haversine(
            lats[search_slice], lngs[search_slice],
            route.latlng[nn_indices_search, 0], route.latlng[nn_indices_search, 1]
        )
        close_mask = geo_devs_search < 50  # within 50m of route
        leadin_end_idx = 0  # index where rider first reaches the route
        if np.any(close_mask) and raw_traveled_m is not None:
            # Map back to absolute indices
            close_indices = np.where(close_mask)[0] + search_start_idx
            leadin_end_idx = close_indices[0]
            # Use several close points for robustness
            sample = close_indices[:min(20, len(close_indices))]
            offsets = route.distance[kd_indices[sample, 0]] - raw_traveled_m[sample]
            initial_offset = np.median(offsets)
            leadin_est = -initial_offset if initial_offset < 0 else 0
            if leadin_est > 50:
                print(f"  Rider {rider['rank']}: detected ~{leadin_est:.0f}m lead-in "
                      f"(first on-route point at idx {leadin_end_idx}, "
                      f"raw dist {raw_traveled_m[leadin_end_idx]:.0f}m)")
        else:
            # Fallback: use first few points as before
            n_start = min(10, n_points)
            start_dists = route.distance[kd_indices[:n_start, 0]]
            initial_offset = np.median(start_dists)
        
        # ---- Vectorized alignment (replaces per-point Python loop) ----
        
        # 1. Lead-in points: assign offset + raw_traveled, deviation=999
        if leadin_end_idx > 0 and raw_traveled_m is not None:
            leadin_slice = slice(0, leadin_end_idx)
            route_distances[leadin_slice] = initial_offset + raw_traveled_m[leadin_slice]
            deviations[leadin_slice] = 999
        
        # 2. On-route points: vectorized KNN scoring
        on_route = slice(leadin_end_idx, n_points)
        or_indices = kd_indices[on_route]    # shape (N_or, K)
        or_kd_dists = kd_dists[on_route]     # shape (N_or, K)
        or_lats = lats[on_route]
        or_lngs = lngs[on_route]
        n_or = len(or_lats)
        
        if n_or > 0:
            # Route distances for each KNN candidate: (N_or, K)
            cand_dists = route.distance[or_indices]
            
            if raw_traveled_m is not None:
                # Expected distance for each on-route point: (N_or,)
                expected_m = initial_offset + raw_traveled_m[on_route]
                
                if is_loop:
                    # Wrap offsets for expected: shape (5,)
                    wrap_k = np.arange(-2, 3, dtype=np.float64)
                    # expected_candidates: (N_or, 5)
                    expected_all = expected_m[:, None] + wrap_k[None, :] * route_total
                    # candidate with wrap offsets: (N_or, K, 5)
                    cand_all = cand_dists[:, :, None] + wrap_k[None, None, :] * route_total
                    # Absolute differences: (N_or, K, 5, 5) — each cand+wrap vs each expected+wrap
                    # But this is equivalent to: for each (cand+k1*T) vs each (exp+k2*T)
                    # = |cand - exp + (k1-k2)*T|. Since k1,k2 ∈ {-2..2}, k1-k2 ∈ {-4..4}
                    # Simplification: compute |cand - exp + m*T| for m in {-4..4}
                    wrap_m = np.arange(-4, 5, dtype=np.float64)
                    # diff: (N_or, K, 9)
                    diff = cand_dists[:, :, None] - expected_m[:, None, None] + wrap_m[None, None, :] * route_total
                    scores = np.min(np.abs(diff), axis=2)  # (N_or, K) — best wrap per candidate
                else:
                    # Non-loop: single expected, no wrapping
                    # |cand_dist - expected_m| → (N_or, K)
                    scores = np.abs(cand_dists - expected_m[:, None])
            else:
                # No raw distance: use KD-tree geometric distance as score
                scores = or_kd_dists
            
            # Pick the best candidate (lowest score) per point
            best_j = np.argmin(scores, axis=1)  # (N_or,)
            row_idx = np.arange(n_or)
            best_route_idx = or_indices[row_idx, best_j]  # route point indices
            best_dists = route.distance[best_route_idx]    # (N_or,)
            
            # 3. Vectorized segment interpolation for sub-point precision
            # For each best route point, try projecting onto both neighbor segments
            route_ll = route.latlng  # (R, 2)
            route_d = route.distance  # (R,)
            n_route = len(route_d)
            
            # Best route point coordinates
            best_lat = route_ll[best_route_idx, 0]
            best_lng = route_ll[best_route_idx, 1]
            
            # Haversine from rider to best route point (vectorized)
            best_dev_m = haversine(or_lats, or_lngs, best_lat, best_lng)
            interp_dist = best_dists.copy()
            
            # Try both neighbors (idx-1 and idx+1)
            for delta in [-1, 1]:
                nbr_idx = best_route_idx + delta
                # Mask for valid neighbors (within route bounds)
                valid = (nbr_idx >= 0) & (nbr_idx < n_route)
                if not np.any(valid):
                    continue
                
                vi = np.where(valid)[0]  # indices into on-route arrays
                bi = best_route_idx[vi]
                ni = nbr_idx[vi]
                
                # Segment vector: best - neighbor (in scaled coords)
                r_lat = route_ll[bi, 0] - route_ll[ni, 0]
                r_lng = (route_ll[bi, 1] - route_ll[ni, 1]) * lng_scale
                # Vector from neighbor to rider
                p_lat = or_lats[vi] - route_ll[ni, 0]
                p_lng = (or_lngs[vi] - route_ll[ni, 1]) * lng_scale
                
                seg_len_sq = r_lat * r_lat + r_lng * r_lng
                # Only project where segment has non-zero length
                has_len = seg_len_sq > 0
                vi2 = vi[has_len]
                if len(vi2) == 0:
                    continue
                    
                t = (p_lat[has_len] * r_lat[has_len] + p_lng[has_len] * r_lng[has_len]) / seg_len_sq[has_len]
                t = np.clip(t, 0.0, 1.0)
                
                # Only consider interior projections (0 < t < 1)
                interior = (t > 0) & (t < 1)
                vi3 = vi2[interior]
                if len(vi3) == 0:
                    continue
                t_int = t[interior]
                bi3 = best_route_idx[vi3]
                ni3 = nbr_idx[vi3]
                
                # Interpolated route distance
                d1 = route_d[ni3]
                d2 = route_d[bi3]
                cand_interp = d1 + t_int * (d2 - d1)
                
                # Projected point on segment
                proj_lat = route_ll[ni3, 0] + t_int * (route_ll[bi3, 0] - route_ll[ni3, 0])
                proj_lng = route_ll[ni3, 1] + t_int * (route_ll[bi3, 1] - route_ll[ni3, 1])
                proj_dev = haversine(or_lats[vi3], or_lngs[vi3], proj_lat, proj_lng)
                
                # Use interpolated distance where projection is closer
                improved = proj_dev < best_dev_m[vi3]
                imp_idx = vi3[improved]
                interp_dist[imp_idx] = cand_interp[improved]
                best_dev_m[imp_idx] = proj_dev[improved]
            
            route_distances[on_route] = interp_dist
            
            # 4. Vectorized deviation: haversine to nearest route point
            nn_idx = or_indices[:, 0]  # nearest neighbor (first KD result)
            deviations[on_route] = haversine(
                or_lats, or_lngs,
                route_ll[nn_idx, 0], route_ll[nn_idx, 1]
            )
        
        # Collect per-rider start offset for loop routes
        if is_loop and raw_dist_km is not None:
            all_start_offsets.append(initial_offset)
        rider_projections.append((route_distances, deviations, raw_dist_km, initial_offset))
    
    # Compute global start offset for loop routes
    if is_loop and all_start_offsets:
        loop_start_offset = np.median(all_start_offsets)
        print(f"  Loop route: global start_offset={loop_start_offset:.0f}m, "
              f"route_total={route_total:.0f}m, gap={start_end_gap:.0f}m")
    
    # Pass 2: Apply unwrapping and rebasing with the global offset
    for rider, proj in zip(riders, rider_projections):
        if proj is None:
            continue
        
        route_distances, deviations, raw_dist_km, rider_offset = proj
        df = rider['data']
        
        # --- Detect late joiners on loop routes (flag only, trim after unwrap) ---
        # Late joiners start from a different pen and ride a different lead-in
        # path.  Their lead-in distance values overlap with the route loop data
        # at wrong altitudes, making them appear off the elevation profile.
        # We flag them here but trim AFTER unwrap/rebase because the unwrap
        # needs the full raw_traveled_m (starting from ride start) to compute
        # correct expected values.
        late_joiner_mask = None
        if is_loop and loop_start_offset is not None:
            offset_diff = abs(rider_offset - loop_start_offset)
            if offset_diff > route_total * 0.1:  # >10% of loop = different start
                leadin_mask = deviations >= 500
                if leadin_mask.any() and not leadin_mask.all():
                    n_dropped = leadin_mask.sum()
                    print(f"  Rider {rider['rank']}: late joiner (offset diff "
                          f"{offset_diff:.0f}m), dropping {n_dropped} lead-in points")
                    late_joiner_mask = ~leadin_mask  # True = on-route (keep)
        
        # --- Fix loop-route ambiguity ---
        if is_loop and raw_dist_km is not None:
            raw_traveled_m = (raw_dist_km - raw_dist_km[0]) * 1000
            
            # Use the global start offset for all riders
            expected = loop_start_offset + raw_traveled_m
            
            # For each point, pick the wrap variant closest to expected (vectorized)
            wrap_k = np.arange(-2, 3, dtype=np.float64)
            # (N, 5) candidates
            candidates = route_distances[:, None] + wrap_k[None, :] * route_total
            diffs = np.abs(candidates - expected[:, None])
            best_wrap = np.argmin(diffs, axis=1)
            route_distances = candidates[np.arange(len(route_distances)), best_wrap]
            
            # Rebase with the global offset so all riders share the same reference
            route_distances -= loop_start_offset
        
        # --- Apply late joiner trim (after unwrap/rebase, before smoothing) ---
        if late_joiner_mask is not None:
            df = df.iloc[np.where(late_joiner_mask)[0]].copy()
            route_distances = route_distances[late_joiner_mask]
            deviations = deviations[late_joiner_mask]
            if raw_dist_km is not None:
                raw_dist_km = raw_dist_km[late_joiner_mask]
        
        # Smooth route distances using Zwift distance deltas to eliminate
        # route-point quantization (~13m steps). The KD-tree alignment gives
        # the correct segment of the route but snaps to discrete points.
        # We use the rider's actual Zwift distance deltas (cm precision) for
        # smooth sub-meter positioning, with a gentle pull toward the aligned
        # values to prevent cumulative drift.
        if raw_dist_km is not None and len(raw_dist_km) == len(route_distances):
            raw_deltas_m = np.diff(raw_dist_km) * 1000  # meters per step
            smooth = np.empty_like(route_distances)
            smooth[0] = route_distances[0]
            alpha = 0.05  # 5% drift correction toward route alignment per step
            for i in range(1, len(smooth)):
                delta = max(raw_deltas_m[i - 1], 0.0)  # never go backwards
                predicted = smooth[i - 1] + delta
                smooth[i] = predicted + alpha * (route_distances[i] - predicted)
            route_distances = smooth
        elif 'speed_kmh' in df.columns and 'time_sec' in df.columns:
            # Fallback: use speed * dt
            speeds_mps = df['speed_kmh'].values / 3.6
            times = df['time_sec'].values
            smooth = np.empty_like(route_distances)
            smooth[0] = route_distances[0]
            alpha = 0.05
            for i in range(1, len(smooth)):
                dt = max(times[i] - times[i - 1], 0)
                predicted = smooth[i - 1] + speeds_mps[i] * dt
                smooth[i] = predicted + alpha * (route_distances[i] - predicted)
            route_distances = smooth
        
        rider['data'] = df.copy()
        rider['data']['distance_km'] = route_distances / 1000.0
        rider['data']['route_deviation_m'] = deviations
        
        # Report alignment quality (exclude lead-in markers where deviation=999)
        on_route_mask = deviations < 900
        if np.any(on_route_mask):
            avg_deviation = np.mean(deviations[on_route_mask])
            if avg_deviation > 20:
                print(f"  Warning: Rider {rider['rank']} has high avg route deviation: {avg_deviation:.1f}m")
    
    return riders, loop_start_offset


def find_course_landmarks(df: pd.DataFrame, min_angle_change: float = 20, 
                          min_spacing_km: float = 0.3, window_size: int = 5) -> List[Dict]:
    """
    Find distinctive course features (turn apexes) using lat/lng data.
    """
    if 'lat' not in df.columns or 'lng' not in df.columns:
        return []
    
    lats = df['lat'].values
    lngs = df['lng'].values
    dists = df['distance_km'].values
    n = len(df)
    
    if n < window_size * 2 + 1:
        return []
    
    # Calculate angle change at each point
    bearings = []
    for i in range(n):
        if i < window_size or i >= n - window_size:
            bearings.append(np.nan)
            continue
        
        dx1 = lngs[i] - lngs[i - window_size]
        dy1 = lats[i] - lats[i - window_size]
        dx2 = lngs[i + window_size] - lngs[i]
        dy2 = lats[i + window_size] - lats[i]
        
        if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
            bearings.append(np.nan)
            continue
        
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        angle_change = angle2 - angle1
        
        while angle_change > np.pi:
            angle_change -= 2 * np.pi
        while angle_change < -np.pi:
            angle_change += 2 * np.pi
        
        bearings.append(np.degrees(angle_change))
    
    bearings = np.array(bearings)
    
    # Find local maxima (turn apexes)
    landmarks = []
    for i in range(window_size, n - window_size):
        if np.isnan(bearings[i]):
            continue
        if landmarks and dists[i] - landmarks[-1]['distance_km'] < min_spacing_km:
            continue
        
        abs_angle = np.abs(bearings[i])
        if abs_angle < min_angle_change:
            continue
        
        # Check if local maximum
        search_range = min(window_size, 3)
        is_apex = True
        for j in range(i - search_range, i + search_range + 1):
            if j != i and not np.isnan(bearings[j]):
                if np.abs(bearings[j]) > abs_angle:
                    is_apex = False
                    break
        
        if is_apex:
            landmarks.append({
                'lat': lats[i],
                'lng': lngs[i],
                'distance_km': dists[i],
                'angle_change': bearings[i]
            })
    
    return landmarks


def find_rider_at_landmark(rider_df: pd.DataFrame, landmark: Dict, 
                           max_dist_m: float = 20, search_range_km: float = 0.5) -> Optional[Dict]:
    """Find when a rider passes closest to a landmark location."""
    if 'lat' not in rider_df.columns or 'lng' not in rider_df.columns:
        return None
    
    lats = rider_df['lat'].values
    lngs = rider_df['lng'].values
    dists = rider_df['distance_km'].values
    
    expected_dist = landmark['distance_km']
    mask = np.abs(dists - expected_dist) < search_range_km
    
    if not np.any(mask):
        return None
    
    indices = np.where(mask)[0]
    distances = haversine(lats[mask], lngs[mask], landmark['lat'], landmark['lng'])
    
    min_idx_local = np.argmin(distances)
    min_dist = distances[min_idx_local]
    min_idx = indices[min_idx_local]
    
    if min_dist < max_dist_m:
        return {
            'distance_km': rider_df['distance_km'].iloc[min_idx],
            'proximity_m': min_dist
        }
    return None


def align_rider_distances(riders: List[Dict], landmarks: List[Dict]) -> Tuple[List[Dict], Dict[int, float]]:
    """
    Align all rider distances using course landmarks.
    
    Returns:
        riders: Riders with corrected distances
        offsets: Dict of rank -> average offset applied
    """
    if not riders or not landmarks:
        return riders, {}
    
    # Use rider with most data as reference
    ref_rider = max(riders, key=lambda r: len(r['data']))
    ref_df = ref_rider['data']
    
    offsets = {}
    
    for rider in riders:
        if rider['rank'] == ref_rider['rank']:
            offsets[rider['rank']] = 0.0
            continue
        
        landmark_offsets = []
        for lm in landmarks:
            ref_result = find_rider_at_landmark(ref_df, lm)
            rider_result = find_rider_at_landmark(rider['data'], lm)
            
            if ref_result and rider_result:
                offset = lm['distance_km'] - rider_result['distance_km']
                landmark_offsets.append({
                    'ref_dist': lm['distance_km'],
                    'offset': offset
                })
        
        if len(landmark_offsets) < 2:
            offsets[rider['rank']] = 0.0
            continue
        
        # Interpolate offsets
        ref_dists = [lo['ref_dist'] for lo in landmark_offsets]
        offset_vals = [lo['offset'] for lo in landmark_offsets]
        
        offset_interp = interp1d(
            ref_dists, offset_vals,
            kind='linear',
            bounds_error=False,
            fill_value=(offset_vals[0], offset_vals[-1])
        )
        
        rider['data'] = rider['data'].copy()
        rider['data']['distance_km'] = rider['data']['distance_km'] + offset_interp(rider['data']['distance_km'])
        offsets[rider['rank']] = float(np.mean(offset_vals))
    
    return riders, offsets


# ---------------------------------------------------------------------------
# Timestamp offset correction
# ---------------------------------------------------------------------------

# Minimum jump factor to consider a timestamp offset (relative to normal tick)
OFFSET_JUMP_FACTOR = 10


def fix_timestamp_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and correct timestamp offsets where the server clock jumped
    but the telemetry (distance, speed, power, HR) remained continuous.

    This happens when Zwift's server reassigns a world-time reference mid-ride.
    The actual ride was uninterrupted — only ``timeInSec`` is wrong.

    Returns a new DataFrame with corrected ``time_sec``.
    """
    times = df['time_sec'].values.copy()
    if len(times) < 3:
        return df

    diffs = np.diff(times)
    normal_tick = float(np.median(diffs))
    if normal_tick <= 0:
        return df

    threshold = normal_tick * OFFSET_JUMP_FACTOR
    jump_indices = np.where(diffs > threshold)[0]

    if len(jump_indices) == 0:
        return df

    cumulative_correction = 0.0
    corrected = times.copy().astype(float)

    for ji in jump_indices:
        actual_jump = diffs[ji]

        # Check telemetry continuity across the jump
        if not _is_continuous_across_jump(df, ji, normal_tick):
            continue  # genuine gap — don't correct

        excess = actual_jump - normal_tick
        cumulative_correction += excess

        # Shift everything after this index back
        corrected[ji + 1:] -= excess

    if cumulative_correction == 0:
        return df

    result = df.copy()
    result['time_sec'] = corrected
    return result


def _is_continuous_across_jump(df: pd.DataFrame, idx: int, normal_tick: float) -> bool:
    """Return True if the telemetry is continuous across a time jump at *idx*."""
    if idx + 1 >= len(df):
        return False

    row_before = df.iloc[idx]
    row_after = df.iloc[idx + 1]

    speed_before = row_before.get('speed_kmh', 0)
    speed_after = row_after.get('speed_kmh', 0)
    avg_speed_ms = ((speed_before + speed_after) / 2) / 3.6

    dist_before = row_before.get('distance_km', 0) * 1000
    dist_after = row_after.get('distance_km', 0) * 1000
    actual_dist_delta = dist_after - dist_before

    expected_dist_delta = avg_speed_ms * normal_tick

    if expected_dist_delta <= 0:
        return False

    ratio = actual_dist_delta / expected_dist_delta
    if ratio < 0.2 or ratio > 3.0:
        return False

    if speed_before > 0:
        speed_ratio = speed_after / speed_before
        if speed_ratio < 0.5 or speed_ratio > 2.0:
            return False

    return True


def load_raw_rider_data(csv_path: Path, json_path: Path) -> Dict[str, Any]:
    """Load raw rider data from CSV and JSON files."""
    df = pd.read_csv(csv_path)
    
    with open(json_path) as f:
        meta = json.load(f)
    
    # Fix server-side timestamp offsets (e.g. world-time reference jumps)
    # This must run before lat/lng interpolation so times are consistent
    df = fix_timestamp_offsets(df)
    if 'timeInSec' in meta:
        # Apply the same correction to the JSON times for lat/lng interp
        temp = pd.DataFrame({'time_sec': meta['timeInSec'],
                             'speed_kmh': [s * 3.6 / 100 for s in meta.get('speedInCmPerSec', meta.get('timeInSec'))],
                             'distance_km': [d / 100000 for d in meta.get('distanceInCm', [0] * len(meta['timeInSec']))]})
        temp = fix_timestamp_offsets(temp)
        meta['timeInSec'] = temp['time_sec'].tolist()
    
    # Add lat/lng from JSON
    if 'latlng' in meta and len(meta['latlng']) > 0:
        json_times = meta.get('timeInSec', [])
        latlng = meta['latlng']
        
        if len(json_times) == len(latlng):
            lat_interp = interp1d(json_times, [ll[0] for ll in latlng], 
                                  fill_value='extrapolate', bounds_error=False)
            lng_interp = interp1d(json_times, [ll[1] for ll in latlng],
                                  fill_value='extrapolate', bounds_error=False)
            df['lat'] = lat_interp(df['time_sec'])
            df['lng'] = lng_interp(df['time_sec'])
    
    return {
        'data': df,
        'metadata': meta
    }


def detect_route(riders: List[Dict], data_path: Path) -> Tuple[str, Optional[str], Optional[float]]:
    """
    Detect the route from race metadata, rider data, or GPS coordinates.
    
    Returns:
        (route_name, route_slug, leadin_distance_m)
        route_slug may be None if unknown
        leadin_distance_m is the known lead-in distance in meters, or None
    """
    route_name = ""
    route_slug = None
    leadin_distance_m = None
    
    # Method 0: Check race_meta.json for route_id from event API
    meta_path = data_path / 'race_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            route_id = meta.get('route_id')
            if route_id:
                try:
                    from shared.route_lookup import get_route_info
                    route_info = get_route_info(route_id)
                    if route_info:
                        route_name = route_info['name']
                        route_slug = route_name.lower().replace(' ', '-').replace("'", "")
                        leadin_distance_m = route_info.get('leadinDistanceInMeters')
                        print(f"Detected route from event metadata: {route_name} (ID {route_id})")
                        return route_name, route_slug, leadin_distance_m
                except ImportError:
                    pass
        except Exception as e:
            print(f"Warning: Could not read race_meta.json: {e}")
    
    # Method 1: Match route name from race_name in metadata (most reliable when route name is in race title)
    if not route_name:
        meta_path = data_path / 'race_meta.json'
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                race_name = meta.get('race_name', '')
                if race_name:
                    from shared.route_lookup import load_route_cache
                    routes_cache = load_route_cache()
                    if routes_cache:
                        # Build list of route names sorted longest-first to prefer
                        # more specific matches (e.g. "Kaze Kicker" over "Kaze")
                        candidates = sorted(
                            [(info['name'], rid) for rid, info in routes_cache.items()
                             if info.get('name')],
                            key=lambda x: len(x[0]), reverse=True
                        )
                        race_name_lower = race_name.lower()
                        for rname, rid in candidates:
                            if rname.lower() in race_name_lower:
                                route_name = rname
                                leadin_distance_m = routes_cache[rid].get('leadinDistanceInMeters')
                                print(f"Detected route from race name: {route_name} (matched in '{race_name}')")
                                break
            except Exception as e:
                print(f"Warning: Route detection from race name failed: {e}")
    
    # Method 2: Check routes_cache.json for matching GPS coordinates + distance (fallback)
    if not route_name:
        try:
            from shared.route_lookup import load_route_cache
            routes_cache = load_route_cache()
            if routes_cache:
                # Use median distance of top-ranked riders for more robust matching
                import statistics
                ranked_riders = sorted(
                    [r for r in riders if r.get('metadata', {}).get('rank')],
                    key=lambda r: r['metadata']['rank']
                )
                top_riders = ranked_riders[:10] if len(ranked_riders) >= 10 else ranked_riders
                if top_riders and 'distance_km' in top_riders[0]['data'].columns:
                    rider_distances = [r['data']['distance_km'].max() * 1000 for r in top_riders]
                    rider_distance = statistics.median(rider_distances)
                    
                    sample_rider = top_riders[0]
                    if 'lat' in sample_rider['data'].columns:
                        sample_lat = sample_rider['data']['lat'].iloc[len(sample_rider['data'])//2]
                        sample_lng = sample_rider['data']['lng'].iloc[len(sample_rider['data'])//2]
                        
                        world = detect_world_from_coords(sample_lat, sample_lng)
                        if world:
                            print(f"Detected world: {world}, median rider distance (top {len(top_riders)}): {rider_distance/1000:.1f} km")
                            
                            best_match = None
                            best_diff = float('inf')
                            for route_id, info in routes_cache.items():
                                route_map = info.get('map', '').upper()
                                # Normalize world names for comparison
                                if route_map == world.upper() or (
                                    world.upper() == 'NEW_YORK' and route_map == 'NEWYORK') or (
                                    world.upper() == 'MAKURI' and route_map == 'MAKURIISLANDS'):
                                    total_dist = info['distanceInMeters'] + info['leadinDistanceInMeters']
                                    diff = abs(rider_distance - total_dist)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_match = info
                            
                            if best_match and best_diff < 3000:  # Within 3km
                                route_name = best_match['name']
                                leadin_distance_m = best_match.get('leadinDistanceInMeters')
                                print(f"Detected route from GPS + distance matching: {route_name} (diff: {best_diff:.0f}m)")
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Route detection from GPS failed: {e}")
    
    # Convert route name to slug
    if route_name:
        route_slug = route_name.lower().replace(' ', '-').replace("'", "")
    
    return route_name, route_slug, leadin_distance_m


def detect_world_from_coords(lat: float, lng: float) -> Optional[str]:
    """Detect which Zwift world based on GPS coordinates."""
    # Approximate bounding boxes for each world.
    # Some worlds use virtual coordinates (near Watopia/Makuri area).
    # Lat ranges may overlap; lng ranges disambiguate.
    worlds = {
        'LONDON': {'lat_min': 51.0, 'lat_max': 52.0, 'lng_min': -1.0, 'lng_max': 0.5},
        'WATOPIA': {'lat_min': -12.0, 'lat_max': -11.0, 'lng_min': 166.5, 'lng_max': 167.5},
        'NEW_YORK': {'lat_min': 40.5, 'lat_max': 41.0, 'lng_min': -74.5, 'lng_max': -73.5},
        'INNSBRUCK': {'lat_min': 47.0, 'lat_max': 47.5, 'lng_min': 11.0, 'lng_max': 11.6},
        'RICHMOND': {'lat_min': 37.0, 'lat_max': 38.0, 'lng_min': -78.0, 'lng_max': -77.0},
        'FRANCE': {'lat_min': -22.0, 'lat_max': -21.5, 'lng_min': 166.0, 'lng_max': 166.5},
        'PARIS': {'lat_min': 48.5, 'lat_max': 49.0, 'lng_min': 2.0, 'lng_max': 3.0},
        'MAKURI': {'lat_min': -11.0, 'lat_max': -10.6, 'lng_min': 165.5, 'lng_max': 166.0},
        'SCOTLAND': {'lat_min': 55.5, 'lat_max': 56.5, 'lng_min': -5.5, 'lng_max': -3.0},
        'YORKSHIRE': {'lat_min': 53.5, 'lat_max': 54.5, 'lng_min': -2.0, 'lng_max': -1.0},
        'BOLOGNA': {'lat_min': 44.0, 'lat_max': 45.0, 'lng_min': 11.0, 'lng_max': 11.6},
        'CRIT_CITY': {'lat_min': -10.5, 'lat_max': -10.3, 'lng_min': 165.7, 'lng_max': 165.9},
    }
    
    for world, bounds in worlds.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lng_min'] <= lng <= bounds['lng_max']):
            return world
    
    return None


def clean_race_data(
    data_dir: Path,
    cache: bool = True,
    progress_callback: Optional[callable] = None
) -> CleanedRaceData:
    """
    Load and clean all race data from a directory.
    
    Args:
        data_dir: Path to directory with race data
        cache: Whether to cache cleaned data
        progress_callback: Optional callback(step, total, message)
        
    Returns:
        CleanedRaceData object
    """
    data_path = Path(data_dir)
    cache_path = data_path / 'cleaned_cache.json'
    
    # Check cache
    if cache and cache_path.exists():
        cached = load_from_cache(cache_path)
        if cached:
            return cached
    
    # Load summary
    summary_file = data_path / 'complete_race_summary.csv'
    if summary_file.exists():
        summary = pd.read_csv(summary_file)
    else:
        summary = None
    
    # Find all rider files
    csv_files = sorted(data_path.glob('rank*_*.csv'))
    total_steps = len(csv_files) + 3  # +3 for alignment, processing, elevation
    
    if progress_callback:
        progress_callback(0, total_steps, "Loading riders...")
    
    # Load all riders
    riders = []
    for i, csv_file in enumerate(csv_files):
        json_file = csv_file.with_name(csv_file.name.replace('.csv', '_raw.json'))
        if not json_file.exists():
            continue
        
        # Parse rank from filename
        rank = int(csv_file.stem.split('_')[0].replace('rank', ''))
        
        rider_data = load_raw_rider_data(csv_file, json_file)
        
        # Get activity_id from filename
        activity_id = csv_file.stem.split('_')[1] if '_' in csv_file.stem else ''
        
        # Get name and team from summary
        name = f"Rider {rank}"
        team = ""
        weight_kg = 75.0
        player_id = None
        activity_start_time = None
        if summary is not None:
            match = summary[summary['rank'] == rank]
            if len(match) > 0:
                name = match['name'].values[0] if 'name' in match.columns else name
                team = match['team'].values[0] if 'team' in match.columns else ""
                weight_kg = float(match['weight_kg'].values[0]) if 'weight_kg' in match.columns else 75.0
                if 'player_id' in match.columns and pd.notna(match['player_id'].values[0]):
                    player_id = int(match['player_id'].values[0])
                if 'activity_start_time' in match.columns and pd.notna(match['activity_start_time'].values[0]):
                    activity_start_time = str(match['activity_start_time'].values[0])
        
        riders.append({
            'rank': rank,
            'activity_id': activity_id,
            'name': name,
            'team': team,
            'weight_kg': weight_kg,
            'player_id': player_id,
            'activity_start_time': activity_start_time,
            'data': rider_data['data'],
            'metadata': rider_data['metadata']
        })
        
        if progress_callback:
            progress_callback(i + 1, total_steps, f"Loaded {name}")
    
    if not riders:
        raise ValueError(f"No rider data found in {data_dir}")
    
    # Try to detect the route
    route_name, route_slug, leadin_distance_m = detect_route(riders, data_path)
    
    # Detect the world from GPS coordinates for surface/CRR lookup
    world = None
    ref_rider = max(riders, key=lambda r: len(r['data']))
    if 'lat' in ref_rider['data'].columns:
        sample_lat = ref_rider['data']['lat'].iloc[len(ref_rider['data']) // 2]
        sample_lng = ref_rider['data']['lng'].iloc[len(ref_rider['data']) // 2]
        world = detect_world_from_coords(sample_lat, sample_lng)
        if world:
            print(f"Detected world: {world}")
    
    if progress_callback:
        progress_callback(len(csv_files) + 1, total_steps, "Aligning distances...")
    
    # Try route-based alignment first (preferred)
    route_data = None
    finish_line = None
    loop_start_offset = None
    if route_slug:
        route_data = load_route_data(route_slug)
        if route_data:
            print(f"Using ZwiftMap route data for alignment: {route_data.route_name}")
            print(f"  Route length: {route_data.total_distance_m/1000:.2f} km, {len(route_data.distance)} points")
            riders, loop_start_offset = align_riders_to_route(riders, route_data, leadin_distance_m=leadin_distance_m)
            # Use race metadata for finish line (more reliable than route total,
            # especially on loop routes where the segment includes lead-in)
            finish_line = determine_finish_line(riders, data_path)
            if not finish_line:
                # Fallback to route total if metadata unavailable
                finish_line = route_data.total_distance_m / 1000.0
            print(f"  Finish line set to: {finish_line:.3f} km")
    
    # Fallback to landmark-based alignment if no route data
    if route_data is None:
        print("No route data available, using landmark-based alignment")
        ref_rider = max(riders, key=lambda r: len(r['data']))
        landmarks = find_course_landmarks(ref_rider['data'])
        riders, offsets = align_rider_distances(riders, landmarks)
        finish_line = determine_finish_line(riders, data_path)
    
    if progress_callback:
        progress_callback(len(csv_files) + 2, total_steps, "Processing riders...")
    
    # Find global time range
    min_time = min(r['data']['time_sec'].min() for r in riders)
    max_time = max(r['data']['time_sec'].max() for r in riders)
    
    # Process each rider
    cleaned_riders = []
    finish_times = []
    for rider in riders:
        df = rider['data']
        
        # Cut off at finish line (with 30m tolerance for alignment noise)
        # All riders from the Zwift API are finishers — detect crossing time for cooldown cutoff
        finish_time = None
        finish_tolerance_km = 0.03  # 30m tolerance
        if finish_line:
            crossed = df[df['distance_km'] >= (finish_line - finish_tolerance_km)]
            if len(crossed) > 0:
                finish_time = crossed['time_sec'].min()
                # Keep data only up to finish + a small buffer
                df = df[df['time_sec'] <= finish_time + 5].copy()
                finish_times.append(finish_time)
        
        # Compute per-point CRR from GPS surface polygons
        if world and 'lat' in df.columns and 'lng' in df.columns:
            try:
                from shared.surface_lookup import compute_crr_array
                df['crr'] = compute_crr_array(
                    df['lat'].values,
                    df['lng'].values,
                    world
                )
            except Exception as e:
                print(f"Warning: Could not compute CRR for {rider['name']}: {e}")
                df['crr'] = 0.004
        else:
            df['crr'] = 0.004
        
        # Set time_sec as index for O(1) lookups
        df = df.set_index('time_sec')
        
        cleaned_riders.append(RiderData(
            rank=rider['rank'],
            activity_id=rider['activity_id'],
            name=rider['name'],
            team=rider['team'],
            data=df,
            finish_time_sec=finish_time,
            weight_kg=rider.get('weight_kg', 75.0),
            player_id=rider.get('player_id'),
            activity_start_time=rider.get('activity_start_time'),
        ))
    
    # Cap max_time to the last finisher's time (+ buffer) so the timeline
    # doesn't extend into post-race cooldown data
    if finish_times:
        last_finish = max(finish_times)
        max_time = min(max_time, last_finish + 10)
    
    if progress_callback:
        progress_callback(len(csv_files) + 3, total_steps, "Building elevation profile...")
    
    # Build elevation profile - prefer route data if available, else use reference rider
    ref_rider = max(riders, key=lambda r: len(r['data']))
    if route_data is not None:
        if loop_start_offset is not None:
            # Loop route: rebase elevation profile to match rebased rider distances.
            # Rider distances were shifted by -start_offset, so the elevation profile
            # needs to start from start_offset on the route and wrap around.
            route_dist_m = route_data.distance
            route_alt = route_data.altitude
            route_total_m = route_data.total_distance_m
            
            # Shift distances: new_dist = (original - start_offset) mod route_total
            shifted_dist = (route_dist_m - loop_start_offset) % route_total_m
            
            # Sort by shifted distance to get a proper profile
            sort_idx = np.argsort(shifted_dist)
            elev_dist = shifted_dist[sort_idx] / 1000.0
            elev_alt = route_alt[sort_idx]
            
            # Detect lead-in: if the ref rider has off-route points (deviation=999)
            # at the start, the race has a lead-in that isn't covered by the route
            # data. Use the ref rider's telemetry altitude for that portion and
            # offset the loop data to start where the lead-in ends.
            leadin_offset_km = 0.0
            leadin_elev = None
            ref_data = ref_rider['data']
            if 'route_deviation_m' in ref_data.columns and 'altitude_m' in ref_data.columns:
                dev = ref_data['route_deviation_m'].values
                # Lead-in points have deviation=999 (set in align_riders_to_route)
                leadin_mask = dev >= 500
                if leadin_mask.any() and not leadin_mask.all():
                    # Find where lead-in ends (first on-route point)
                    first_onroute_idx = np.argmax(~leadin_mask)
                    leadin_offset_km = ref_data['distance_km'].iloc[first_onroute_idx]
                    
                    if leadin_offset_km > 0.1:  # Significant lead-in (>100m)
                        # Extract telemetry for the lead-in portion
                        leadin_data = ref_data.iloc[:first_onroute_idx]
                        leadin_dist = leadin_data['distance_km'].values
                        leadin_alt = leadin_data['altitude_m'].values
                        
                        # Resample lead-in at regular 10m intervals for smoothness
                        leadin_range = np.arange(0, leadin_offset_km, 0.01)
                        if len(leadin_range) > 0 and len(leadin_dist) > 1:
                            leadin_interp = interp1d(
                                leadin_dist, leadin_alt,
                                kind='linear', bounds_error=False,
                                fill_value=(leadin_alt[0], leadin_alt[-1])
                            )
                            leadin_elev = pd.DataFrame({
                                'distance_km': leadin_range,
                                'altitude_m': leadin_interp(leadin_range)
                            })
                            print(f"  Lead-in elevation: {leadin_offset_km:.2f} km from rider telemetry "
                                  f"(alt {leadin_alt.min():.0f}-{leadin_alt.max():.0f}m)")
            
            # Offset loop elevation to start after the lead-in
            elev_dist = elev_dist + leadin_offset_km
            
            # Extend to cover full race distance (riders may go beyond one loop)
            max_rider_dist = max(r['data']['distance_km'].max() for r in riders if len(r['data']) > 0)
            if max_rider_dist > elev_dist[-1]:
                n_extra_laps = int(np.ceil((max_rider_dist - elev_dist[-1]) / (route_total_m / 1000.0)))
                all_dist = [elev_dist]
                all_alt = [elev_alt]
                for lap in range(1, n_extra_laps + 1):
                    all_dist.append(elev_dist + lap * route_total_m / 1000.0)
                    all_alt.append(elev_alt)
                elev_dist = np.concatenate(all_dist)
                elev_alt = np.concatenate(all_alt)
            
            # Combine lead-in telemetry + loop route data
            if leadin_elev is not None:
                loop_profile = pd.DataFrame({
                    'distance_km': elev_dist,
                    'altitude_m': elev_alt
                })
                elevation_profile = pd.concat([leadin_elev, loop_profile], ignore_index=True)
            else:
                elevation_profile = pd.DataFrame({
                    'distance_km': elev_dist,
                    'altitude_m': elev_alt
                })
        else:
            elevation_profile = pd.DataFrame({
                'distance_km': route_data.distance / 1000.0,
                'altitude_m': route_data.altitude
            })
    else:
        elevation_profile = build_elevation_profile(ref_rider['data'])
    
    # Extract source activity ID from race metadata
    source_activity_id = None
    meta_path = data_path / 'race_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            source_activity_id = str(meta.get('source_activity_id', '')) or None
        except Exception:
            pass
    
    result = CleanedRaceData(
        race_id=data_path.name,
        route_name=route_name,
        finish_line_km=finish_line or 0.0,
        riders=cleaned_riders,
        elevation_profile=elevation_profile,
        min_time=min_time,
        max_time=max_time,
        source_activity_id=source_activity_id,
        route_slug=route_slug,
        world=world
    )
    
    # Cache if requested
    if cache:
        save_to_cache(result, cache_path)
    
    return result


def determine_finish_line(riders: List[Dict], data_path: Path) -> Optional[float]:
    """Determine the finish line distance."""
    # Try to get from race_meta.json route_id
    meta_path = data_path / 'race_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            
            # Use segment distance saved from subgroupResults on the source activity
            segment_dist_cm = meta.get('segment_distance_cm')
            if segment_dist_cm:
                finish = segment_dist_cm / 100000.0
                print(f"Finish line from race segment distance: {finish:.2f} km")
                return finish
            
            # Fall back to route_id from event API
            route_id = meta.get('route_id')
            if route_id:
                from shared.route_lookup import get_total_race_distance
                finish = get_total_race_distance(route_id)
                if finish:
                    print(f"Finish line from route data: {finish:.2f} km")
                    return finish
        except Exception as e:
            print(f"Warning: Could not get finish line from meta data: {e}")
    
    # Try from metadata segmentDistanceInCentimeters
    for rider in riders:
        if rider['metadata']:
            subgroup_results = rider['metadata'].get('subgroupResults', [])
            if subgroup_results:
                segment_dist_cm = subgroup_results[0].get('segmentDistanceInCentimeters')
                if segment_dist_cm:
                    return segment_dist_cm / 100000.0
    
    # Fallback: use top-ranked riders' max distances (they definitely finished)
    # Sort by rank and use the top finishers for a reliable estimate
    sorted_riders = sorted(riders, key=lambda r: r['rank'])
    top_riders = sorted_riders[:min(5, len(sorted_riders))]
    finish_distances = [r['data']['distance_km'].max() for r in top_riders]
    finish = float(np.median(finish_distances))
    print(f"Finish line estimated from top {len(top_riders)} riders' max distances: {finish:.2f} km")
    return finish


def build_elevation_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build a smooth elevation profile from rider data."""
    # Sample at regular distance intervals
    dist_range = np.arange(0, df['distance_km'].max(), 0.01)  # Every 10m
    
    if 'altitude_m' not in df.columns:
        return pd.DataFrame({'distance_km': dist_range, 'altitude_m': 0})
    
    interp = interp1d(
        df['distance_km'].values,
        df['altitude_m'].values,
        kind='linear',
        bounds_error=False,
        fill_value=(df['altitude_m'].iloc[0], df['altitude_m'].iloc[-1])
    )
    
    return pd.DataFrame({
        'distance_km': dist_range,
        'altitude_m': interp(dist_range)
    })


def save_to_cache(data: CleanedRaceData, cache_path: Path):
    """Save cleaned data to cache."""
    cache_dir = cache_path.parent / 'cleaned_data'
    cache_dir.mkdir(exist_ok=True)
    
    # Helper to convert numpy types to Python types
    def to_python(val):
        if hasattr(val, 'item'):
            return val.item()
        return val
    
    # Save each rider's data
    riders_info = []
    for rider in data.riders:
        rider_file = cache_dir / f'rider_{rider.rank}.parquet'
        rider.data.to_parquet(rider_file)
        
        riders_info.append({
            'rank': int(rider.rank),
            'activity_id': str(rider.activity_id),
            'name': str(rider.name) if rider.name else "",
            'team': str(rider.team) if rider.team else "",
            'weight_kg': float(rider.weight_kg),
            'player_id': int(rider.player_id) if rider.player_id else None,
            'activity_start_time': rider.activity_start_time,
            'finish_time_sec': to_python(rider.finish_time_sec) if rider.finish_time_sec is not None else None,
            'data_file': rider_file.name
        })
    
    # Save elevation profile
    elevation_file = cache_dir / 'elevation.parquet'
    data.elevation_profile.to_parquet(elevation_file)
    
    # Save metadata
    meta = {
        'cleaning_version': CLEANING_VERSION,
        'race_id': str(data.race_id),
        'route_name': str(data.route_name) if data.route_name else "",
        'route_slug': str(data.route_slug) if data.route_slug else None,
        'finish_line_km': float(data.finish_line_km),
        'min_time': float(data.min_time),
        'max_time': float(data.max_time),
        'source_activity_id': str(data.source_activity_id) if data.source_activity_id else None,
        'world': data.world,
        'riders': riders_info
    }
    
    with open(cache_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_from_cache(cache_path: Path) -> Optional[CleanedRaceData]:
    """Load cleaned data from cache."""
    try:
        cache_dir = cache_path.parent / 'cleaned_data'
        
        with open(cache_path) as f:
            meta = json.load(f)
        
        # Check cleaning version — discard stale caches
        cached_version = meta.get('cleaning_version', 0)
        if cached_version != CLEANING_VERSION:
            print(f"Cache version mismatch (cached={cached_version}, current={CLEANING_VERSION}) — re-cleaning")
            return None
        
        # Load riders
        riders = []
        for rider_info in meta['riders']:
            rider_file = cache_dir / rider_info['data_file']
            if not rider_file.exists():
                return None
            
            data = pd.read_parquet(rider_file)
            # Set time_sec as index for O(1) lookups (if not already indexed)
            if 'time_sec' in data.columns:
                data = data.set_index('time_sec')
            riders.append(RiderData(
                rank=rider_info['rank'],
                activity_id=rider_info['activity_id'],
                name=rider_info['name'],
                team=rider_info['team'],
                data=data,
                finish_time_sec=rider_info['finish_time_sec'],
                weight_kg=rider_info.get('weight_kg', 75.0),
                player_id=rider_info.get('player_id'),
                activity_start_time=rider_info.get('activity_start_time'),
            ))
        
        # Load elevation
        elevation_file = cache_dir / 'elevation.parquet'
        if not elevation_file.exists():
            return None
        elevation = pd.read_parquet(elevation_file)
        
        return CleanedRaceData(
            race_id=meta['race_id'],
            route_name=meta['route_name'],
            finish_line_km=meta['finish_line_km'],
            riders=riders,
            elevation_profile=elevation,
            min_time=meta['min_time'],
            max_time=meta['max_time'],
            source_activity_id=meta.get('source_activity_id'),
            route_slug=meta.get('route_slug'),
            world=meta.get('world')
        )
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None
