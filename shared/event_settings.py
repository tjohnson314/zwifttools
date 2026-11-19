"""Readable event settings, with raw codes retained for unknown values."""

import json
import re
from datetime import timezone
from decimal import Decimal, InvalidOperation

from shared.route_lookup import get_route_info
from shared.world_config import WORLD_NAMES
from shared.datetime_utils import parse_iso_datetime


POWERUPS = {
    0: "Feather",
    1: "Draft",
    2: "Small XP",
    3: "Large XP",
    4: "Burrito",
    5: "Aero",
    6: "Ghost",
    7: "Steamroller",
    8: "Anvil",
    9: "Coffee Stop",
    10: "Speed Boost",
    15: "None",
}

RULES = {
    "NO_POWERUPS": "Powerups disabled",
    "NO_DRAFTING": "Drafting disabled",
    "NO_TT_BIKES": "Time-trial bikes not allowed",
    "LADIES_ONLY": "Women only",
    "MEN_ONLY": "Men only",
    "SHOW_RACE_RESULTS": "Race results shown",
    "ALLOWS_LATE_JOIN": "Late joining allowed",
    "RUBBERBANDING": "Rubberbanding enabled",
    "ENFORCE_NO_ZPOWER": "zPower / estimated power not allowed",
    "ONLY_ZPOWER": "zPower riders only",
    "ENFORCE_HRM": "Heart-rate monitor required",
}


def _display(value):
    if value is None or value == "":
        return "Not specified"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _time(value):
    if not value:
        return "Not specified"
    try:
        parsed = parse_iso_datetime(value)
    except (TypeError, ValueError):
        return f"Unrecognized time: {_display(value)}"
    if parsed.tzinfo is None:
        return f"{value} (time zone not specified)"
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _quantity(value, unit):
    return "Not specified" if value is None else f"{_display(value)} {unit}"


def _powerup(value):
    if not re.fullmatch(r"-?[0-9]+", value):
        return f"Unknown powerup (ID {value})"
    return f"{POWERUPS.get(int(value), 'Unknown powerup')} (ID {value})"


def _distribution(value):
    parts = value.strip('"').split(",")
    if len(parts) % 2:
        return f"Unrecognized distribution: {value}"
    awards = []
    for powerup_id, percent in zip(parts[::2], parts[1::2]):
        try:
            valid = Decimal(percent).is_finite()
        except InvalidOperation:
            valid = False
        if not valid:
            return f"Unrecognized distribution: {value}"
        awards.append(f"{_powerup(powerup_id.strip())}: {percent.strip()}%")
    return "; ".join(awards)


def _arch_powerup_pairs(value):
    parts = [part.strip() for part in value.strip('"').split(",")]
    if len(parts) % 2 or not all(
        re.fullmatch(r"-?[0-9]+", arch_id) for arch_id in parts[::2]
    ):
        return None
    return list(zip(parts[::2], parts[1::2]))


def _arch_powerup_rows(value):
    pairs = _arch_powerup_pairs(value)
    if pairs is None:
        return [("Arch-specific powerups", f"Unrecognized arch override: {value}")]
    return [
        (
            f"Arch {arch_id}" + (" (including arches without an ID)" if int(arch_id) == 0 else ""),
            _powerup(powerup_id),
        )
        for arch_id, powerup_id in pairs
    ]


def _powerup_rows(tags):
    rows = []
    for tag in tags:
        key, separator, value = tag.partition("=")
        if key == "starting_powerup" and separator:
            rows.append(("Starting powerup", _powerup(value.strip('"'))))
        elif key == "arch_powerup" and separator:
            rows.extend(_arch_powerup_rows(value))
        elif key == "powerup_percent" and separator:
            rows.append(("Fallback arch award distribution", _distribution(value)))
    if not any(tag.startswith("starting_powerup=") for tag in tags):
        rows.append(("Starting powerup", "Not specified in these tags"))
    if not any(tag.startswith("arch_powerup=") for tag in tags):
        rows.append(("Arch-specific powerups", "Not specified in these tags"))
    if not any(tag.startswith("powerup_percent=") for tag in tags):
        rows.append(("Fallback arch award distribution", "Not specified in these tags"))
    return rows


def _rule_labels(rules):
    return [f"{RULES.get(rule, 'Unknown rule')} ({rule})" for rule in rules]


def _pen_summary(event, pen):
    from shared.event_route import build_event_route

    def value(key):
        return pen[key] if key in pen else event.get(key)

    def rows(fields):
        return [(label, _display(value(key))) for key, label in fields]

    map_id = value("mapId")
    route_id = value("routeId")
    route = get_route_info(route_id)
    course = [
        ("World", f"{WORLD_NAMES.get(map_id, 'Unknown world')} (mapId {_display(map_id)})"),
        ("Route", f"{route['name'] if route else 'Unknown route'} (ID {_display(route_id)})"),
        ("Laps", _display(value("laps"))),
        ("Distance setting", _quantity(value("distanceInMeters"), "m")),
        ("Duration setting", _quantity(value("durationInSeconds"), "seconds")),
    ]
    if route:
        course.extend([
            ("Route lap distance (local catalog)", f"{route['distanceInMeters']} m"),
            ("Route lead-in (local catalog)", f"{route['leadinDistanceInMeters']} m"),
            ("Route ascent per lap (local catalog)", f"{route['ascentInMeters']} m"),
        ])
    event_tags = event.get("tags") or []
    pen_tags = pen.get("tags") or []
    tag_sets = [("Event tags", event_tags)]
    if "tags" in pen and pen_tags != event_tags:
        tag_sets.append(("Pen tags", pen_tags))
    elif "tags" in pen:
        tag_sets = [("Event and pen tags (identical)", pen_tags)]
    powerup_tag_sets = [
        (title, [tag for tag in tags if tag.startswith(
            ("starting_powerup=", "arch_powerup=", "powerup_percent=")
        )])
        for title, tags in tag_sets
    ]
    if "tags" in pen and (
        len(powerup_tag_sets) == 1
        or sorted(powerup_tag_sets[0][1]) == sorted(powerup_tag_sets[1][1])
    ):
        powerup_tag_sets = [("Event and pen powerup tags (identical)", powerup_tag_sets[0][1])]
    powerups = [{"title": title, "rows": _powerup_rows(tags)} for title, tags in powerup_tag_sets]
    all_rules = (event.get("rulesSet") or []) + (pen.get("rulesSet") or [])
    powerup_note = (
        "Powerups disabled: NO_POWERUPS is present in the event or pen rules. "
        "The powerup tags are retained for inspection but do not apply."
        if "NO_POWERUPS" in all_rules else
        "No NO_POWERUPS rule is present in the supplied event or pen rules. "
    )
    if "NO_POWERUPS" not in all_rules and not any(
        tag.startswith("powerup_percent=") for tag in event_tags + pen_tags
    ):
        powerup_note += (
            " Default for arches without a matching override: random Feather, "
            "Draft, or Aero."
        )
    sections = [
        {"title": "Event and pen", "rows": [
            ("Name", _display(value("name"))),
            ("Event type", _display(event.get("eventType") or event.get("type")).removeprefix("EVENT_TYPE_").replace("_", " ").title()),
            ("Sport", _display(event.get("sport")).replace("_", " ").title()),
            ("Subgroup ID", _display(pen.get("id"))),
            ("Rules ID", _display(value("rulesId"))),
            ("Description", _display(value("description"))),
        ]},
        {"title": "Course", "rows": course,
         "note": "Zero distance/duration settings do not mean a zero-length race; a lap-based event uses its route. Catalog measurements are separate from event overrides."},
        {"title": "Schedule (UTC)", "rows": [
            ("Event start", _time(event.get("eventStart"))),
            ("Pen start", _time(value("eventSubgroupStart") if pen else event.get("eventStart"))),
            *[(label, _time(value(key))) for key, label in [
                ("registrationStart", "Registration opens"),
                ("registrationEnd", "Registration closes"),
                ("lineUpStart", "Pen opens"),
                ("lineUpEnd", "Pen closes"),
            ]],
            ("Late-join setting", _quantity(value("lateJoinInMinutes"), "minutes")),
        ]},
        {"title": "Riders and entry", "rows": rows([
            ("totalEntrantCount", "Entrants"),
            ("totalSignedUpCount", "Signed up"),
            ("totalJoinedCount", "Joined"),
            ("fieldLimit", "Field limit"),
            ("rangeAccessLabel", "Entry range"),
            ("categoryEnforcement", "Category enforcement"),
            ("startLocation", "Start location ID"),
            ("paceType", "Pace type ID"),
            ("fromPaceValue", "Pace range minimum"),
            ("toPaceValue", "Pace range maximum"),
        ])},
        {"title": "Equipment and event options", "rows": rows([
            ("bikeHash", "Bike restriction hash"),
            ("jerseyHash", "Jersey hash"),
            ("workoutHash", "Workout hash"),
            ("overrideMapPreferences", "Override map preferences"),
            ("timeTrialOptions", "Time-trial options"),
            ("accessRules", "Access rules"),
            ("eventPointsConfigs", "Points configuration"),
            ("invitedLeaders", "Invited leaders"),
            ("invitedSweepers", "Invited sweepers"),
        ])},
        {"title": "Service-account state", "rows": rows([
            ("signedUp", "Service account signed up"),
            ("registered", "Service account registered"),
            ("signupStatus", "Signup status code"),
            ("registrationStatus", "Registration status code"),
            ("followeeEntrantCount", "Followed-rider entrants"),
            ("followeeSignedUpCount", "Followed riders signed up"),
            ("followeeJoinedCount", "Followed riders joined"),
        ]), "note": "These account-relative fields refer to the site's Zwift service account, not the active rider."},
    ]
    return {
        "label": _display(pen.get("subgroupLabel") or pen.get("label")) if pen else "Event",
        "sections": sections,
        "event_rules": _rule_labels(event.get("rulesSet") or []),
        "pen_rules": _rule_labels(pen.get("rulesSet") or []),
        "identical_rules": "rulesSet" in pen and sorted(event.get("rulesSet") or []) == sorted(pen.get("rulesSet") or []),
        "has_pen": bool(pen),
        "powerups": powerups,
        "powerup_note": powerup_note,
        "route_preview": build_event_route(map_id, route_id, powerup_tag_sets, all_rules),
        "controls": [
            {"title": title, "values": [
                meaning for tag, meaning in [
                    ("showplacements", "Rider placements shown"),
                    ("steering_disabled", "Steering disabled"),
                    ("nobraking", "Braking disabled"),
                ] if tag in tags
            ]} for title, tags in tag_sets
        ],
        "tags": [
            {"title": title, "values": [
                {"raw": tag, "meaning": {
                    "showplacements": "Show rider placements",
                    "steering_disabled": "Steering disabled",
                    "nobraking": "Braking disabled",
                }.get(tag)} for tag in tags
            ]} for title, tags in tag_sets
        ],
    }


def summarize_event(event):
    pens = event.get("eventSubgroups") or []
    ordered_pens = sorted(pens, key=lambda pen: str(pen.get("subgroupLabel") or pen.get("label") or ""))
    visibility = {
        "CULLING_SUBGROUP_ONLY": "Only riders in the same pen (CULLING_SUBGROUP_ONLY)",
        "CULLING_EVENT_ONLY": "Only riders in the same event, across all pens (CULLING_EVENT_ONLY)",
    }.get(event.get("cullingType"), _display(event.get("cullingType")))
    return {
        "name": _display(event.get("name")),
        "id": _display(event.get("id")),
        "description": _display(event.get("description")),
        "settings": [
            ("Event start", _time(event.get("eventStart"))),
            ("Rider visibility mode", visibility),
        ] + [(label, _display(event.get(key))) for key, label in [
            ("totalEntrantCount", "Total entrants"),
            ("totalJoinedCount", "Total joined"),
            ("privateEvent", "Private event"),
            ("unlisted", "Unlisted"),
            ("visible", "Visible"),
            ("restricted", "Restricted"),
            ("invisibleToNonParticipants", "Hidden from non-participants in-game"),
            ("categoryEnforcement", "Category enforcement"),
            ("worldId", "Zwift worldId (raw; course world uses mapId)"),
        ]],
        "pens": [_pen_summary(event, pen) for pen in ordered_pens] if pens else [_pen_summary(event, {})],
    }
