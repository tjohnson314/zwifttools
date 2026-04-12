import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

const CATEGORIES = ['A', 'B', 'C', 'D', 'E'];
const CAT_POINTS = [3, 2, 1]; // 1st, 2nd, 3rd

// Team tag patterns (case-insensitive)
// Flexible matching: brackets/parens optional, common abbreviations accepted
const TEAM_PATTERNS = [
    {tag: 'RtB', re: /[\[\(]?\s*r\.?t\.?b\.?\s*[\]\)]?/i},
    {tag: 'RtB', re: /[\[\(]?\s*road\s*to\s*bonk\s*[\]\)]?/i},
    {tag: 'Fellowship', re: /[\[\(]?\s*fellowship\s*[\]\)]?/i},
    {tag: 'Fellowship', re: /[\[\(]?\s*fell\s*[\]\)]?/i},
];

// Subgroup cache: eventSubgroupId -> category letter
const subgroupCache = new Map();

// Persistent rider cache — accumulates riders across nearby updates so we
// don't lose data on riders who drift out of the Zwift client's view range.
const riderCache = new Map(); // athleteId -> rider object

let lastRefresh = 0;
const REFRESH_INTERVAL = 2000;

// DOM refs
let noDataEl, scoreboardEl, scoreRtbEl, scoreFellowshipEl,
    barRtbEl, barFellowshipEl, categoriesEl, bonusListEl, riderCountEl;


export function main() {
    noDataEl = document.querySelector('.no-data');
    scoreboardEl = document.querySelector('.scoreboard');
    scoreRtbEl = document.getElementById('score-rtb');
    scoreFellowshipEl = document.getElementById('score-fellowship');
    barRtbEl = document.getElementById('bar-rtb');
    barFellowshipEl = document.getElementById('bar-fellowship');
    categoriesEl = document.getElementById('categories');
    bonusListEl = document.getElementById('bonus-list');
    riderCountEl = document.getElementById('rider-count');

    Common.subscribe('nearby', onNearby);
}


function parseTeam(name) {
    if (!name) return null;
    for (const {tag, re} of TEAM_PATTERNS) {
        if (re.test(name)) return tag;
    }
    return null;
}


function cleanName(name) {
    if (!name) return '?';
    return name.trim();
}


function getSubgroupLabel(eventSubgroupId) {
    if (!eventSubgroupId) return null;
    if (subgroupCache.has(eventSubgroupId)) {
        return subgroupCache.get(eventSubgroupId);
    }
    const sg = Common.getEventSubgroup(eventSubgroupId);
    if (sg && !(sg instanceof Promise)) {
        const label = sg.subgroupLabel || null;
        subgroupCache.set(eventSubgroupId, label);
        return label;
    }
    return null;
}


// Compute a human-readable gap between two riders using their gap values
// (which are time offsets in seconds relative to the watching athlete).
function computeGap(leader, rival) {
    // gap values are relative to the watched rider, so the difference gives
    // the inter-rider gap. Both should be non-null for a meaningful result.
    if (leader.gap != null && rival.gap != null) {
        const delta = Math.abs(rival.gap - leader.gap);
        if (delta < 60) {
            return `${delta.toFixed(1)}s`;
        }
        const mins = Math.floor(delta / 60);
        const secs = Math.round(delta % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    // Fallback to distance gap
    if (leader.distance && rival.distance) {
        const delta = Math.abs(leader.distance - rival.distance);
        if (delta < 1000) {
            return `${Math.round(delta)}m`;
        }
        return `${(delta / 1000).toFixed(1)}km`;
    }
    return null;
}


function onNearby(data) {
    const now = Date.now();
    if (now - lastRefresh < REFRESH_INTERVAL) return;
    lastRefresh = now;

    if (!data || !data.length) {
        // Still render from cache if we have accumulated riders
        if (riderCache.size > 0) {
            const scores = calculateScores([...riderCache.values()]);
            render(scores, riderCache.size);
            return;
        }
        noDataEl.classList.remove('hidden');
        scoreboardEl.classList.add('hidden');
        noDataEl.textContent = 'Waiting for race data...';
        return;
    }

    // Process current nearby data into cache
    for (const entry of data) {
        const name = entry.athlete?.sanitizedFullname || '';
        const team = parseTeam(name);
        if (!team) continue;

        // Skip riders in the pen / not moving (speed ≤ 0)
        const speed = entry.state?.speed;
        if (!speed || speed <= 0) continue;

        const category = getSubgroupLabel(entry.state?.eventSubgroupId);
        const weight = entry.athlete?.weight || 0;
        const peak15 = entry.stats?.power?.peaks?.[15]?.avg || 0;
        const peak1200 = entry.stats?.power?.peaks?.[1200]?.avg || 0;

        const powerAvg = entry.stats?.power?.avg || 0;

        const rider = {
            athleteId: entry.athleteId,
            name: cleanName(name),
            team,
            category,
            eventPosition: entry.eventPosition || Infinity,
            gap: entry.gap,
            gapDistance: entry.gapDistance,
            distance: entry.state?.distance || 0,
            hrAvg: entry.stats?.hr?.avg || 0,
            peak15Wkg: weight > 0 ? peak15 / weight : 0,
            peak1200Wkg: weight > 0 ? peak1200 / weight : 0,
            powerAvgWkg: weight > 0 ? powerAvg / weight : 0,
            weight,
            lastSeen: now,
        };

        // Always overwrite with fresher data
        riderCache.set(rider.athleteId, rider);
    }

    if (riderCache.size === 0) {
        noDataEl.classList.remove('hidden');
        scoreboardEl.classList.add('hidden');
        noDataEl.textContent = 'No [RtB] or [Fellowship] riders found nearby';
        return;
    }

    const scores = calculateScores([...riderCache.values()]);
    render(scores, riderCache.size);
}


function calculateScores(riders) {
    const teamScores = {RtB: 0, Fellowship: 0};
    const catResults = {};
    const bonuses = [];

    // Per-category: group riders, sort by event position, award 3/2/1
    for (const cat of CATEGORIES) {
        const catRiders = riders
            .filter(r => r.category === cat)
            .sort((a, b) => {
                if (a.eventPosition !== b.eventPosition) {
                    return a.eventPosition - b.eventPosition;
                }
                return b.distance - a.distance;
            });

        const top3 = catRiders.slice(0, 3).map((r, i) => {
            const points = CAT_POINTS[i] || 0;
            teamScores[r.team] += points;
            // Find gap to next rider on the opposite team
            const rival = catRiders.find(x => x.team !== r.team &&
                (x.eventPosition > r.eventPosition ||
                 (x.eventPosition === r.eventPosition && x.distance < r.distance)));
            const rivalGap = rival ? computeGap(r, rival) : null;
            return {...r, points, position: i + 1, rivalGap};
        });

        catResults[cat] = {riders: top3, total: catRiders.length};
    }

    // Overall winner (best event position across all riders)
    const allSorted = [...riders].sort((a, b) => {
        if (a.eventPosition !== b.eventPosition) {
            return a.eventPosition - b.eventPosition;
        }
        return b.distance - a.distance;
    });

    if (allSorted.length > 0 && allSorted[0].eventPosition !== Infinity) {
        const winner = allSorted[0];
        const rival = allSorted.find(x => x.team !== winner.team);
        const rivalGap = rival ? computeGap(winner, rival) : null;
        teamScores[winner.team] += 2;
        bonuses.push({
            label: 'Overall Winner',
            icon: 'trophy',
            rider: winner,
            value: rivalGap ? `+${rivalGap}` : '',
            points: 2,
        });
    } else {
        bonuses.push({label: 'Overall Winner', icon: 'trophy', rider: null, value: 'pending', points: 2});
    }

    // Highest average HR
    const hrCandidates = riders.filter(r => r.hrAvg > 0);
    if (hrCandidates.length > 0) {
        const best = hrCandidates.reduce((a, b) => b.hrAvg > a.hrAvg ? b : a);
        teamScores[best.team] += 1;
        bonuses.push({
            label: 'Highest Avg HR',
            icon: 'ecg_heart',
            rider: best,
            value: `${Math.round(best.hrAvg)} bpm`,
            points: 1,
        });
    } else {
        bonuses.push({label: 'Highest Avg HR', icon: 'ecg_heart', rider: null, value: 'pending', points: 1});
    }

    // Highest 20-min peak power (W/kg)
    const p1200Candidates = riders.filter(r => r.peak1200Wkg > 0);
    if (p1200Candidates.length > 0) {
        const best = p1200Candidates.reduce((a, b) => b.peak1200Wkg > a.peak1200Wkg ? b : a);
        teamScores[best.team] += 1;
        bonuses.push({
            label: 'Best 20min W/kg',
            icon: 'bolt',
            rider: best,
            value: `${best.peak1200Wkg.toFixed(2)} W/kg`,
            points: 1,
        });
    } else {
        // Fallback: show highest average power so far with a note
        const avgCandidates = riders.filter(r => r.powerAvgWkg > 0);
        if (avgCandidates.length > 0) {
            const best = avgCandidates.reduce((a, b) => b.powerAvgWkg > a.powerAvgWkg ? b : a);
            bonuses.push({
                label: 'Best 20min W/kg',
                icon: 'bolt',
                rider: best,
                value: `${best.powerAvgWkg.toFixed(2)} avg*`,
                note: '* avg power — race < 20 min',
                points: 1,
            });
        } else {
            bonuses.push({label: 'Best 20min W/kg', icon: 'bolt', rider: null, value: '< 20 min', points: 1});
        }
    }

    // Highest 15-sec peak power (W/kg)
    const p15Candidates = riders.filter(r => r.peak15Wkg > 0);
    if (p15Candidates.length > 0) {
        const best = p15Candidates.reduce((a, b) => b.peak15Wkg > a.peak15Wkg ? b : a);
        teamScores[best.team] += 1;
        bonuses.push({
            label: 'Best 15s W/kg',
            icon: 'offline_bolt',
            rider: best,
            value: `${best.peak15Wkg.toFixed(2)} W/kg`,
            points: 1,
        });
    } else {
        bonuses.push({label: 'Best 15s W/kg', icon: 'offline_bolt', rider: null, value: 'no data yet', points: 1});
    }

    return {teamScores, catResults, bonuses};
}


function render(scores, totalRiders) {
    noDataEl.classList.add('hidden');
    scoreboardEl.classList.remove('hidden');

    const {teamScores, catResults, bonuses} = scores;

    // Total scores
    scoreRtbEl.textContent = teamScores.RtB;
    scoreFellowshipEl.textContent = teamScores.Fellowship;

    // Score bar
    const total = teamScores.RtB + teamScores.Fellowship || 1;
    barRtbEl.style.width = `${(teamScores.RtB / 35) * 100}%`;
    barFellowshipEl.style.width = `${(teamScores.Fellowship / 35) * 100}%`;

    // Categories
    renderCategories(catResults);

    // Bonuses
    renderBonuses(bonuses);

    // Rider count
    riderCountEl.textContent = totalRiders;
}


const CAT_HUES = {A: 0, B: 90, C: 180, D: 60, E: 260};

function renderCategories(catResults) {
    let html = '';
    for (const cat of CATEGORIES) {
        const {riders, total} = catResults[cat];
        const hue = CAT_HUES[cat];

        html += `<div class="category">`;
        html += `<div class="cat-header">`;
        html += `<span class="cat-badge" style="--hue: ${hue}deg;">${cat}</span>`;
        html += `<span class="cat-count">${total} riders</span>`;
        html += `</div>`;

        if (riders.length === 0) {
            html += `<div class="cat-empty">No riders yet</div>`;
        } else {
            for (const r of riders) {
                const teamClass = r.team === 'RtB' ? 'team-rtb' : 'team-fellowship';
                html += `<div class="cat-rider">`;
                html += `<span class="rider-pos">${r.position}.</span>`;
                html += `<span class="rider-name ${teamClass}">${esc(r.name)}</span>`;
                if (r.rivalGap) {
                    html += `<span class="rider-gap">+${esc(r.rivalGap)}</span>`;
                }
                html += `<span class="rider-team-tag ${teamClass}">${r.team === 'RtB' ? 'RtB' : 'Fell'}</span>`;
                html += `<span class="rider-points">+${r.points}</span>`;
                html += `</div>`;
            }
        }
        html += `</div>`;
    }
    categoriesEl.innerHTML = html;
}


function renderBonuses(bonuses) {
    let html = '';
    for (const b of bonuses) {
        html += `<div class="bonus-row">`;
        html += `<span class="bonus-icon"><ms>${b.icon}</ms></span>`;
        html += `<span class="bonus-label">${esc(b.label)}</span>`;
        if (b.rider) {
            const teamClass = b.rider.team === 'RtB' ? 'team-rtb' : 'team-fellowship';
            html += `<span class="bonus-rider ${teamClass}">${esc(b.rider.name)}</span>`;
            if (b.value) {
                html += `<span class="bonus-value">${esc(b.value)}</span>`;
            }
            html += `<span class="bonus-points">+${b.points}</span>`;
        } else {
            html += `<span class="bonus-pending">${esc(b.value)}</span>`;
            html += `<span class="bonus-points dimmed">+${b.points}</span>`;
        }
        if (b.note) {
            html += `<div class="bonus-note">${esc(b.note)}</div>`;
        }
        html += `</div>`;
    }
    bonusListEl.innerHTML = html;
}


function esc(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
