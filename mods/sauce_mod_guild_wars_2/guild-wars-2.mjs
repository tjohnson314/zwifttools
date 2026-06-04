import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

const CATEGORIES = ['A', 'B', 'C', 'D', 'E'];
const CAT_POINTS = [3, 2, 1];
const TOTAL_POINTS = 49;

// Update this list to match the communities in your race.
const COMMUNITY_DEFS = [
    {
        key: 'RtB',
        short: 'RtB',
        patterns: [
            /[\[\(]?\s*r\.?t\.?b\.?\s*[\]\)]?/i,
            /[\[\(]?\s*road\s*to\s*bonk\s*[\]\)]?/i,
        ],
    },
    {
        key: 'Fellowship',
        short: 'Fell',
        patterns: [
            /[\[\(]?\s*fellowship\s*[\]\)]?/i,
            /[\[\(]?\s*fell\s*[\]\)]?/i,
        ],
    },
];

const COMMUNITY_KEYS = COMMUNITY_DEFS.map(x => x.key);
const subgroupCache = new Map();
const riderCache = new Map();
let enabledCategories = new Set(CATEGORIES);

let lastRefresh = 0;
const REFRESH_INTERVAL = 2000;

let noDataEl;
let scoreboardEl;
let teamLeftNameEl;
let teamRightNameEl;
let scoreLeftEl;
let scoreRightEl;
let barLeftEl;
let barRightEl;
let scoreTotalEl;
let categoriesEl;
let bonusListEl;
let riderCountEl;


export function main() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({enabledCategories: [...CATEGORIES]});
    enabledCategories = getEnabledCategories();

    noDataEl = document.querySelector('.no-data');
    scoreboardEl = document.querySelector('.scoreboard');
    teamLeftNameEl = document.getElementById('team-left-name');
    teamRightNameEl = document.getElementById('team-right-name');
    scoreLeftEl = document.getElementById('score-left');
    scoreRightEl = document.getElementById('score-right');
    barLeftEl = document.getElementById('bar-left');
    barRightEl = document.getElementById('bar-right');
    scoreTotalEl = document.getElementById('score-total');
    categoriesEl = document.getElementById('categories');
    bonusListEl = document.getElementById('bonus-list');
    riderCountEl = document.getElementById('rider-count');

    teamLeftNameEl.textContent = COMMUNITY_DEFS[0].key;
    teamRightNameEl.textContent = COMMUNITY_DEFS[1].key;

    Common.settingsStore.addEventListener('changed', () => {
        enabledCategories = getEnabledCategories();
        if (riderCache.size > 0) {
            render(calculateScores([...riderCache.values()]), riderCache.size);
        }
    });

    Common.subscribe('nearby', onNearby);
}


function getEnabledCategories() {
    const configured = Common.settingsStore.get('enabledCategories');
    const source = Array.isArray(configured) ? configured : CATEGORIES;
    const filtered = source.filter(x => CATEGORIES.includes(x));
    return new Set(filtered);
}


function parseCommunity(name) {
    if (!name) {
        return null;
    }
    for (const def of COMMUNITY_DEFS) {
        for (const pattern of def.patterns) {
            if (pattern.test(name)) {
                return def.key;
            }
        }
    }
    return null;
}


function getRiderName(athlete) {
    if (!athlete) {
        return '';
    }
    return athlete.fullname || athlete.sanitizedFullname ||
        [athlete.firstName, athlete.lastName].filter(x => x).join(' ');
}


function detectCommunity(athlete) {
    const fromName = parseCommunity(getRiderName(athlete));
    if (fromName) {
        return fromName;
    }

    const fromTeam = parseCommunity(athlete?.team || '');
    if (fromTeam) {
        return fromTeam;
    }

    return null;
}


function getCommunityShort(key) {
    const def = COMMUNITY_DEFS.find(x => x.key === key);
    return def ? def.short : key;
}


function getCommunityClass(key) {
    return key === COMMUNITY_DEFS[0].key ? 'team-left' : 'team-right';
}


function cleanName(name) {
    return name ? name.trim() : '?';
}


function getSubgroupLabel(eventSubgroupId) {
    if (!eventSubgroupId) {
        return null;
    }
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


function normalizeCategory(value) {
    if (!value) {
        return null;
    }
    const raw = String(value).trim().toUpperCase();
    if (CATEGORIES.includes(raw)) {
        return raw;
    }
    const m = raw.match(/\b([A-E])\b/);
    return m ? m[1] : null;
}


function resolveCategory(entry) {
    const subgroupLabel = getSubgroupLabel(entry.state?.eventSubgroupId);
    const category =
        normalizeCategory(subgroupLabel) ||
        normalizeCategory(entry?.eventSubgroup?.subgroupLabel) ||
        normalizeCategory(entry?.eventSubgroupLabel) ||
        normalizeCategory(entry?.category) ||
        normalizeCategory(entry?.state?.category);
    return category || null;
}


function rankCompare(a, b) {
    if (a.eventPosition !== b.eventPosition) {
        return a.eventPosition - b.eventPosition;
    }
    return b.distance - a.distance;
}


function computeGap(leader, rival) {
    if (leader.gap != null && rival.gap != null) {
        const delta = Math.abs(rival.gap - leader.gap);
        if (delta < 60) {
            return `${delta.toFixed(1)}s`;
        }
        const mins = Math.floor(delta / 60);
        const secs = Math.round(delta % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    if (leader.distance && rival.distance) {
        const delta = Math.abs(leader.distance - rival.distance);
        if (delta < 1000) {
            return `${Math.round(delta)}m`;
        }
        return `${(delta / 1000).toFixed(1)}km`;
    }
    return null;
}


function toAgeYears(value) {
    if (!value) {
        return null;
    }
    if (typeof value === 'number') {
        if (value > 0 && value < 120) {
            return value;
        }
        if (value > 1000000000) {
            const ms = value > 1000000000000 ? value : value * 1000;
            const birthDate = new Date(ms);
            if (!Number.isNaN(birthDate.getTime())) {
                return calcAgeFromDate(birthDate);
            }
        }
        return null;
    }
    if (typeof value === 'string') {
        const parsed = new Date(value);
        if (!Number.isNaN(parsed.getTime())) {
            return calcAgeFromDate(parsed);
        }
    }
    return null;
}


function calcAgeFromDate(birthDate) {
    const now = new Date();
    let age = now.getFullYear() - birthDate.getFullYear();
    const hasBirthdayPassed =
        now.getMonth() > birthDate.getMonth() ||
        (now.getMonth() === birthDate.getMonth() && now.getDate() >= birthDate.getDate());
    if (!hasBirthdayPassed) {
        age -= 1;
    }
    return age >= 0 ? age : null;
}


function getGender(athlete) {
    if (!athlete) {
        return null;
    }
    const raw = athlete.gender || athlete.sex || athlete.sexType;
    if (typeof raw === 'string') {
        const lower = raw.toLowerCase();
        if (lower.startsWith('f')) {
            return 'female';
        }
        if (lower.startsWith('m')) {
            return 'male';
        }
    }
    if (typeof athlete.isMale === 'boolean') {
        return athlete.isMale ? 'male' : 'female';
    }
    return null;
}


function getAge(athlete) {
    if (!athlete) {
        return null;
    }
    const direct = toAgeYears(athlete.age);
    if (direct != null) {
        return direct;
    }
    const dob =
        athlete.dob ||
        athlete.birthDate ||
        athlete.dateOfBirth ||
        athlete.birthday;
    return toAgeYears(dob);
}


function onNearby(data) {
    const now = Date.now();
    if (now - lastRefresh < REFRESH_INTERVAL) {
        return;
    }
    lastRefresh = now;

    if (!data || !data.length) {
        if (enabledCategories.size === 0) {
            noDataEl.classList.remove('hidden');
            scoreboardEl.classList.add('hidden');
            noDataEl.textContent = 'Enable at least one category in settings';
            return;
        }
        if (riderCache.size > 0) {
            render(calculateScores([...riderCache.values()]), riderCache.size);
            return;
        }
        noDataEl.classList.remove('hidden');
        scoreboardEl.classList.add('hidden');
        noDataEl.textContent = 'Waiting for race data...';
        return;
    }

    if (enabledCategories.size === 0) {
        noDataEl.classList.remove('hidden');
        scoreboardEl.classList.add('hidden');
        noDataEl.textContent = 'Enable at least one category in settings';
        return;
    }

    for (const entry of data) {
        const name = getRiderName(entry.athlete);

        const community = detectCommunity(entry.athlete);
        if (!community) {
            continue;
        }

        const speed = entry.state?.speed;
        if (!speed || speed <= 0) {
            continue;
        }

        const category = resolveCategory(entry);
        if (!category) {
            continue;
        }

        const weight = entry.athlete?.weight || 0;
        const peak30 = entry.stats?.power?.peaks?.[30]?.avg || 0;
        const peak300 = entry.stats?.power?.peaks?.[300]?.avg || 0;
        const peak1200 = entry.stats?.power?.peaks?.[1200]?.avg || 0;
        const powerAvg = entry.stats?.power?.avg || 0;

        riderCache.set(entry.athleteId, {
            athleteId: entry.athleteId,
            name: cleanName(name),
            community,
            category,
            eventPosition: entry.eventPosition || Infinity,
            gap: entry.gap,
            distance: entry.state?.distance || 0,
            peak30Wkg: weight > 0 ? peak30 / weight : 0,
            peak300Wkg: weight > 0 ? peak300 / weight : 0,
            peak1200Wkg: weight > 0 ? peak1200 / weight : 0,
            powerAvgWkg: weight > 0 ? powerAvg / weight : 0,
            gender: getGender(entry.athlete),
            age: getAge(entry.athlete),
            lastSeen: now,
        });
    }

    if (riderCache.size === 0) {
        noDataEl.classList.remove('hidden');
        scoreboardEl.classList.add('hidden');
        noDataEl.textContent = 'No tagged community riders found nearby';
        return;
    }

    render(calculateScores([...riderCache.values()]), riderCache.size);
}


function calculateScores(riders) {
    const scores = Object.fromEntries(COMMUNITY_KEYS.map(x => [x, 0]));
    const catResults = {};
    const bonuses = [];
    const activeCategories = CATEGORIES.filter(cat => enabledCategories.has(cat));
    const activeRiders = riders.filter(r => activeCategories.includes(r.category));

    for (const cat of activeCategories) {
        const catRiders = riders
            .filter(r => r.category === cat)
            .sort(rankCompare);

        const top3 = catRiders.slice(0, 3).map((r, i) => {
            const points = CAT_POINTS[i] || 0;
            scores[r.community] += points;
            const rival = catRiders.find(x => x.community !== r.community && rankCompare(x, r) > 0);
            return {
                ...r,
                points,
                position: i + 1,
                rivalGap: rival ? computeGap(r, rival) : null,
            };
        });

        catResults[cat] = {riders: top3, total: catRiders.length};

        awardCategoryBonus(cat, catRiders, {
            scores,
            bonuses,
            label: 'Best 5m W/kg',
            icon: 'fitness_center',
            metric: r => r.peak300Wkg || r.powerAvgWkg,
            points: 1,
            higherWins: true,
            note: '* incomplete data: using current avg W/kg where 5m peak is unavailable',
        });

        awardCategoryBonus(cat, catRiders, {
            scores,
            bonuses,
            label: 'Best 30s W/kg',
            icon: 'offline_bolt',
            metric: r => r.peak30Wkg,
            points: 1,
            higherWins: true,
        });

        const top10 = catRiders.slice(0, 10);
        awardCategoryBonus(cat, top10, {
            scores,
            bonuses,
            label: 'Lowest 20m W/kg (Top 10)',
            icon: 'south',
            metric: r => r.peak1200Wkg || r.powerAvgWkg,
            points: 1,
            higherWins: false,
            note: '* incomplete data: using current avg W/kg where 20m peak is unavailable',
        });
    }

    awardGlobalFastestBonus(activeRiders, {
        scores,
        bonuses,
        label: 'Fastest Woman',
        icon: 'female',
        points: 2,
        predicate: r => r.gender === 'female',
    });

    awardGlobalFastestBonus(activeRiders, {
        scores,
        bonuses,
        label: 'Fastest 60+',
        icon: 'elderly',
        points: 2,
        predicate: r => r.age != null && r.age >= 60,
    });

    return {
        scores,
        catResults,
        bonuses,
        activeCategories,
        totalPoints: activeCategories.length * 9 + 4,
    };
}


function awardCategoryBonus(category, riders, cfg) {
    const candidates = riders.filter(r => cfg.metric(r) > 0);
    if (!candidates.length) {
        cfg.bonuses.push({
            label: `${category} ${cfg.label}`,
            icon: cfg.icon,
            rider: null,
            value: 'pending',
            points: cfg.points,
            note: cfg.note || '',
        });
        return;
    }

    const winner = candidates.reduce((best, cur) => {
        const bestValue = cfg.metric(best);
        const curValue = cfg.metric(cur);
        if (cfg.higherWins) {
            return curValue > bestValue ? cur : best;
        }
        return curValue < bestValue ? cur : best;
    });

    cfg.scores[winner.community] += cfg.points;
    cfg.bonuses.push({
        label: `${category} ${cfg.label}`,
        icon: cfg.icon,
        rider: winner,
        value: `${cfg.metric(winner).toFixed(2)} W/kg`,
        points: cfg.points,
        note: cfg.note || '',
    });
}


function awardGlobalFastestBonus(riders, cfg) {
    const candidates = riders
        .filter(cfg.predicate)
        .sort(rankCompare);

    if (!candidates.length || candidates[0].eventPosition === Infinity) {
        cfg.bonuses.push({
            label: cfg.label,
            icon: cfg.icon,
            rider: null,
            value: 'pending',
            points: cfg.points,
        });
        return;
    }

    const winner = candidates[0];
    cfg.scores[winner.community] += cfg.points;
    cfg.bonuses.push({
        label: cfg.label,
        icon: cfg.icon,
        rider: winner,
        value: `Pos ${winner.eventPosition}`,
        points: cfg.points,
    });
}


const CAT_HUES = {A: 0, B: 90, C: 180, D: 60, E: 260};

function render(result, totalRiders) {
    noDataEl.classList.add('hidden');
    scoreboardEl.classList.remove('hidden');

    const leftTeam = COMMUNITY_DEFS[0].key;
    const rightTeam = COMMUNITY_DEFS[1].key;

    scoreLeftEl.textContent = result.scores[leftTeam];
    scoreRightEl.textContent = result.scores[rightTeam];
    scoreTotalEl.textContent = `/\u2009${result.totalPoints}`;

    const maxPoints = result.totalPoints || TOTAL_POINTS;
    barLeftEl.style.width = `${(result.scores[leftTeam] / maxPoints) * 100}%`;
    barRightEl.style.width = `${(result.scores[rightTeam] / maxPoints) * 100}%`;

    renderCategories(result.catResults, result.activeCategories);
    renderBonuses(result.bonuses);
    riderCountEl.textContent = totalRiders;
}


function renderCategories(catResults, activeCategories) {
    let html = '';
    for (const cat of activeCategories) {
        const {riders, total} = catResults[cat];
        const hue = CAT_HUES[cat];

        html += '<div class="category">';
        html += '<div class="cat-header">';
        html += `<span class="cat-badge" style="--hue: ${hue}deg;">${cat}</span>`;
        html += `<span class="cat-count">${total} riders</span>`;
        html += '</div>';

        if (!riders.length) {
            html += '<div class="cat-empty">No riders yet</div>';
        } else {
            for (const r of riders) {
                const teamClass = getCommunityClass(r.community);
                html += '<div class="cat-rider">';
                html += `<span class="rider-pos">${r.position}.</span>`;
                html += `<span class="rider-name ${teamClass}">${esc(r.name)}</span>`;
                if (r.rivalGap) {
                    html += `<span class="rider-gap">+${esc(r.rivalGap)}</span>`;
                }
                html += `<span class="rider-team-tag ${teamClass}">${esc(getCommunityShort(r.community))}</span>`;
                html += `<span class="rider-points">+${r.points}</span>`;
                html += '</div>';
            }
        }

        html += '</div>';
    }
    categoriesEl.innerHTML = html;
}


export function settingsMain() {
    Common.enableSentry();
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({enabledCategories: [...CATEGORIES]});

    const form = document.querySelector('form#options');
    const noteEl = document.getElementById('category-note');
    if (!form) {
        return;
    }

    const checkboxes = [...form.querySelectorAll('input[name="category"]')];
    const selected = getEnabledCategories();
    for (const cb of checkboxes) {
        cb.checked = selected.has(cb.value);
    }

    form.addEventListener('change', () => {
        const values = checkboxes.filter(x => x.checked).map(x => x.value);
        Common.settingsStore.set('enabledCategories', values);
        if (noteEl) {
            noteEl.textContent = values.length ?
                'Selected categories are used for leaderboard and category bonuses.' :
                'No categories selected. Enable at least one category to show results.';
        }
    });

    if (noteEl && !checkboxes.some(x => x.checked)) {
        noteEl.textContent = 'No categories selected. Enable at least one category to show results.';
    }
}


function renderBonuses(bonuses) {
    let html = '';
    for (const b of bonuses) {
        html += '<div class="bonus-row">';
        html += `<span class="bonus-icon"><ms>${b.icon}</ms></span>`;
        html += `<span class="bonus-label">${esc(b.label)}</span>`;
        if (b.rider) {
            const teamClass = getCommunityClass(b.rider.community);
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
        html += '</div>';
    }
    bonusListEl.innerHTML = html;
}


function esc(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
