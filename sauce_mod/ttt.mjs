import * as Common from '/pages/src/common.mjs';
import {human as H} from '/shared/sauce/locale.mjs';

Common.enableSentry();

const doc = document.documentElement;
let settings;
let tbody;
let table;
let noDataEl;
let teamIds;
let lastRefresh = 0;
const refreshInterval = 1000;


export function main() {
    Common.settingsStore.setDefault({
        targetGapMeters: 2.0,
        teamAthleteIds: [],
        filterToTeam: false,
        refreshInterval: 1,
    });
    settings = Common.settingsStore.get();
    teamIds = new Set(settings.teamAthleteIds || []);

    tbody = document.querySelector('.gaps-table tbody');
    table = document.querySelector('.gaps-table');
    noDataEl = document.querySelector('.no-data');

    Common.settingsStore.addEventListener('changed', ev => {
        settings = Common.settingsStore.get();
        teamIds = new Set(settings.teamAthleteIds || []);
    });

    Common.subscribe('nearby', onNearby);
}


function onNearby(data) {
    const now = Date.now();
    if (now - lastRefresh < refreshInterval) {
        return;
    }
    lastRefresh = now;

    if (!data || !data.length) {
        table.classList.add('hidden');
        noDataEl.classList.remove('hidden');
        noDataEl.textContent = 'Waiting for nearby data...';
        return;
    }

    // Filter to team members if configured, otherwise show all nearby
    let riders;
    if (settings.filterToTeam && teamIds.size > 0) {
        riders = data.filter(x => teamIds.has(x.athleteId));
    } else {
        riders = data;
    }

    if (riders.length < 2) {
        table.classList.add('hidden');
        noDataEl.classList.remove('hidden');
        noDataEl.textContent = settings.filterToTeam ?
            'Need 2+ team members nearby (check settings)' :
            'Need 2+ riders nearby';
        return;
    }

    // Filter to riders within 100m and with valid gap data, sort by gapDistance
    riders = riders
        .filter(x => x.gapDistance != null && Math.abs(x.gapDistance) <= 100)
        .sort((a, b) => a.gapDistance - b.gapDistance);

    if (riders.length < 2) {
        table.classList.add('hidden');
        noDataEl.classList.remove('hidden');
        noDataEl.textContent = 'Need 2+ riders within 100m';
        return;
    }

    // Compute inter-rider gaps
    const rows = riders.map((rider, i) => {
        const name = rider.athlete?.fLast || rider.athlete?.sanitizedFullname || `#${rider.athleteId}`;
        const power = rider.state?.power;
        const speed = rider.state?.speed;
        const isWatching = !!rider.watching;

        let interGap = null;
        if (i > 0) {
            interGap = Math.abs(rider.gapDistance - riders[i - 1].gapDistance);
        }

        return {name, power, speed, interGap, isWatching, athleteId: rider.athleteId};
    });

    // Render
    table.classList.remove('hidden');
    noDataEl.classList.add('hidden');
    renderRows(rows);
}


function renderRows(rows) {
    // Reuse or create TR elements
    while (tbody.children.length > rows.length) {
        tbody.lastElementChild.remove();
    }
    while (tbody.children.length < rows.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="pos"></td>
            <td class="name"></td>
            <td class="gap-dist"></td>
            <td class="power"></td>
            <td class="speed"></td>`;
        tbody.appendChild(tr);
    }

    const targetGap = settings.targetGapMeters ?? 2.0;

    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const tr = tbody.children[i];
        const cells = tr.children;

        cells[0].textContent = i + 1;
        cells[1].textContent = row.name;

        if (row.interGap != null) {
            const gapM = row.interGap;
            cells[2].textContent = gapM.toFixed(1) + 'm';
            // Color code: green if close to target, yellow if drifting, red if too far
            tr.classList.remove('gap-close', 'gap-ok', 'gap-far', 'gap-danger');
            if (gapM < targetGap * 0.5) {
                tr.classList.add('gap-close');
            } else if (gapM <= targetGap * 1.5) {
                tr.classList.add('gap-ok');
            } else if (gapM <= targetGap * 3) {
                tr.classList.add('gap-far');
            } else {
                tr.classList.add('gap-danger');
            }
        } else {
            cells[2].textContent = '—';
            tr.classList.remove('gap-close', 'gap-ok', 'gap-far', 'gap-danger');
        }

        cells[3].textContent = row.power != null ? row.power + 'w' : '—';
        cells[4].textContent = row.speed != null ? row.speed.toFixed(1) : '—';

        tr.classList.toggle('is-watching', row.isWatching);
        tr.classList.toggle('is-team', teamIds.has(row.athleteId));
    }
}


export function settingsMain() {
    Common.enableSentry();
    Common.settingsStore.setDefault({
        targetGapMeters: 2.0,
        teamAthleteIds: [],
        filterToTeam: false,
    });

    const form = document.querySelector('form#settings');
    const settings = Common.settingsStore.get();

    // Populate form
    const targetGapInput = form.querySelector('[name="targetGapMeters"]');
    if (targetGapInput) {
        targetGapInput.value = settings.targetGapMeters ?? 2.0;
    }
    const filterInput = form.querySelector('[name="filterToTeam"]');
    if (filterInput) {
        filterInput.checked = !!settings.filterToTeam;
    }
    const teamInput = form.querySelector('[name="teamAthleteIds"]');
    if (teamInput) {
        teamInput.value = (settings.teamAthleteIds || []).join(', ');
    }

    form.addEventListener('input', () => {
        const targetGap = parseFloat(targetGapInput?.value);
        if (!isNaN(targetGap) && targetGap > 0) {
            Common.settingsStore.set('targetGapMeters', targetGap);
        }
    });

    form.addEventListener('change', ev => {
        const el = ev.target;
        if (el.name === 'filterToTeam') {
            Common.settingsStore.set('filterToTeam', el.checked);
        }
        if (el.name === 'teamAthleteIds') {
            const ids = el.value.split(/[,\s]+/).map(x => Number(x.trim())).filter(x => x);
            Common.settingsStore.set('teamAthleteIds', ids);
        }
    });

    // Add nearby athletes list for easy team selection
    loadNearbyForTeamPicker();
}


async function loadNearbyForTeamPicker() {
    const listEl = document.querySelector('#nearby-athletes');
    if (!listEl) {
        return;
    }

    try {
        const data = await Common.rpc.getNearbyData();
        if (!data || !data.length) {
            listEl.innerHTML = '<div class="empty">No nearby athletes (start a ride first)</div>';
            return;
        }

        const settings = Common.settingsStore.get();
        const currentTeam = new Set(settings.teamAthleteIds || []);

        listEl.innerHTML = '';
        for (const rider of data) {
            const name = rider.athlete?.sanitizedFullname || `Athlete ${rider.athleteId}`;
            const label = document.createElement('label');
            label.classList.add('athlete-pick');
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = rider.athleteId;
            cb.checked = currentTeam.has(rider.athleteId);
            cb.addEventListener('change', () => {
                const ids = Array.from(listEl.querySelectorAll('input:checked')).map(x => Number(x.value));
                Common.settingsStore.set('teamAthleteIds', ids);
                const teamInput = document.querySelector('[name="teamAthleteIds"]');
                if (teamInput) {
                    teamInput.value = ids.join(', ');
                }
            });
            label.appendChild(cb);
            label.appendChild(document.createTextNode(' ' + name));
            listEl.appendChild(label);
        }
    } catch (e) {
        listEl.innerHTML = '<div class="empty">Could not load nearby athletes</div>';
    }
}
