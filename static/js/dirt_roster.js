'use strict';

const dirtFields = ['time_zone', 'league', 'category', 'team'];
const dirtLabels = {time_zone: 'time zones', league: 'leagues', category: 'categories', team: 'teams'};
const dirtEsc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const dirtVelo = [['endurance', 'Endurance'], ['pursuit', 'Pursuit'], ['sprint', 'Sprint'], ['punch', 'Punch'], ['climb', 'Climb'], ['timeTrial', 'Time Trial']];
const dirtIntervals = [[5, '5 seconds'], [15, '15 seconds'], [30, '30 seconds'], [60, '1 minute'], [120, '2 minutes'], [300, '5 minutes'], [1200, '20 minutes']];
const dirtRosterColumns = [['name', 'Rider'], ['zwift_id', 'Zwift ID'], ['team', 'Team'], ['time_zone', 'Time zone'], ['league', 'League'], ['category', 'Category']];
const dirtTeamColumns = [['name', 'Team'], ['time_zone', 'Time zone'], ['league', 'League'], ['category', 'Category'], ['rider_count', 'Riders']];

function dirtColumns(tab, unit = 'w', source = 'live') {
    if (tab === 'roster') return dirtRosterColumns;
    const identity = [['name', 'Rider'], ['team', 'Team']];
    if (tab === 'velo') return [...identity, ...dirtVelo.map(([key, title]) => [`velo_${key}`, title])];
    const suffix = unit === 'wkg' ? 'W/kg' : 'W';
    return [...identity, ...dirtIntervals.map(([duration, title]) => [
        `power_${duration}_${source}`, `${title} ${source === 'live' ? 'Live estimate' : '90-day best'} (${suffix})`,
    ])];
}

function dirtStatsRows(rows, stats, unit = 'w') {
    return rows.map(row => {
        const record = stats?.riders?.[String(row.zwift_id)];
        const values = {...row, stats_status: record?.status || 'pending', stats_reason: record?.reason || ''};
        for (const [key] of dirtVelo) values[`velo_${key}`] = record?.velo?.[key] ?? null;
        for (const [duration] of dirtIntervals) {
            for (const source of ['live', 'best_90d']) values[`power_${duration}_${source}`] = record?.power?.[duration]?.[source]?.[unit] ?? null;
        }
        return values;
    });
}

function dirtStatsCoverage(stats) {
    const records = Object.values(stats?.riders || {}).filter(record => record.status === 'ok');
    return {
        velo: records.filter(record => dirtVelo.some(([key]) => record.velo?.[key] == null)).length,
        power: records.filter(record => dirtIntervals.some(([duration]) => ['live', 'best_90d'].some(source => ['w', 'wkg'].some(unit => record.power?.[duration]?.[source]?.[unit] == null)))).length,
    };
}

function dirtMatches(team, filters, except = '') {
    return dirtFields.every(key => key === except || !filters[key] || String(key === 'team' ? team.team_id : team[key]) === filters[key]);
}

function dirtFilter(teams, filters) {
    const query = (filters.q || '').trim().toLocaleLowerCase();
    return teams.filter(team => dirtMatches(team, filters)).flatMap(team => {
        if (!query || team.name.toLocaleLowerCase().includes(query)) return [team];
        const riders = team.riders.filter(rider => `${rider.name} ${rider.zwift_id ?? ''}`.toLocaleLowerCase().includes(query));
        return riders.length ? [{...team, riders}] : [];
    });
}

function dirtRows(teams) {
    return teams.flatMap(team => team.riders.map(rider => ({...rider, team: team.name, team_id: team.team_id, time_zone: team.time_zone, league: team.league, category: team.category})));
}

function dirtSorted(rows, key, direction) {
    return [...rows].sort((first, second) => {
        const left = first[key], right = second[key];
        if (left == null || right == null) return left == null ? (right == null ? 0 : 1) : -1;
        return direction * (typeof left === 'number' && typeof right === 'number'
            ? left - right : String(left).localeCompare(String(right), undefined, {numeric: true, sensitivity: 'base'}));
    });
}

function dirtCSV(rows, columns = dirtRosterColumns) {
    const quote = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
    return [columns.map(([, title]) => title), ...rows.map(row => columns.map(([key]) => row[key]))]
        .map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
}

async function initDirtRoster() {
    const element = id => document.getElementById(id);
    const status = element('status');
    let data;
    try {
        const response = await fetch('/api/dirt/roster');
        data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Roster unavailable.');
    } catch (error) {
        status.textContent = error.message;
        element('snapshot-state').textContent = 'Roster unavailable';
        element('snapshot-state').className = 'pending';
        return;
    }
    const parameters = new URLSearchParams(location.search);
    let view = parameters.get('view') === 'teams' ? 'teams' : 'riders';
    let tab = ['velo', 'power'].includes(parameters.get('tab')) ? parameters.get('tab') : 'roster';
    let unit = parameters.get('unit') === 'wkg' ? 'wkg' : 'w';
    let powerSource = parameters.get('source') === 'best_90d' ? 'best_90d' : 'live';
    const sorts = {roster: {key: 'name', direction: 1}, velo: {key: 'name', direction: 1}, power: {key: 'name', direction: 1}};
    let stats = null, statsError = '', statsLoading = true;
    let statsCoverage = {velo: 0, power: 0};
    let downloadURL;
    const filters = Object.fromEntries([...dirtFields, 'q'].map(key => [key, parameters.get(key) || '']));
    element('search').value = filters.q;
    element('total-riders').textContent = data.rider_count.toLocaleString();
    element('total-teams').textContent = data.team_count.toLocaleString();
    element('total-leagues').textContent = new Set(data.teams.map(team => team.league)).size;

    function render() {
        const sort = sorts[tab];
        const rosterView = tab === 'roster';
        const teamView = rosterView && view === 'teams';
        for (const mode of ['roster', 'velo', 'power']) {
            element(`${mode}-tab`).setAttribute('aria-selected', String(tab === mode));
            element(`${mode}-tab`).tabIndex = tab === mode ? 0 : -1;
        }
        element('roster-panel').setAttribute('aria-labelledby', `${tab}-tab`);
        element('roster-views').hidden = !rosterView;
        element('power-controls').hidden = tab !== 'power';
        document.querySelectorAll('input[name="power-unit"]').forEach(input => { input.checked = input.value === unit; });
        document.querySelectorAll('input[name="power-source"]').forEach(input => { input.checked = input.value === powerSource; });
        const snapshot = rosterView ? data : stats;
        element('snapshot-date').textContent = snapshot ? new Date(snapshot.fetched_at).toLocaleString() : '';
        element('snapshot-date').dateTime = snapshot?.fetched_at || '';
        element('snapshot-state').textContent = rosterView
            ? (data.complete ? 'Roster collected' : `${data.rosters_fetched} / ${data.team_count} rosters collected`)
            : (statsLoading ? 'Loading ZwiftRacing stats' : stats ? (stats.complete ? 'Stats collected' : `${stats.fetched_count} / ${stats.rider_count} riders collected`) : 'Stats unavailable');
        element('snapshot-state').className = snapshot?.complete ? '' : 'pending';
        const warnings = [];
        if (rosterView) {
            if (!data.complete) warnings.push(`Collection in progress: ${data.profiles_fetched} rider profiles fetched.`);
            const missingIDs = new Set(data.teams.flatMap(team => team.riders.filter(rider => rider.profile_fetched && rider.zwift_id == null).map(rider => rider.rider_id))).size;
            if (missingIDs) warnings.push(`${missingIDs} riders without a Zwift ID.`);
        } else {
            if (statsError) warnings.push(statsError);
            else if (statsLoading) warnings.push('Loading ZwiftRacing stats...');
            else if (!stats?.complete) warnings.push(`Stats collection in progress: ${stats?.fetched_count || 0} / ${stats?.rider_count || data.rider_count} riders.`);
            if (stats?.unavailable_count) warnings.push(`${stats.unavailable_count} ZwiftRacing profiles unavailable.`);
            if (statsCoverage[tab]) warnings.push(`Series data: ${statsCoverage[tab]} ${statsCoverage[tab] === 1 ? 'rider has' : 'riders have'} ${tab === 'velo' ? 'missing category scores' : 'incomplete power data'} on ZwiftRacing.`);
        }
        status.textContent = warnings.join(' ');
        status.hidden = !warnings.length;
        for (const key of dirtFields) {
            const select = element(key);
            const available = data.teams.filter(team => dirtMatches(team, filters, key));
            const choices = new Map(available.map(team => key === 'team' ? [String(team.team_id), team.name] : [team[key], team[key]]));
            if (filters[key] && !choices.has(filters[key])) filters[key] = '';
            select.replaceChildren(new Option(`All ${dirtLabels[key]}`, ''), ...[...choices].sort((first, second) => first[1].localeCompare(second[1], undefined, {numeric: true})).map(([value, name]) => new Option(name, value)));
            select.value = filters[key];
        }
        const teams = dirtFilter(data.teams, filters);
        const assignments = dirtRows(teams);
        const uniqueRiders = new Set(assignments.map(rider => rider.rider_id)).size;
        element('result-count').textContent = `${uniqueRiders.toLocaleString()} ${uniqueRiders === 1 ? 'rider' : 'riders'} / ${assignments.length.toLocaleString()} ${assignments.length === 1 ? 'assignment' : 'assignments'} / ${teams.length} ${teams.length === 1 ? 'team' : 'teams'}`;
        element('riders-view').setAttribute('aria-pressed', String(view === 'riders'));
        element('teams-view').setAttribute('aria-pressed', String(view === 'teams'));
        const columns = teamView ? dirtTeamColumns : dirtColumns(tab, unit, powerSource);
        const table = element('roster');
        table.className = rosterView ? '' : 'stats-table';
        table.querySelector('caption').textContent = rosterView ? 'DIRT Racing Series season 14 team assignments' : tab === 'velo' ? 'Rider vELO category scores' : `Rider ${powerSource === 'live' ? 'live estimate' : '90-day best'} power in ${unit === 'wkg' ? 'watts per kilogram' : 'watts'}`;
        const header = element('roster').querySelector('thead');
        const columnLabel = (title, index) => tab === 'power' && index >= 2 ? dirtIntervals[index - 2][1].replace(' seconds', 's').replace(' minutes', 'min').replace(' minute', 'min') : title;
        const heading = ([key, title], index) => `<th scope="col"${sort.key === key ? ` aria-sort="${sort.direction === 1 ? 'ascending' : 'descending'}"` : ''}><button type="button" data-sort="${key}" aria-label="${dirtEsc(title)}" title="${dirtEsc(title)}">${columnLabel(title, index)}${sort.key === key ? (sort.direction === 1 ? ' &#8593;' : ' &#8595;') : ''}</button></th>`;
        header.innerHTML = `<tr>${columns.map(heading).join('')}</tr>`;
        const linkTeam = (id, name) => `<a href="/dirt?team=${id}" data-team="${id}">${dirtEsc(name)}</a>`;
        let rows;
        const metricRows = dirtStatsRows(assignments, stats, unit);
        if (teamView) {
            rows = dirtSorted(teams.map(team => ({...team, rider_count: team.roster_fetched ? team.riders.length : null})), sort.key, sort.direction).map(team => `<tr><td>${linkTeam(team.team_id, team.name)}</td><td>${dirtEsc(team.time_zone)}</td><td>${dirtEsc(team.league)}</td><td class="category">${dirtEsc(team.category)}</td><td>${team.roster_fetched ? team.riders.length : '<span class="muted">Pending</span>'}</td></tr>`);
        } else if (rosterView) {
            rows = dirtSorted(assignments, sort.key, sort.direction).map(rider => `<tr><td><a href="https://dirtracingseries.com/riders/profile/${rider.rider_id}/" target="_blank" rel="noopener">${dirtEsc(rider.name)}</a></td><td>${rider.zwift_id == null ? `<span class="muted">${rider.profile_fetched ? 'Unavailable' : 'Pending'}</span>` : `<a href="https://zwiftpower.com/profile.php?z=${rider.zwift_id}" target="_blank" rel="noopener">${rider.zwift_id}</a>`}</td><td>${linkTeam(rider.team_id, rider.team)}</td><td>${dirtEsc(rider.time_zone)}</td><td>${dirtEsc(rider.league)}</td><td class="category">${dirtEsc(rider.category)}</td></tr>`);
        } else {
            rows = dirtSorted(metricRows, sort.key, sort.direction).map(rider => `<tr>${columns.map(([key, title], index) => {
                if (key === 'name') return `<td><a href="https://zwiftracing.app/riders/${rider.zwift_id}/${tab === 'velo' ? 'velo' : 'power'}" target="_blank" rel="noopener">${dirtEsc(rider.name)}</a>${rider.stats_status === 'ok' ? '' : `<span class="stats-state" title="${dirtEsc(rider.stats_reason)}">${rider.stats_status === 'unavailable' ? 'Unavailable' : 'Pending'}</span>`}</td>`;
                if (key === 'team') return `<td>${linkTeam(rider.team_id, rider.team)}</td>`;
                return `<td data-metric="${key}" data-label="${dirtEsc(columnLabel(title, index))}" class="metric">${rider[key] == null ? '<span class="muted" title="No value supplied">&mdash;</span>' : dirtEsc(tab === 'velo' ? Math.round(Math.min(rider[key], 1000)) : rider[key])}</td>`;
            }).join('')}</tr>`);
        }
        element('roster').querySelector('tbody').innerHTML = rows.join('') || `<tr><td class="empty" colspan="${columns.length}">${!teamView && teams.length ? 'No riders on these team rosters.' : 'No matching results.'}</td></tr>`;
        const nextParameters = new URLSearchParams();
        for (const [key, value] of Object.entries(filters)) if (value) nextParameters.set(key, value);
        if (view !== 'riders') nextParameters.set('view', view);
        if (tab !== 'roster') nextParameters.set('tab', tab);
        if (unit !== 'w') nextParameters.set('unit', unit);
        if (powerSource !== 'live') nextParameters.set('source', powerSource);
        history.replaceState(null, '', `/dirt${nextParameters.size ? '?' + nextParameters.toString() : ''}`);
        if (downloadURL) URL.revokeObjectURL(downloadURL);
        const exportRows = rosterView ? assignments : metricRows;
        const exportColumns = rosterView ? dirtRosterColumns : [...dirtRosterColumns, ...columns.slice(2)];
        downloadURL = URL.createObjectURL(new Blob(['\ufeff', dirtCSV(dirtSorted(exportRows, sort.key, sort.direction), exportColumns)], {type: 'text/csv;charset=utf-8'}));
        element('download').href = downloadURL;
        element('download').download = `dirt-season-14-${tab}${tab === 'power' ? '-' + powerSource + '-' + unit : ''}.csv`;
        element('download').setAttribute('aria-disabled', String(!assignments.length));
    }
    element('filters').addEventListener('submit', event => event.preventDefault());
    element('search').addEventListener('input', event => { filters.q = event.target.value; render(); });
    for (const key of dirtFields) element(key).addEventListener('change', event => { filters[key] = event.target.value; render(); });
    element('clear').addEventListener('click', () => { for (const key of Object.keys(filters)) filters[key] = ''; element('search').value = ''; render(); });
    for (const mode of ['riders', 'teams']) element(`${mode}-view`).addEventListener('click', () => { view = mode; sorts.roster = {key: 'name', direction: 1}; render(); });
    const modes = ['roster', 'velo', 'power'];
    for (const mode of modes) {
        element(`${mode}-tab`).addEventListener('click', () => { tab = mode; render(); });
        element(`${mode}-tab`).addEventListener('keydown', event => {
            const index = modes.indexOf(mode);
            const next = event.key === 'ArrowRight' ? (index + 1) % modes.length : event.key === 'ArrowLeft' ? (index + modes.length - 1) % modes.length : event.key === 'Home' ? 0 : event.key === 'End' ? modes.length - 1 : null;
            if (next == null) return;
            event.preventDefault(); tab = modes[next]; render(); element(`${tab}-tab`).focus();
        });
    }
    document.querySelectorAll('input[name="power-unit"]').forEach(input => input.addEventListener('change', () => { unit = input.value; render(); }));
    document.querySelectorAll('input[name="power-source"]').forEach(input => input.addEventListener('change', () => {
        powerSource = input.value;
        sorts.power.key = sorts.power.key.replace(/_(live|best_90d)$/, `_${powerSource}`);
        render();
    }));
    element('roster').addEventListener('click', event => {
        const teamLink = event.target.closest('[data-team]');
        const sortButton = event.target.closest('[data-sort]');
        if (teamLink) {
            event.preventDefault();
            filters.team = teamLink.dataset.team;
            filters.q = ''; element('search').value = '';
            view = 'riders'; sorts[tab] = {key: 'name', direction: 1}; render();
        } else if (sortButton) {
            const sort = sorts[tab];
            sort.direction = sort.key === sortButton.dataset.sort ? -sort.direction : 1;
            sort.key = sortButton.dataset.sort; render();
        }
    });
    render();
    try {
        const response = await fetch('/api/dirt/stats');
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'ZwiftRacing stats unavailable.');
        stats = payload;
        statsCoverage = dirtStatsCoverage(stats);
    } catch (error) {
        statsError = error.message;
    }
    statsLoading = false;
    render();
}

if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', initDirtRoster);
if (typeof module !== 'undefined') module.exports = {dirtFilter, dirtRows, dirtSorted, dirtCSV, dirtStatsRows, dirtColumns, dirtStatsCoverage};