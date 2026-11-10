'use strict';

let routeCatalog = [];
let workoutCatalog = [];
let selectedWorkout = null;
let customWorkout = false;
let bikeSelector = null;
let progressChart = null;
let workoutRequest = 0;
let estimateRequest = 0;
let estimating = false;
const editedProfileFields = new Set();
const element = id => document.getElementById(id);
const SETTINGS_KEY = 'workoutRouteSettings';
let savedSettings = {};
let restoringSettings = true;
let routesLoaded = false;
let workoutsLoaded = false;
let bikesLoaded = false;
const editedBikeFields = new Set();

function storageError(error) {
    console.warn('Could not access workout route settings:', error);
    element('storageStatus').textContent = 'Browser settings could not be loaded or saved. You can still use the page, but your choices may not survive a reload.';
}

function loadSettings() {
    try {
        const value = localStorage.getItem(SETTINGS_KEY);
        if (value === null) return {};
        const settings = JSON.parse(value);
        if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
            throw new Error('Invalid saved workout route settings.');
        }
        return settings;
    } catch (error) {
        storageError(error);
        return {};
    }
}

function saveSettings() {
    if (restoringSettings) return;
    const settings = {
        ...savedSettings,
        riderWeight: element('riderWeight').value,
        riderHeight: element('riderHeight').value,
        ftp: element('ftp').value,
        includeLeadin: element('includeLeadin').checked,
        untargetedPercent: element('untargetedPercent').value,
        workoutSearch: element('workoutSearch').value,
    };
    if (routesLoaded) {
        settings.world = element('worldFilter').value;
        settings.routeId = element('routeSelect').value;
    }
    if (bikesLoaded) Object.assign(settings, bikeSelector.getConfig());
    if (workoutsLoaded || customWorkout) {
        settings.workoutId = customWorkout ? '' : element('workoutSelect').value;
    }
    savedSettings = settings;
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (error) {
        storageError(error);
    }
}

function restoreInputs(settings) {
    for (const id of ['riderWeight', 'riderHeight', 'ftp', 'untargetedPercent', 'workoutSearch']) {
        if (typeof settings[id] === 'string') {
            element(id).value = settings[id];
            if (['riderWeight', 'riderHeight', 'ftp'].includes(id) && settings[id] !== '') {
                editedProfileFields.add(id);
            }
        }
    }
    if (typeof settings.includeLeadin === 'boolean') {
        element('includeLeadin').checked = settings.includeLeadin;
    }
}

function restoreChoice(id, value) {
    if (typeof value !== 'string' || ![...element(id).options].some(option => option.value === value)) return false;
    element(id).value = value;
    return true;
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    if (response.redirected) throw new Error('Your session expired. Reload the page to sign in.');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
}

function formatDuration(seconds) {
    const rounded = Math.round(seconds);
    return `${Math.floor(rounded / 3600)}:${String(Math.floor(rounded / 60) % 60).padStart(2, '0')}:${String(rounded % 60).padStart(2, '0')}`;
}

function invalidateEstimate() {
    estimateRequest++;
    element('results').hidden = true;
    updateEstimateButton();
}

function updateEstimateButton() {
    element('estimateBtn').disabled = estimating || !selectedWorkout
        || !element('routeSelect').value || !bikeSelector?.isReady();
}

function renderWarnings(id, warnings) {
    element(id).replaceChildren();
    for (const warning of warnings) {
        const item = document.createElement('li');
        item.textContent = warning;
        element(id).appendChild(item);
    }
}

function showWorkout(workout, uploaded = false) {
    selectedWorkout = workout;
    customWorkout = uploaded;
    element('workoutInfo').textContent = workout
        ? `${uploaded ? 'Custom: ' : ''}${workout.name} - ${formatDuration(workout.duration_sec)} - ${workout.source}`
        : 'Choose a built-in workout or upload a custom file.';
    const warnings = workout ? [...workout.warnings] : [];
    if (workout?.ftp_override != null) {
        warnings.push(`Prescribed targets use a fixed ${workout.ftp_override} W reference. Editing rider FTP only affects blocks without targets.`);
    }
    renderWarnings('workoutWarnings', warnings);
    const untargeted = !!workout?.segments.some(segment => segment.power_low === null);
    element('untargetedGroup').hidden = !untargeted;
    element('untargetedPercent').disabled = !untargeted;
    invalidateEstimate();
    saveSettings();
}

function filterWorkouts() {
    const search = element('workoutSearch').value.toLowerCase();
    const previous = element('workoutSelect').value;
    const matches = workoutCatalog.filter(workout => `${workout.name} ${workout.source}`.toLowerCase().includes(search));
    element('workoutSelect').replaceChildren(new Option('Choose a built-in workout', ''));
    for (const workout of matches) {
        element('workoutSelect').appendChild(new Option(
            `${workout.name} (${formatDuration(workout.duration_sec)}) - ${workout.source}`, workout.id));
    }
    if (matches.some(workout => workout.id === previous)) element('workoutSelect').value = previous;
    else if (!customWorkout) {
        workoutRequest++;
        showWorkout(null);
    }
}

function filterRoutes() {
    const world = element('worldFilter').value;
    const previous = element('routeSelect').value;
    const routes = routeCatalog.filter(route => !world || route.world === world);
    element('routeSelect').replaceChildren(new Option('Choose a route', ''));
    for (const route of routes) {
        element('routeSelect').appendChild(new Option(
            `${route.name} - ${route.distance_km} km (${route.world_name})`, route.id));
    }
    if (routes.some(route => String(route.id) === previous)) element('routeSelect').value = previous;
    showRouteInfo();
}

function showRouteInfo() {
    const route = routeCatalog.find(item => String(item.id) === element('routeSelect').value);
    element('routeInfo').textContent = route
        ? `${route.distance_km} km, ${route.ascent_m} m ascent. ` +
          (element('includeLeadin').checked ? `Plus lead-in: ${route.leadin_distance_km} km, ${route.leadin_ascent_m} m ascent.` : 'Lead-in excluded.')
        : '';
    invalidateEstimate();
}

async function selectBuiltIn() {
    const request = ++workoutRequest;
    const id = element('workoutSelect').value;
    showWorkout(null);
    element('error').textContent = '';
    if (!id) return;
    element('workoutInfo').textContent = 'Loading workout...';
    try {
        const workout = await fetchJSON(`/api/workouts/${encodeURIComponent(id)}`);
        if (request === workoutRequest) showWorkout(workout);
    } catch (error) {
        if (request === workoutRequest) {
            showWorkout(null);
            element('error').textContent = error.message;
        }
    }
}

async function uploadWorkout() {
    const file = element('workoutFile').files[0];
    if (!file) {
        element('error').textContent = 'Choose a custom .zwo file first.';
        return;
    }
    const request = ++workoutRequest;
    element('workoutSelect').value = '';
    showWorkout(null);
    element('error').textContent = '';
    element('uploadBtn').disabled = true;
    element('workoutInfo').textContent = 'Loading custom workout...';
    try {
        const body = new FormData();
        body.append('file', file);
        const workout = await fetchJSON('/api/workouts/upload', { method: 'POST', body });
        if (request === workoutRequest) showWorkout(workout, true);
    } catch (error) {
        if (request === workoutRequest) {
            showWorkout(null);
            element('error').textContent = error.message;
        }
    } finally {
        element('uploadBtn').disabled = false;
    }
}

function buildRequest() {
    const config = bikeSelector.getConfig();
    const body = {
        route_id: element('routeSelect').value,
        include_leadin: element('includeLeadin').checked,
        rider_weight_kg: Number(element('riderWeight').value),
        rider_height_cm: Number(element('riderHeight').value),
        ftp_w: Number(element('ftp').value),
        frame_id: config.frameId, wheel_id: config.wheelId, upgrade_level: config.level,
    };
    if (!element('untargetedPercent').disabled) {
        body.untargeted_percent = Number(element('untargetedPercent').value);
    }
    if (customWorkout) body.custom_workout = selectedWorkout;
    else body.workout_id = selectedWorkout.id;
    return body;
}

function displayResult(result) {
    element('resultTitle').textContent = result.completed
        ? 'Yes - estimated route completion during the workout'
        : 'Not quite - workout ends before the route finish';
    element('resultSummary').textContent = result.completed
        ? `Route finish at ${formatDuration(result.finish_time_sec)}, with ${formatDuration(result.workout_time_remaining_sec)} of workout remaining.`
        : `Predicted progress: ${result.distance_covered_km.toFixed(2)} km of ${result.route_distance_km.toFixed(2)} km. ${result.distance_remaining_km.toFixed(2)} km still to go.`;
    element('resultDetails').textContent =
        `${result.route_name} ${result.include_leadin ? '(including lead-in)' : '(excluding lead-in)'} | ${result.workout_name} | ` +
        `Workout: ${formatDuration(result.workout_duration_sec)} | Rider FTP: ${result.rider_ftp_w} W | Target reference: ${result.reference_ftp_w} W | ` +
        `Modeled route distance: ${result.route_distance_km.toFixed(2)} km.`;
    renderWarnings('resultWarnings', result.warnings);
    element('results').hidden = false;
    if (progressChart) progressChart.destroy();
    const profile = result.profile;
    const points = key => profile.time_sec.map((time, index) => ({ x: time / 60, y: profile[key][index] }));
    progressChart = new Chart(element('progressChart'), {
        type: 'line',
        data: { datasets: [
            { label: 'Target power (W)', data: points('power_w'), yAxisID: 'power', borderColor: '#f7931e', pointRadius: 0, stepped: 'after' },
            { label: 'Distance (km)', data: points('distance_km'), yAxisID: 'distance', borderColor: '#78b4ff', pointRadius: 0 },
            { label: 'Route finish (km)', data: [
                { x: 0, y: result.route_distance_km },
                { x: profile.time_sec.at(-1) / 60, y: result.route_distance_km },
            ], yAxisID: 'distance', borderColor: '#aaa', borderDash: [6, 6], pointRadius: 0 },
        ] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Workout time (minutes)' } },
                power: { position: 'left', title: { display: true, text: 'Power (W)' } },
                distance: { position: 'right', title: { display: true, text: 'Distance (km)' }, grid: { drawOnChartArea: false } },
            },
        },
    });
}

async function runEstimate(event) {
    event.preventDefault();
    if (!selectedWorkout || !element('estimateForm').reportValidity()) return;
    const request = ++estimateRequest;
    estimating = true;
    updateEstimateButton();
    element('estimateBtn').textContent = 'Calculating...';
    element('error').textContent = '';
    element('results').hidden = true;
    try {
        const result = await fetchJSON('/api/workout_route_plan', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildRequest()),
        });
        if (request === estimateRequest) displayResult(result);
    } catch (error) {
        if (request === estimateRequest) element('error').textContent = error.message;
    } finally {
        estimating = false;
        element('estimateBtn').textContent = 'Check route completion';
        updateEstimateButton();
    }
}

async function loadProfile() {
    try {
        const profile = await fetchJSON('/api/my_profile');
        const fields = { riderWeight: profile.weight_kg, riderHeight: profile.height_cm, ftp: profile.ftp_w };
        for (const [id, value] of Object.entries(fields)) {
            if (!editedProfileFields.has(id) && value != null) element(id).value = value;
        }
        element('profileStatus').textContent = 'Profile loaded. You can edit all values; fill in any missing measurements or FTP.';
    } catch (error) {
        element('profileStatus').textContent = `Profile could not be loaded: ${error.message} Enter rider data manually.`;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    savedSettings = loadSettings();
    restoreInputs(savedSettings);
    element('estimateForm').addEventListener('submit', runEstimate);
    element('estimateForm').addEventListener('input', invalidateEstimate);
    element('estimateForm').addEventListener('input', saveSettings);
    element('estimateForm').addEventListener('change', saveSettings);
    for (const id of ['riderWeight', 'riderHeight', 'ftp']) {
        element(id).addEventListener('input', () => editedProfileFields.add(id));
    }
    for (const id of ['frameSelect', 'wheelSelect', 'upgradeLevel']) {
        element(id).addEventListener('change', () => editedBikeFields.add(id));
    }
    element('worldFilter').addEventListener('change', filterRoutes);
    element('routeSelect').addEventListener('change', showRouteInfo);
    element('includeLeadin').addEventListener('change', showRouteInfo);
    element('workoutSearch').addEventListener('input', filterWorkouts);
    element('workoutSelect').addEventListener('change', selectBuiltIn);
    element('uploadBtn').addEventListener('click', uploadWorkout);
    bikeSelector = new BikeSelector({
        frameSelect: 'frameSelect', wheelSelect: 'wheelSelect', levelSelect: 'upgradeLevel',
        weightOut: 'bikeWeight', cdaOut: 'bikeCda', framePlaceholder: 'Choose a frame',
        onChange: () => { invalidateEstimate(); saveSettings(); },
    });
    const outcomes = await Promise.allSettled([
        loadProfile(),
        bikeSelector.load().then(async () => {
            if (!editedBikeFields.size) {
                await bikeSelector.setConfig({
                    frameId: savedSettings.frameId,
                    wheelId: savedSettings.wheelId,
                    level: savedSettings.level,
                });
            }
            bikesLoaded = true;
        }),
        fetchJSON('/api/tt_pacing/routes').then(data => {
            routeCatalog = data.routes;
            const worlds = new Map(routeCatalog.map(route => [route.world, route.world_name]));
            for (const [world, name] of worlds) element('worldFilter').appendChild(new Option(name, world));
            restoreChoice('worldFilter', savedSettings.world);
            filterRoutes();
            restoreChoice('routeSelect', savedSettings.routeId);
            showRouteInfo();
            routesLoaded = true;
        }),
        fetchJSON('/api/workouts').then(async data => {
            workoutCatalog = data.workouts;
            const restoreWorkout = workoutRequest === 0;
            filterWorkouts();
            workoutsLoaded = true;
            if (restoreWorkout && restoreChoice('workoutSelect', savedSettings.workoutId)) {
                await selectBuiltIn();
            }
        }),
    ]);
    const failures = outcomes.filter(outcome => outcome.status === 'rejected')
        .map(outcome => outcome.reason.message);
    if (failures.length) {
        element('error').textContent = [element('error').textContent, ...failures].filter(Boolean).join(' ');
    }
    restoringSettings = false;
    saveSettings();
    updateEstimateButton();
});
