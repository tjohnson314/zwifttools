import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

let recording = false;
let startTime = null;
let telemetryRows = [];
let surfaceTags = [];       // [{timestamp, worldTime, lat, lng, surface, distance}]
let currentSurface = 'Tarmac';
let selfAthleteId = null;

// UI elements
let btnRecord, btnDownload, indicatorEl, statusTextEl, sampleCountEl,
    durationEl, currentSurfaceEl, tagCountEl, tagLogList;


export function main() {
    btnRecord = document.querySelector('.btn-record');
    btnDownload = document.querySelector('.btn-download');
    indicatorEl = document.querySelector('.indicator');
    statusTextEl = document.querySelector('.status-text');
    sampleCountEl = document.querySelector('.sample-count');
    durationEl = document.querySelector('.duration');
    currentSurfaceEl = document.querySelector('.current-surface');
    tagCountEl = document.querySelector('.tag-count');
    tagLogList = document.querySelector('.tag-log-list');

    btnRecord.addEventListener('click', toggleRecording);
    btnDownload.addEventListener('click', downloadCSV);

    for (const btn of document.querySelectorAll('.surface-btn')) {
        btn.addEventListener('click', () => setSurface(btn.dataset.surface, btn));
    }

    Common.subscribe('athlete/self', onSelfUpdate);
}


function setSurface(surface, btnEl) {
    if (surface === currentSurface) return;
    currentSurface = surface;
    currentSurfaceEl.textContent = surface;

    for (const b of document.querySelectorAll('.surface-btn')) {
        b.classList.toggle('active', b.dataset.surface === surface);
    }

    if (recording) {
        const tag = {
            timestamp: Date.now(),
            worldTime: lastWorldTime ?? '',
            lat: lastLat ?? '',
            lng: lastLng ?? '',
            distance: lastDistance ?? '',
            surface,
        };
        surfaceTags.push(tag);
        tagCountEl.textContent = surfaceTags.length;
        appendTagLog(tag);
    }
}


let lastWorldTime = null;
let lastLat = null;
let lastLng = null;
let lastDistance = null;


function toggleRecording() {
    if (recording) {
        stopRecording();
    } else {
        startRecording();
    }
}


function startRecording() {
    recording = true;
    startTime = Date.now();
    telemetryRows = [];
    surfaceTags = [];

    // Record initial surface tag
    surfaceTags.push({
        timestamp: Date.now(),
        worldTime: lastWorldTime ?? '',
        lat: lastLat ?? '',
        lng: lastLng ?? '',
        distance: lastDistance ?? '',
        surface: currentSurface,
    });

    btnRecord.textContent = 'Stop';
    btnRecord.classList.add('active');
    btnDownload.disabled = true;
    indicatorEl.className = 'indicator recording';
    statusTextEl.textContent = 'Recording';
    tagCountEl.textContent = surfaceTags.length;
    tagLogList.innerHTML = '';
    appendTagLog(surfaceTags[0]);
}


function stopRecording() {
    recording = false;
    btnRecord.textContent = 'Start';
    btnRecord.classList.remove('active');
    btnDownload.disabled = telemetryRows.length === 0;
    indicatorEl.className = 'indicator stopped';
    statusTextEl.textContent = `Stopped — ${telemetryRows.length} samples`;
}


function onSelfUpdate(data) {
    if (!data) return;

    selfAthleteId = data.athleteId;
    const state = data.state;
    if (!state) return;

    const latlng = state.latlng;
    lastWorldTime = state.worldTime;
    lastLat = latlng ? latlng[0] : null;
    lastLng = latlng ? latlng[1] : null;
    lastDistance = state.distance;

    if (!recording) return;

    const now = Date.now();

    telemetryRows.push({
        athleteId: selfAthleteId,
        timestamp: now,
        worldTime: state.worldTime ?? '',
        power: state.power ?? '',
        hr: state.heartrate ?? '',
        speed: state.speed ?? '',
        cadence: state.cadence ?? '',
        lat: lastLat ?? '',
        lng: lastLng ?? '',
        draft: state.draft ?? '',
        altitude: state.altitude ?? '',
        distance: state.distance ?? '',
        roadId: state.roadId ?? '',
        roadTime: state.roadTime ?? '',
        courseId: state.courseId ?? '',
        surface: currentSurface,
    });

    // Update UI
    sampleCountEl.textContent = telemetryRows.length;
    const elapsed = Math.floor((now - startTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    durationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}


function appendTagLog(tag) {
    const div = document.createElement('div');
    div.className = 'tag-entry';
    const elapsed = startTime ? Math.floor((tag.timestamp - startTime) / 1000) : 0;
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    div.innerHTML = `${mins}:${secs.toString().padStart(2, '0')} → <span class="tag-surface">${tag.surface}</span>`;
    tagLogList.prepend(div);
}


function downloadCSV() {
    if (telemetryRows.length === 0) return;

    const sessionTs = startTime ? new Date(startTime).toISOString().replace(/[:.]/g, '-') : 'unknown';

    const header = 'athlete_id,timestamp,world_time,power,hr,speed,cadence,lat,lng,draft,altitude,distance,road_id,road_time,course_id,surface';
    const lines = telemetryRows.map(r =>
        `${r.athleteId},${r.timestamp},${r.worldTime},${r.power},${r.hr},${r.speed},${r.cadence},${r.lat},${r.lng},${r.draft},${r.altitude},${r.distance},${r.roadId},${r.roadTime},${r.courseId},${r.surface}`
    );
    downloadFile(`surface_telemetry_${sessionTs}.csv`, [header, ...lines].join('\n'));
}


function downloadFile(filename, content) {
    const blob = new Blob([content], {type: 'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
