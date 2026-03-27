import * as Common from '/pages/src/common.mjs';

Common.enableSentry();

// Recording state
let recording = false;
let startTime = null;
let telemetryRows = [];        // [{athleteId, timestamp, power, hr, speed, cadence, lat, lng, draft}, ...]
const athleteProfiles = new Map(); // athleteId -> {firstName, lastName, weight, height, frameHue}
const riderIds = new Set();

// UI elements
let btnRecord, btnDownload, indicatorEl, statusTextEl, riderCountEl, sampleCountEl, durationEl;

export function main() {
    btnRecord = document.querySelector('.btn-record');
    btnDownload = document.querySelector('.btn-download');
    indicatorEl = document.querySelector('.indicator');
    statusTextEl = document.querySelector('.status-text');
    riderCountEl = document.querySelector('.rider-count');
    sampleCountEl = document.querySelector('.sample-count');
    durationEl = document.querySelector('.duration');

    btnRecord.addEventListener('click', toggleRecording);
    btnDownload.addEventListener('click', downloadCSV);

    Common.subscribe('nearby', onNearby);
}


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
    athleteProfiles.clear();
    riderIds.clear();

    btnRecord.textContent = 'Stop Recording';
    btnRecord.classList.add('active');
    btnDownload.disabled = true;
    indicatorEl.className = 'indicator recording';
    statusTextEl.textContent = 'Recording';
}


function stopRecording() {
    recording = false;

    btnRecord.textContent = 'Start Recording';
    btnRecord.classList.remove('active');
    btnDownload.disabled = telemetryRows.length === 0;
    indicatorEl.className = 'indicator stopped';
    statusTextEl.textContent = `Stopped — ${telemetryRows.length} samples`;
}


function onNearby(data) {
    if (!recording || !data || !data.length) {
        return;
    }

    const now = Date.now();

    for (const entry of data) {
        const id = entry.athleteId;
        const state = entry.state;
        if (!state) continue;

        // Record telemetry row
        const latlng = state.latlng;
        telemetryRows.push({
            athleteId: id,
            timestamp: now,
            worldTime: state.worldTime,
            power: state.power ?? '',
            hr: state.heartrate ?? '',
            speed: state.speed ?? '',
            cadence: state.cadence ?? '',
            lat: latlng ? latlng[0] : '',
            lng: latlng ? latlng[1] : '',
            draft: state.draft ?? '',
            altitude: state.altitude ?? '',
            distance: state.distance ?? '',
        });

        riderIds.add(id);

        // Capture/update profile info
        if (!athleteProfiles.has(id) || !athleteProfiles.get(id).weight) {
            const athlete = entry.athlete;
            athleteProfiles.set(id, {
                firstName: athlete?.sanitizedFullname?.split(/\s+/)[0] ?? '',
                lastName: athlete?.sanitizedFullname?.split(/\s+/).slice(1).join(' ') ?? '',
                weight: athlete?.weight ?? '',
                height: athlete?.height ?? '',
                frameHue: state.frameHue ?? '',
            });
        }
    }

    // Update UI
    riderCountEl.textContent = riderIds.size;
    sampleCountEl.textContent = telemetryRows.length;
    const elapsed = Math.floor((now - startTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    durationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}


async function downloadCSV() {
    if (telemetryRows.length === 0) return;

    const sessionTs = startTime ? new Date(startTime).toISOString().replace(/[:.]/g, '-') : 'unknown';

    // Telemetry CSV
    const telHeader = 'athlete_id,timestamp,world_time,power,hr,speed,cadence,lat,lng,draft,altitude,distance';
    const telLines = telemetryRows.map(r =>
        `${r.athleteId},${r.timestamp},${r.worldTime},${r.power},${r.hr},${r.speed},${r.cadence},${r.lat},${r.lng},${r.draft},${r.altitude},${r.distance}`
    );
    downloadFile(`telemetry_${sessionTs}.csv`, [telHeader, ...telLines].join('\n'));

    // Delay so the browser processes the first download before triggering the second
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Profiles CSV
    const profHeader = 'athlete_id,first_name,last_name,weight_kg,height_cm,frame_hue';
    const profLines = [];
    for (const [id, p] of athleteProfiles) {
        profLines.push(`${id},${csvEscape(p.firstName)},${csvEscape(p.lastName)},${p.weight},${p.height},${p.frameHue}`);
    }
    downloadFile(`profiles_${sessionTs}.csv`, [profHeader, ...profLines].join('\n'));
}


function csvEscape(value) {
    const s = String(value);
    if (s.includes(',') || s.includes('"') || s.includes('\n')) {
        return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
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
