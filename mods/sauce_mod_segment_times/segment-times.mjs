import * as Common from '/pages/src/common.mjs';
import {SegmentTracker, formatTime, validCourseId} from './tracker.mjs';

Common.enableSentry();

const tracker = new SegmentTracker();
const courses = new Map();
let resultsEl;
let statusEl;

export function main() {
    resultsEl = document.getElementById('results');
    statusEl = document.getElementById('status');
    Common.subscribe('athlete/self', entry => { void onEntry(entry); });
    Common.subscribe('nearby', entries => {
        if (Array.isArray(entries)) {
            for (const entry of entries) void onEntry(entry);
        }
    });
}

function loadCourse(courseId) {
    if (!courses.has(courseId)) {
        const pending = Common.rpc.getCourseSegments(courseId).then(segments => {
            if (!Array.isArray(segments)) throw new Error('Invalid segment catalog');
            const count = tracker.setCourseSegments(courseId, segments);
            if (!count) {
                statusEl.textContent = `No segment data available for course ${courseId}.`;
                statusEl.classList.add('error');
            } else {
                statusEl.textContent = 'Tracking nearby riders.';
                statusEl.classList.remove('error');
            }
            return true;
        }).catch(error => {
            console.error('Could not load Sauce segment catalog:', error);
            statusEl.textContent = `Could not load segments for course ${courseId}. Reopen the window to retry.`;
            statusEl.classList.add('error');
            return false;
        });
        courses.set(courseId, pending);
    }
    return courses.get(courseId);
}

async function onEntry(entry) {
    const courseId = entry?.state?.courseId;
    if (!validCourseId(courseId) || !await loadCourse(courseId)) return;
    for (const result of tracker.update(entry)) {
        const row = document.createElement('div');
        row.className = 'result';
        for (const text of [
            entry.athlete?.sanitizedFullname || `Rider ${entry.athleteId}`,
            result.name,
            formatTime(result.timeMs),
        ]) {
            const cell = document.createElement('span');
            cell.textContent = text;
            row.appendChild(cell);
        }
        const atBottom = resultsEl.scrollHeight - resultsEl.scrollTop - resultsEl.clientHeight < 10;
        resultsEl.appendChild(row);
        if (atBottom) resultsEl.scrollTop = resultsEl.scrollHeight;
    }
}
