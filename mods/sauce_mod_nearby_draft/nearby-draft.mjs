import * as Common from '/pages/src/common.mjs';
import {MODEL} from './bike-model.mjs';
import * as Surface from './surface.mjs';
import {DEFAULTS, DraftTracker, selectNearby, draftGeometry} from './draft-model.mjs';

Common.enableSentry();

let settings;
let self;
let nearby = [];
let rowsEl, emptyEl, countEl;
let tracker = new DraftTracker();
const rows = new Map();

function loadSettings() {
    settings = {...DEFAULTS};
    for (const key of Object.keys(DEFAULTS)) {
        const value = Common.settingsStore.get(key);
        if (value !== undefined) settings[key] = value;
    }
}

export function main() {
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();
    rowsEl = document.querySelector('.draft-rows');
    emptyEl = document.querySelector('.empty');
    countEl = document.querySelector('.rider-count');
    Common.settingsStore.addEventListener('changed', () => {
        loadSettings();
        tracker = new DraftTracker();
        refresh();
    });
    Common.subscribe('athlete/self', data => {
        if (self && String(self.athleteId) !== String(data?.athleteId)) tracker = new DraftTracker();
        self = data;
        refresh();
    });
    Common.subscribe('nearby', data => {
        nearby = Array.isArray(data) ? data : [];
        refresh();
    });
}

function refresh() {
    const watching = nearby.find(entry => entry.watching);
    const selected = selectNearby(nearby, self?.athleteId, watching?.athleteId ?? self?.athleteId);
    const selfIndex = selected.findIndex(entry => entry.isSelf);
    if (self?.state && (selfIndex >= 0 || !watching || String(watching.athleteId) === String(self.athleteId))) {
        const nearbySelf = selfIndex >= 0 ? selected[selfIndex] : null;
        const latest = nearbySelf?.state.worldTime > self.state.worldTime ? nearbySelf : self;
        const ownEntry = {...self, ...latest,
            athlete: {...self.athlete, ...nearbySelf?.athlete}, gap: 0, isSelf: true};
        if (selfIndex >= 0) selected[selfIndex] = ownEntry;
        else selected.push(ownEntry);
        selected.sort((left, right) => left.gap - right.gap);
    }
    const ids = selected.map(entry => String(entry.athleteId));
    tracker.retain(ids);
    for (const [id, row] of rows) {
        if (!ids.includes(id)) {
            row.element.remove();
            rows.delete(id);
        }
    }
    selected.forEach((entry, index) => {
        const id = String(entry.athleteId);
        let row = rows.get(id);
        if (!row) {
            row = createRow();
            rows.set(id, row);
        }
        if (rowsEl.children[index] !== row.element) rowsEl.insertBefore(row.element, rowsEl.children[index] || null);
        Surface.preload(entry.state.courseId);
        const {style} = Surface.lookupStyle(entry.state.courseId, entry.state.roadId, entry.state.roadTime);
        const summary = tracker.update(entry, settings, style);
        renderRow(row, entry, summary);
    });
    countEl.textContent = `${selected.length} rider${selected.length === 1 ? '' : 's'}`;
    emptyEl.hidden = selected.length > 0;
    emptyEl.textContent = !self?.state ? 'Waiting for rider data...' :
        watching && String(watching.athleteId) !== String(self?.athleteId) ?
            'Your rider is outside the nearby feed.' : 'No riders within 15 seconds.';
}

function createRow() {
    const element = document.createElement('div');
    element.className = 'rider-row';
    element.innerHTML = `
        <div class="rider-head"><span class="rider-name"></span><span class="rider-gap"></span></div>
        <div class="draft-head"><span class="draft-value"></span><span class="residual-value"></span></div>
        <div class="draft-bar"><div class="draft-eff"></div><div class="draft-res"></div><div class="draft-zero"></div><div class="draft-limit"></div></div>
        <div class="draft-scale"><span class="draft-min"></span><span class="draft-max"></span></div>`;
    const row = {element};
    for (const name of ['rider-name', 'rider-gap', 'draft-value', 'residual-value',
        'draft-eff', 'draft-res', 'draft-zero', 'draft-limit', 'draft-min', 'draft-max', 'draft-bar']) {
        row[name] = element.querySelector(`.${name}`);
    }
    return row;
}

function renderRow(row, entry, summary) {
    row.element.classList.toggle('self', entry.isSelf);
    row['rider-name'].textContent = entry.athlete?.sanitizedFullname || `Rider ${entry.athleteId}`;
    row['rider-gap'].textContent = entry.isSelf ? 'You' : `${entry.gap > 0 ? '+' : ''}${entry.gap.toFixed(1)}s`;
    const percent = summary.percent == null ? '' : ` (${summary.percent.toFixed(0)}% of max)`;
    row['draft-value'].textContent = `${summary.effective.toFixed(0)} W${percent}`;
    row['residual-value'].textContent = `${summary.residual.toFixed(0)} W`;
    row['draft-value'].title = 'Effective draft';
    row['residual-value'].title = 'Draft subtracted by the negative power-balance residual';
    row['draft-bar'].title = `Effective: ${summary.effective.toFixed(0)} W; residual: ${summary.residual.toFixed(0)} W; max draft: ${summary.maxDraft.toFixed(0)} W`;
    const geometry = draftGeometry(summary);
    row['draft-min'].textContent = geometry.minimum < 0 ? `${geometry.minimum.toFixed(0)} W` : '0';
    row['draft-max'].textContent = geometry.maximum === summary.maxDraft ?
        `max ${summary.maxDraft.toFixed(0)} W` : `${geometry.maximum.toFixed(0)} W`;
    row['draft-limit'].style.left = `calc(${geometry.limit}% - 1px)`;
    row['draft-limit'].title = `Max draft: ${summary.maxDraft.toFixed(0)} W`;
    row['draft-eff'].style.left = `${geometry.greenLeft}%`;
    row['draft-eff'].style.width = `${geometry.greenWidth}%`;
    row['draft-res'].style.left = `${geometry.blueLeft}%`;
    row['draft-res'].style.width = `${geometry.blueWidth}%`;
    row['draft-zero'].style.left = `${geometry.zero}%`;
}

export function settingsMain() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DEFAULTS});
    loadSettings();
    const framePicker = document.querySelector('[data-set="frameId"]');
    const wheelPicker = document.querySelector('[data-set="wheelId"]');
    for (const frame of MODEL.frames) framePicker.add(new Option(frame.name, frame.id));
    wheelPicker.add(new Option('Frame built-in wheels', ''));
    for (const wheel of MODEL.wheels) wheelPicker.add(new Option(`${wheel.make} ${wheel.model}`.trim(), wheel.id));
    document.querySelector('form').addEventListener('submit', event => event.preventDefault());
    for (const element of document.querySelectorAll('[data-set]')) {
        const key = element.dataset.set;
        if (element.type === 'checkbox') element.checked = settings[key];
        else element.value = settings[key];
        element.addEventListener('change', () => {
            if (!element.checkValidity()) {
                element.reportValidity();
                return;
            }
            const value = element.type === 'checkbox' ? element.checked :
                element.type === 'number' || key === 'upgradeLevel' ? Number(element.value) : element.value;
            settings[key] = value;
            Common.settingsStore.set(key, value);
        });
    }
}