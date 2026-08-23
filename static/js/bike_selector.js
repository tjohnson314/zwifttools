'use strict';

// Memoised frame/wheel fetch shared across every BikeSelector instance on a page
// (the comparison page mounts two), so the data is only requested once.
let _bikeDataPromise = null;

function _loadBikeData() {
    if (!_bikeDataPromise) {
        _bikeDataPromise = Promise.all([fetch('/api/frames'), fetch('/api/wheels')])
            .then(async ([framesResp, wheelsResp]) => {
                if (!framesResp.ok) throw new Error(`Failed to load frames (${framesResp.status})`);
                if (!wheelsResp.ok) throw new Error(`Failed to load wheels (${wheelsResp.status})`);
                return { frames: await framesResp.json(), wheels: await wheelsResp.json() };
            })
            .catch(err => { _bikeDataPromise = null; throw err; });  // allow a later retry
    }
    return _bikeDataPromise;
}

/**
 * Reusable single-bike configuration selector.
 *
 * Manages a frame / wheels / upgrade-level trio of <select> elements plus a
 * "Total weight" + "Total CdA bias" summary, backed by the same /api/frames,
 * /api/wheels and /api/bike_stats endpoints the bike-comparison page uses — so
 * both pages present bike choices identically.
 *
 * The class owns no persistence; callers restore state via setConfig() and read
 * it via getConfig(), and are notified of any change through the onChange hook.
 */
class BikeSelector {
    /**
     * @param {Object}   opts
     * @param {string}   opts.frameSelect  id of the frame <select>
     * @param {string}   opts.wheelSelect  id of the wheels <select>
     * @param {string}   opts.levelSelect  id of the upgrade-level <select>
     * @param {string}   opts.weightOut    id of the Total-weight value element
     * @param {string}   opts.cdaOut       id of the Total-CdA-bias value element
     * @param {Function} [opts.onChange]   called (with this) after every change
     * @param {Function} [opts.formatWeight] kg -> display string
     * @param {Function} [opts.formatCda]  m² -> display string
     * @param {string}   [opts.framePlaceholder] leading empty frame option label
     */
    constructor(opts) {
        this.frameSel  = document.getElementById(opts.frameSelect);
        this.wheelSel  = document.getElementById(opts.wheelSelect);
        this.levelSel  = document.getElementById(opts.levelSelect);
        this.weightOut = document.getElementById(opts.weightOut);
        this.cdaOut    = document.getElementById(opts.cdaOut);
        this.onChange  = opts.onChange || (() => {});
        this.formatWeight = opts.formatWeight || ((kg) => `${kg.toFixed(2)} kg`);
        this.formatCda = opts.formatCda ||
            ((m2) => `${m2 >= 0 ? '+' : ''}${m2.toFixed(4)} m²`);
        this.framePlaceholder = opts.framePlaceholder || null;

        this.frames = [];
        this.wheels = [];
        this._stats = null;    // { weightKg, cdaBias } once loaded
        this._ready = false;

        this.frameSel.addEventListener('change', () => this._onFrameChange());
        this.wheelSel.addEventListener('change', () => this._updateStats());
        this.levelSel.addEventListener('change', () => this._updateStats());
    }

    // Fetch (once per page) the frame + wheel catalogue used by all selectors.
    static loadData() { return _loadBikeData(); }

    async load() {
        const { frames, wheels } = await _loadBikeData();
        this.frames = frames;
        this.wheels = wheels;
        this._populateFrames();
        this._syncWheels();
        this._syncLevels();
        await this._updateStats();
    }

    getConfig() {
        return {
            frameId: this.frameSel.value,
            wheelId: this.wheelSel.value || null,
            level: parseInt(this.levelSel.value, 10) || 0,
        };
    }

    setConfig({ frameId, wheelId, level } = {}) {
        if (frameId != null && [...this.frameSel.options].some(o => o.value === frameId)) {
            this.frameSel.value = frameId;
        }
        this._syncWheels();
        this._syncLevels();
        if (wheelId != null && [...this.wheelSel.options].some(o => o.value === wheelId)) {
            this.wheelSel.value = wheelId;
        }
        if (level != null && [...this.levelSel.options].some(o => o.value === String(level))) {
            this.levelSel.value = String(level);
        }
        return this._updateStats();
    }

    isReady() { return this._ready; }

    // Re-render the cached stats (e.g. after a unit toggle) without refetching.
    renderStats() {
        if (!this._stats) { this._renderBlank(); return; }
        if (this.weightOut) {
            this.weightOut.textContent = this._stats.weightKg != null
                ? this.formatWeight(this._stats.weightKg) : 'N/A';
        }
        if (this.cdaOut) {
            this.cdaOut.textContent = this._stats.cdaBias != null
                ? this.formatCda(this._stats.cdaBias) : 'N/A';
        }
    }

    // ── internals ─────────────────────────────────────────────────────────────
    _populateFrames() {
        const opts = this.frames.map(f =>
            `<option value="${f.id}">${f.name}${f.cdaBias != null
                ? ` (CdA ${f.cdaBias >= 0 ? '+' : ''}${f.cdaBias} · ${Math.round(f.weightG)}g)`
                : ''}</option>`);
        this.frameSel.innerHTML =
            (this.framePlaceholder ? `<option value="">${this.framePlaceholder}</option>` : '') +
            opts.join('');
    }

    _frameById(id) { return this.frames.find(f => f.id === id); }

    _frameType(id) {
        const f = this._frameById(id);
        return f ? (f.frameType || 'Standard') : 'Standard';
    }

    _frameHasBuiltInWheels(id) {
        const f = this._frameById(id);
        return f ? f.hasBuiltInWheels : false;
    }

    _compatibleWheels(frameType) {
        return this.wheels.filter(w => {
            if (w.id === '') return false;   // skip the built-in placeholder
            const fits = (w.fitsFrame || 'Standard,TT').split(',').map(s => s.trim());
            return fits.includes(frameType);
        });
    }

    _syncWheels() {
        const frameId = this.frameSel.value;
        const previous = this.wheelSel.value;
        if (!frameId) {
            this.wheelSel.innerHTML = '<option value="">Select a frame first…</option>';
            this.wheelSel.value = '';
            this.wheelSel.disabled = true;
            return;
        }
        if (this._frameHasBuiltInWheels(frameId)) {
            this.wheelSel.innerHTML = '<option value="">(Built-in wheels)</option>';
            this.wheelSel.value = '';
            this.wheelSel.disabled = true;
            return;
        }
        const frameType = this._frameType(frameId);
        const isTT = frameType === 'TT';
        const compatible = this._compatibleWheels(frameType);
        this.wheelSel.innerHTML = compatible.map(w => {
            // TT frames get the wheel's TT-specific (more aero) bias.
            const bias = (isTT && w.cdaBiasTt != null) ? w.cdaBiasTt : w.cdaBias;
            const label = bias != null
                ? ` (CdA ${bias >= 0 ? '+' : ''}${bias} · ${Math.round(w.weightG)}g)` : '';
            return `<option value="${w.id}">${w.name}${label}</option>`;
        }).join('');
        this.wheelSel.disabled = false;
        if (compatible.some(w => w.id === previous)) this.wheelSel.value = previous;
    }

    _syncLevels() {
        const frame = this._frameById(this.frameSel.value);
        const stages = frame && frame.upgradeStages;
        const previous = this.levelSel.value;
        const labelFor = (lvl) => {
            if (lvl === 0) return 'Level 0 (Stock)';
            if (!stages || !stages[lvl] || !stages[0]) return `Level ${lvl}`;
            const dW = Math.round((stages[lvl].weightG || 0) - (stages[0].weightG || 0));
            const dCda = (stages[lvl].cdaBias || 0) - (stages[0].cdaBias || 0);
            return `Level ${lvl} (CdA ${dCda >= 0 ? '+' : ''}${dCda.toFixed(4)} · ${dW > 0 ? '+' : ''}${dW} g)`;
        };
        this.levelSel.innerHTML = [0, 1, 2, 3, 4, 5]
            .map(l => `<option value="${l}">${labelFor(l)}</option>`).join('');
        if ([...this.levelSel.options].some(o => o.value === previous)) {
            this.levelSel.value = previous;
        }
    }

    _onFrameChange() {
        this._syncWheels();
        this._syncLevels();
        this._updateStats();
    }

    async _updateStats() {
        const frameId = this.frameSel.value;
        if (!frameId) {
            this._stats = null;
            this._ready = false;
            this._renderBlank();
            this.onChange(this);
            return;
        }
        const wheelId = this.wheelSel.value;
        const level = this.levelSel.value;
        try {
            const resp = await fetch(
                `/api/bike_stats?frame_id=${encodeURIComponent(frameId)}` +
                `&wheel_id=${encodeURIComponent(wheelId)}` +
                `&upgrade_level=${encodeURIComponent(level)}`);
            if (resp.ok) {
                const s = await resp.json();
                // Tolerate a lagging backend that still sends weight_g (a kg value).
                const weightKg = s.weight_kg != null ? s.weight_kg : s.weight_g;
                this._stats = { weightKg, cdaBias: s.cda_bias };
                this._ready = true;
                this.renderStats();
            } else {
                this._stats = null;
                this._ready = false;
                this._renderBlank();
            }
        } catch (e) {
            console.error('Failed to load bike stats:', e);
            this._stats = null;
            this._ready = false;
            this._renderBlank();
        }
        this.onChange(this);
    }

    _renderBlank() {
        if (this.weightOut) this.weightOut.textContent = '—';
        if (this.cdaOut) this.cdaOut.textContent = '—';
    }
}
