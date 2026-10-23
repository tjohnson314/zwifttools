const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

for (const internal of [false, true]) {
    test(`saved replay requests roster refresh unless just fetched (internal=${internal})`, async () => {
        const requests = [];
        const context = vm.createContext({
            raceOpInProgress: false,
            setRaceBusy() {},
            showLoading() {},
            hideLoading() {},
            showStatus() {},
            async fetch(url, options) {
                requests.push({ url, body: JSON.parse(options.body) });
                return { ok: false, status: 502, async json() { return { error: 'Stop after load' }; } };
            },
        });
        const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
        vm.runInContext(source.slice(
            source.indexOf('async function loadRaceById('),
            source.indexOf('function initRaceData('),
        ), context);
        await context.loadRaceById('race_data_456', internal);
        assert.deepEqual(requests, [{
            url: '/api/race/load',
            body: { race_id: 'race_data_456', refresh_participants: !internal },
        }]);
    });
}
