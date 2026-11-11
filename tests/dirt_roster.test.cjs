const assert = require('node:assert/strict');
const test = require('node:test');
const {dirtFilter, dirtRows, dirtSorted, dirtCSV, dirtStatsRows, dirtColumns, dirtStatsCoverage} = require('../static/js/dirt_roster.js');

const teams = [
    {team_id: 1854, name: 'RtB Yaks', time_zone: 'EMEA C', league: 'Tortellini', category: '04 Sapphire', riders: [
        {rider_id: 10157, name: 'Timothy Johnson', zwift_id: 7681059},
        {rider_id: 10152, name: 'Conor Galvin', zwift_id: 7471860},
    ]},
    {team_id: 2, name: 'Other Team', time_zone: 'US E', league: 'Linguine', category: '08 Silver', riders: [
        {rider_id: 10157, name: 'Timothy Johnson', zwift_id: 7681059},
    ]},
    {team_id: 3, name: 'Empty Team', time_zone: 'EMEA C', league: 'Tortellini', category: '04 Sapphire', riders: []},
];

test('filters league and category and preserves empty teams', () => {
    assert.equal(dirtFilter(teams, {league: 'Tortellini', category: '04 Sapphire'}).length, 2);
    assert.equal(dirtFilter(teams, {league: 'Linguine', time_zone: 'EMEA C'}).length, 0);
});

test('search finds a rider by name or Zwift ID without losing team assignments', () => {
    assert.equal(dirtRows(dirtFilter(teams, {q: '7681059'})).length, 2);
    const result = dirtRows(dirtFilter(teams, {q: 'conor'}));
    assert.equal(result.length, 1);
    assert.equal(result[0].team, 'RtB Yaks');
    assert.equal(dirtRows(dirtFilter(teams, {q: 'YAKS'})).length, 2);
    assert.equal(dirtRows(dirtFilter(teams, {team: '1854'})).length, 2);
});

test('sorting is numeric, immutable and leaves missing IDs last in either direction', () => {
    const rows = [{zwift_id: null}, {zwift_id: 100}, {zwift_id: 2}];
    assert.deepEqual(dirtSorted(rows, 'zwift_id', 1).map(row => row.zwift_id), [2, 100, null]);
    assert.deepEqual(dirtSorted(rows, 'zwift_id', -1).map(row => row.zwift_id), [100, 2, null]);
    assert.equal(rows[0].zwift_id, null);
});

test('CSV preserves raw names, quotes, missing IDs and multiple team assignments', () => {
    const rows = dirtRows(teams);
    rows[0].name = 'A, "Quoted"\nRider';
    rows[0].zwift_id = null;
    const csv = dirtCSV(rows);
    assert.ok(csv.includes('"A, ""Quoted""\nRider","","RtB Yaks"'));
    assert.ok(csv.includes('"Timothy Johnson","7681059","Other Team"'));
});

test('stats join by Zwift ID across assignments and preserve zero, raw extremes and nulls', () => {
    const rows = dirtStatsRows(dirtRows(teams), {riders: {'7681059': {status: 'ok', velo: {sprint: -18.123456, punch: 0}, power: {5: {live: {w: 9999.123456, wkg: -1.5}, best_90d: {w: 0, wkg: 0}}}}}});
    assert.equal(rows[0].velo_sprint, -18.123456);
    assert.equal(rows[0].velo_punch, 0);
    assert.equal(rows[0].power_5_live, 9999.123456);
    assert.equal(rows[0].power_5_best_90d, 0);
    assert.equal(rows[0].velo_climb, null);
    assert.equal(rows[1].stats_status, 'pending');
    assert.equal(rows[2].velo_sprint, rows[0].velo_sprint);
});

test('power uses source W/kg, never recalculating it from watts', () => {
    const rows = dirtStatsRows(dirtRows(teams), {riders: {'7681059': {status: 'ok', power: {1200: {live: {w: 287.29, wkg: 3.59}, best_90d: {w: 252, wkg: 3.76}}}}}}, 'wkg');
    assert.equal(rows[0].power_1200_live, 3.59);
    assert.equal(rows[0].power_1200_best_90d, 3.76);
});

test('vELO rows and CSV retain uncapped source precision', () => {
    const sprint = 1007.8017236673211;
    const stats = {riders: {'7681059': {status: 'ok', velo: {sprint}}}};
    const rows = dirtStatsRows(dirtRows(teams), stats);
    assert.equal(rows[0].velo_sprint, sprint);
    assert.equal(rows[2].velo_sprint, sprint);
    assert.equal(stats.riders['7681059'].velo.sprint, sprint);
    assert.ok(dirtCSV(rows, dirtColumns('velo')).includes(`"${sprint}"`));
});

test('stats tables omit category while the roster keeps it', () => {
    assert.equal(dirtColumns('velo').length, 8);
    for (const tab of ['velo', 'power']) assert.ok(!dirtColumns(tab).some(([key]) => key === 'category'));
    assert.ok(dirtColumns('roster').some(([key]) => key === 'category'));
});

test('power columns and CSV show only the selected source for all seven intervals', () => {
    const row = {name: 'Tim', power_5_live: 10.35, power_5_best_90d: 10.36};
    for (const source of ['live', 'best_90d']) {
        for (const unit of ['w', 'wkg']) {
            const columns = dirtColumns('power', unit, source);
            assert.equal(columns.length, 9);
            assert.deepEqual(columns.slice(2).map(([key]) => key), [5, 15, 30, 60, 120, 300, 1200].map(duration => `power_${duration}_${source}`));
            const csv = dirtCSV([row], columns);
            assert.ok(csv.includes(source === 'live' ? '"10.35"' : '"10.36"'));
            assert.ok(!csv.includes(source === 'live' ? '"10.36"' : '"10.35"'));
            assert.ok(csv.includes(`"20 minutes ${source === 'live' ? 'Live estimate' : '90-day best'} (${unit === 'wkg' ? 'W/kg' : 'W'})"`));
        }
    }
});

test('source coverage counts gaps but not pending profiles, unavailable profiles or zero metrics', () => {
    const velo = Object.fromEntries(['endurance', 'pursuit', 'sprint', 'punch', 'climb', 'timeTrial'].map(key => [key, 0]));
    const power = Object.fromEntries([5, 15, 30, 60, 120, 300, 1200].map(duration => [duration, {live: {w: 0, wkg: 0}, best_90d: {w: 0, wkg: 0}}]));
    assert.deepEqual(dirtStatsCoverage({riders: {
        1: {status: 'ok', velo, power},
        2: {status: 'unavailable'},
        3: {status: 'ok', velo: {}, power: {}},
    }}), {velo: 1, power: 1});
    assert.deepEqual(dirtStatsCoverage(null), {velo: 0, power: 0});
});