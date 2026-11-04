const assert = require('node:assert/strict');
const test = require('node:test');

function reference() {
    return import('../mods/sauce_mod_race_pacer/reference.mjs');
}

function csv(rows, meta = 'course_id,route_id,name,date\n1,tempus-fugit,Race,2026-11-04') {
    return `${meta}\n${rows.join('\n')}\n`;
}

test('CSV course_id (Zwift world ID) maps to the live Sauce courseId', async () => {
    const { csvCourseId, WORLD_TO_COURSE } = await reference();
    assert.equal(csvCourseId({ course_id: '1' }), 6);    // Watopia
    assert.equal(csvCourseId({ course_id: '9' }), 13);   // Makuri Islands
    assert.equal(csvCourseId({ course_id: '13' }), 17);  // Scotland
    assert.equal(csvCourseId({ course_id: '' }), null);
    assert.equal(csvCourseId({ course_id: '99' }), null);
    assert.equal(Object.keys(WORLD_TO_COURSE).length, 13);
});

test('world-to-course table matches the power breakdown build script', async () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const { WORLD_TO_COURSE } = await reference();
    const py = fs.readFileSync(path.join(__dirname, '../tools/build_power_breakdown_data.py'), 'utf8');
    const block = py.slice(py.indexOf('_WORLD_TO_COURSE = {'), py.indexOf('}', py.indexOf('_WORLD_TO_COURSE = {')));
    const pairs = Object.fromEntries([...block.matchAll(/(\d+):\s*(\d+)/g)].map(m => [m[1], Number(m[2])]));
    assert.deepEqual({ ...WORLD_TO_COURSE }, pairs);
});

test('wrong course only when both courses are known and differ', async () => {
    const { isWrongCourse } = await reference();
    assert.equal(isWrongCourse(6, 6), false);
    assert.equal(isWrongCourse(6, 10), true);
    assert.equal(isWrongCourse(null, 10), false);
    assert.equal(isWrongCourse(6, undefined), false);
});

test('route label uses the slug, else the world name for numeric route IDs', async () => {
    const { routeLabel } = await reference();
    assert.equal(routeLabel({ route_id: 'tempus-fugit', course_id: '1' }), 'Tempus Fugit');
    assert.equal(routeLabel({ route_id: '1337403830', course_id: '1' }), 'Watopia');
    assert.equal(routeLabel({ route_id: '', course_id: '6' }), 'Bologna');
    assert.equal(routeLabel({ route_id: '', course_id: '' }), '');
});

test('parser reads optional power, speed and bucket columns', async () => {
    const { parseReferenceCsv } = await reference();
    const out = parseReferenceCsv(csv([
        'time_sec,distance_m,lat,lng,power_watts,speed_kmh,bucket',
        '0,0,,,300,36,1', '10,100,,,300,36,1', '20,200,,,250,,2',
    ]));
    assert.deepEqual(out.times, [0, 10, 20]);
    assert.deepEqual(out.powers, [300, 300, 250]);
    assert.deepEqual(out.speeds, [36, 36, null]);
    assert.deepEqual(out.buckets, [1, 1, 2]);
    assert.equal(out.meta.route_id, 'tempus-fugit');
});

test('parser skips rows with a blank distance or time instead of rejecting the file', async () => {
    const { parseReferenceCsv } = await reference();
    const out = parseReferenceCsv(csv([
        'time_sec,distance_m,lat,lng',
        '0,0,,', '1,,,', ',20,,', '2,20,,', '3,30,,',
    ]));
    assert.deepEqual(out.times, [0, 2, 3]);
    assert.deepEqual(out.distances, [0, 20, 30]);
    assert.equal(out.skipped, 2);
    assert.equal(out.powers, null);
    assert.equal(out.buckets, null);
});

test('parser still rejects non-numeric data and missing columns', async () => {
    const { parseReferenceCsv } = await reference();
    assert.throws(() => parseReferenceCsv(csv(['time_sec,distance_m', '0,0', '1,abc', '2,20'])), /line 5/);
    assert.throws(() => parseReferenceCsv(csv(['time_sec,dist', '0,0', '1,10', '2,20'])), /distance_m/);
    assert.throws(() => parseReferenceCsv('a\nb\nc'), /too short/);
});
