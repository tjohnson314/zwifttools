const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function renderStreamLinks({ riders, streams, currentTime = 100, minTime = 27 }) {
    const panel = { innerHTML: '' };
    const context = vm.createContext({
        document: { getElementById() { return panel; } },
        raceData: { riders, min_time: minTime },
        streamLinks: streams,
        currentTime,
    });
    const source = fs.readFileSync(path.join(__dirname, '../static/js/race_replay.js'), 'utf8');
    vm.runInContext(source.slice(
        source.indexOf('function updateStreamLinks() {'),
        source.indexOf('function formatTime(seconds) {'),
    ), context);
    context.updateStreamLinks();
    return panel.innerHTML;
}

function stream(playerId, offsetSeconds = 600) {
    return {
        zwift_player_id: playerId,
        streamer_name: `Rider ${playerId}`,
        youtube_url: `https://www.youtube.com/watch?v=video${playerId}`,
        offset_seconds: offsetSeconds,
        stream_title: 'Race stream',
    };
}

test('stream timestamps shift elapsed replay time by each rider alignment offset', () => {
    const html = renderStreamLinks({
        riders: [
            { player_id: 1, ttt_time_offset: -27 },
            { player_id: 2, ttt_time_offset: 15 },
        ],
        streams: [stream(1), stream('2')],
    });
    assert.match(html, /video1&t=700s/);
    assert.match(html, /Rider 1 \(11:40\)/);
    assert.match(html, /video2&t=658s/);
    assert.match(html, /Rider 2 \(10:58\)/);
});

test('stream timestamps retain the elapsed replay clock and round only the final timestamp', () => {
    const options = {
        riders: [{ player_id: 1, ttt_time_offset: -16.9 }],
        streams: [stream(1)],
        currentTime: 100.25,
    };
    const html = renderStreamLinks({ ...options, minTime: 27 });
    assert.match(html, /video1&t=690s/);
    assert.match(renderStreamLinks({ ...options, minTime: 40 }), /video1&t=677s/);
});

test('activity 2243088276573454336 moves the matched stream from 14:00 to about 14:30', () => {
    const html = renderStreamLinks({
        riders: [
            { activity_id: '2243088276573454336', player_id: 7681059, ttt_time_offset: -25.147284278436928 },
            { activity_id: '2243086331540488224', player_id: 576041, ttt_time_offset: -25.081417983675692 },
        ],
        streams: [stream(576041, 840)],
        currentTime: 23,
        minTime: 23,
    });
    assert.match(html, /video576041&t=865s/);
    assert.match(html, /Rider 576041 \(14:25\)/);
});

test('missing riders or activity offsets default to zero without borrowing another rider offset', () => {
    const html = renderStreamLinks({
        riders: [
            { player_id: 1, ttt_time_offset: -27 },
            { player_id: 2, ttt_time_offset: null },
            { player_id: 3 },
            { player_id: 4, ttt_time_offset: 0 },
            { player_id: null, ttt_time_offset: -90 },
        ],
        streams: [stream(2), stream(3), stream(4), stream(5), stream(null)],
    });
    for (const playerId of [2, 3, 4, 5, null]) {
        assert.ok(html.includes(`video${playerId}&t=673s`));
    }
});

test('stream timestamp formatting supports hour-long streams', () => {
    const html = renderStreamLinks({
        riders: [{ player_id: 1, ttt_time_offset: -27 }],
        streams: [stream(1, 3600)],
    });
    assert.match(html, /video1&t=3700s/);
    assert.match(html, /Rider 1 \(1:01:40\)/);
});

test('empty stream lists clear stale links', () => {
    assert.equal(renderStreamLinks({ riders: [], streams: [] }), '');
});