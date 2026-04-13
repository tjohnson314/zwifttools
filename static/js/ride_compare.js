/**
 * Ride Compare — overrides for the race replay engine to support
 * comparing arbitrary activities on the same route.
 *
 * Loaded after race_replay.js; replaces DOMContentLoaded init,
 * fetchRace, and handleAuth to adapt the UI.
 */

// Override handleAuth to redirect back to ride-compare
const _origHandleAuth = typeof handleAuth === 'function' ? handleAuth : null;
handleAuth = function () {
    if (isAuthenticated) {
        window.location.href = '/auth/logout';
    } else {
        window.location.href = '/auth/login?next=/ride-compare';
    }
};

// After race_replay.js's DOMContentLoaded runs (which calls bindEvents),
// we re-bind the fetch button to our custom handler.
document.addEventListener('DOMContentLoaded', () => {
    const fetchBtn = document.getElementById('fetch-btn');
    if (fetchBtn) {
        // Clone to remove all existing listeners (from race_replay.js bindEvents)
        const newBtn = fetchBtn.cloneNode(true);
        fetchBtn.parentNode.replaceChild(newBtn, fetchBtn);
        newBtn.addEventListener('click', () => fetchRideCompare());
    }

    // Auto-load from URL params (e.g. ?activity_ids=123,456)
    const params = new URLSearchParams(window.location.search);
    const urlIds = params.get('activity_ids');
    if (urlIds) {
        document.getElementById('activity-ids-input').value = urlIds;
        fetchRideCompare();
    }
});

// ---------------------------------------------------------------------------
// Fetch rides for comparison
// ---------------------------------------------------------------------------
async function fetchRideCompare() {
    const input = document.getElementById('activity-ids-input').value.trim();
    if (!input) return;

    showStatus('Fetching ride data from Zwift API...', 'info');
    showLoading('Connecting to Zwift API...');
    showProgress(0, 0, '');

    const url = `/api/ride_compare/fetch_stream?activity_ids=${encodeURIComponent(input)}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            hideProgress();
            hideLoading();
            showStatus(`Server error: ${response.status}`, 'error');
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalData = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let idx;
            while ((idx = buffer.indexOf('\n\n')) !== -1) {
                const chunk = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                for (const line of chunk.split('\n')) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.progress) {
                            showLoading(`Fetching ride ${data.current}/${data.total}: ${data.name}`);
                            showProgress(data.current, data.total, data.name);
                        } else {
                            finalData = data;
                        }
                    } catch {}
                }
            }
        }

        hideProgress();
        hideLoading();

        if (!finalData) {
            showStatus('No response received from server.', 'error');
            return;
        }

        if (finalData.error) {
            console.error('Ride compare fetch error:', finalData.error);
            showStatus(finalData.error, 'error');
            return;
        }

        if (finalData.success) {
            showStatus(finalData.message, 'success');
            await loadRaceById(finalData.race_id);

            // Update URL for sharing (remove activity_id set by initRaceData)
            const url = new URL(window.location);
            url.searchParams.delete('activity_id');
            url.searchParams.set('activity_ids', input);
            window.history.replaceState({}, '', url);
        }
    } catch (e) {
        hideProgress();
        hideLoading();
        console.error('Fetch stream error:', e);
        showStatus('Connection lost while fetching ride data.', 'error');
    }
}
