(() => {
    'use strict';

    const form = document.getElementById('event-form');
    const input = document.getElementById('event-id');
    const loadButton = document.getElementById('load-event');
    const status = document.getElementById('event-status');
    const result = document.getElementById('event-result');
    const output = document.getElementById('event-json');
    const summary = document.getElementById('event-summary');
    const copyButton = document.getElementById('copy-json');
    const download = document.getElementById('download-json');
    let downloadUrl;

    function showStatus(message, error = false) {
        status.textContent = message;
        status.dataset.error = String(error);
    }

    async function loadEvent(event) {
        event?.preventDefault();
        if (loadButton.disabled) return;
        result.hidden = true;
        output.textContent = '';
        summary.innerHTML = '';
        if (downloadUrl) {
            URL.revokeObjectURL(downloadUrl);
            downloadUrl = undefined;
        }
        download.removeAttribute('href');
        const eventId = input.value.trim();
        if (!/^[0-9]{1,20}$/.test(eventId) || /^0+$/.test(eventId)) {
            showStatus('Enter a positive numeric event ID (up to 20 digits).', true);
            return;
        }
        const normalizedId = eventId.replace(/^0+/, '');
        input.value = normalizedId;
        loadButton.disabled = true;
        input.disabled = true;
        showStatus('Loading event settings...');
        try {
            const response = await fetch(`/api/events/${normalizedId}?view=readable`, { cache: 'no-store' });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `Could not load event (HTTP ${response.status}).`);
            }
            // The server formats the summary and keeps raw JSON in a string to preserve large IDs.
            const data = await response.json();
            const text = data.raw_json;
            summary.innerHTML = data.summary_html;
            output.textContent = text;
            downloadUrl = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
            download.href = downloadUrl;
            download.download = `zwift-event-${normalizedId}.json`;
            result.hidden = false;
            const url = new URL(window.location.href);
            url.searchParams.set('event_id', normalizedId);
            window.history.replaceState(null, '', url);
            showStatus(`Loaded settings for event ${normalizedId}.`);
        } catch (error) {
            showStatus(`Unable to load event settings: ${error.message}`, true);
        } finally {
            loadButton.disabled = false;
            input.disabled = false;
        }
    }

    form.addEventListener('submit', loadEvent);
    copyButton.addEventListener('click', async () => {
        try {
            if (!navigator.clipboard) {
                throw new Error('Clipboard access is unavailable. Select the JSON or use Download JSON instead.');
            }
            await navigator.clipboard.writeText(output.textContent);
            showStatus('JSON copied to clipboard.');
        } catch (error) {
            showStatus(`Could not copy JSON: ${error.message}`, true);
        }
    });

    const initialId = new URLSearchParams(window.location.search).get('event_id');
    if (initialId !== null) {
        input.value = initialId;
        loadEvent();
    }
})();
