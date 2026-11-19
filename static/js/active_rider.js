'use strict';

(() => {
    const STORAGE_KEY = 'zwifttools.activeRider';
    const form = document.getElementById('activeRiderForm');
    const input = document.getElementById('activeRiderId');
    const status = document.getElementById('activeRiderStatus');
    const submit = document.getElementById('setActiveRider');
    const clear = document.getElementById('clearActiveRider');
    let active = null;
    let storageWarning = '';
    let lookup = 0;

    function validRider(value) {
        return value && typeof value.id === 'string' && /^[1-9][0-9]*$/.test(value.id)
            && typeof value.name === 'string' && value.name.trim();
    }

    function render(message = '') {
        input.value = active?.id || '';
        clear.hidden = !active;
        status.textContent = message || [active ? `${active.name} \u00b7 ${active.id}` : 'No rider selected', storageWarning]
            .filter(Boolean).join('. ');
    }

    function url(path, id = active?.id) {
        if (!id) throw new Error('Enter a Zwift ID in the header to choose a rider.');
        const target = new URL(path, window.location.href);
        target.searchParams.set('zwift_id', id);
        return target.pathname + target.search + target.hash;
    }

    async function profile(id = active?.id) {
        if (!id) throw new Error('Enter a Zwift ID in the header to choose a rider.');
        if (!/^[1-9][0-9]*$/.test(String(id))) throw new Error('Zwift ID must be a positive number.');
        const response = await fetch(`/api/profiles/${encodeURIComponent(id)}`, { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Profile lookup failed (${response.status}).`);
        data.id = String(data.id);
        if (!validRider(data) || data.id !== String(id)) throw new Error('Zwift returned an invalid rider profile.');
        return data;
    }

    function persist(rider) {
        try {
            if (rider) localStorage.setItem(STORAGE_KEY, JSON.stringify(rider));
            else localStorage.removeItem(STORAGE_KEY);
        } catch (error) {
            console.warn('Could not save active rider:', error);
            storageWarning = 'Browser storage unavailable; rider selection is kept in this page URL';
        }
    }

    function navigate(id) {
        const target = new URL(window.location.href);
        target.searchParams.set('zwift_id', id || '');
        window.location.assign(target.href);
    }

    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
        if (validRider(saved)) active = { id: saved.id, name: saved.name };
        localStorage.removeItem('zwift_remember_email');
    } catch (error) {
        console.warn('Could not restore active rider:', error);
        storageWarning = 'Browser storage unavailable; use the Zwift ID selector';
    }
    render();

    const params = new URL(window.location.href).searchParams;
    const ready = (async () => {
        if (!params.has('zwift_id')) return;
        const id = params.get('zwift_id').trim();
        if (!id) {
            active = null;
            render();
            return;
        }
        try {
            const data = await profile(id);
            active = { id: data.id, name: data.name };
            persist(active);
            render();
        } catch (error) {
            active = null;
            render(error.message);
        }
    })();

    window.ZwiftRider = { get: () => active && { ...active }, url, profile, ready };
    form.addEventListener('submit', async event => {
        event.preventDefault();
        const request = ++lookup;
        submit.disabled = true;
        const id = input.value.trim();
        status.textContent = 'Looking up rider...';
        try {
            await ready;
            const data = await profile(id);
            if (request !== lookup) return;
            persist({ id: data.id, name: data.name });
            navigate(data.id);
        } catch (error) {
            if (request === lookup) render(`${error.message}${active ? ` Current rider: ${active.name} \u00b7 ${active.id}.` : ''}`);
        } finally {
            if (request === lookup) submit.disabled = false;
        }
    });
    clear.addEventListener('click', () => {
        lookup++;
        persist(null);
        navigate(null);
    });
    window.addEventListener('storage', event => {
        if (event.key === STORAGE_KEY || event.key === null) {
            const target = new URL(window.location.href);
            target.searchParams.delete('zwift_id');
            window.location.assign(target.href);
        }
    });
})();
