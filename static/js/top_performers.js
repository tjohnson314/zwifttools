/* Top Performers scatter: CdA reduction vs weight reduction, per frame + wheel,
   relative to the (always Level 0) Zwift Carbon + Zwift 32mm Carbon reference bike.
   Frame and wheel lists mirror the ZwiftInsider Top Performers chart. */
(function () {
  'use strict';

  // Distinct palette assigned per frame (in chart order).
  const PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
    '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9a6324',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
    '#ffe119', '#e6beff',
  ];

  let DATA = null;
  let chart = null;
  let level = 0;                      // 0 = stock (index 0), 1 = maxed (index 1)
  const frameColor = {};             // frame id -> color
  const selectedFrames = new Set();  // frame ids
  const selectedWheels = new Set();  // wheel ids

  function wheelBiasFor(frame, wheel) {
    return frame.type === 'TT' ? wheel.cdaTT : wheel.cda;
  }

  // Reference totals — ALWAYS Level 0, independent of the toggle.
  function referenceTotals() {
    const rf = DATA.referenceFrame;
    const rw = DATA.referenceWheel;
    const rwBias = rf.type === 'TT' ? rw.cdaTT : rw.cda;
    return { cda: rf.cda + rwBias, wt: rf.weight + rw.weight };
  }

  function buildPoints() {
    const ref = referenceTotals();
    const points = [];

    DATA.frames.forEach(frame => {
      if (!selectedFrames.has(frame.id)) return;
      const color = frameColor[frame.id];

      if (frame.halo && frame.builtInWheel) {
        const totalCda = frame.cda[level] + frame.builtInWheel.cda;
        const totalWt = frame.weight[level] + frame.builtInWheel.weight;
        points.push({
          x: ref.cda - totalCda,
          y: ref.wt - totalWt,
          color,
          label: `${frame.name} (built-in ${frame.builtInWheel.name})`,
        });
        return;
      }

      DATA.wheels.forEach(wheel => {
        if (!selectedWheels.has(wheel.id)) return;
        const totalCda = frame.cda[level] + wheelBiasFor(frame, wheel);
        const totalWt = frame.weight[level] + wheel.weight;
        points.push({
          x: ref.cda - totalCda,
          y: ref.wt - totalWt,
          color,
          label: `${frame.name} + ${wheel.name}`,
        });
      });
    });
    return points;
  }

  function render() {
    const points = buildPoints();
    const datasets = [{
      label: 'Bikes',
      data: points,
      backgroundColor: points.map(p => p.color),
      borderColor: points.map(p => p.color),
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 8,
    }];

    const config = {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.raw;
                return [
                  p.label,
                  `CdA reduction: ${p.x >= 0 ? '+' : ''}${p.x.toFixed(5)} m²`,
                  `Weight reduction: ${p.y >= 0 ? '+' : ''}${p.y.toFixed(0)} g`,
                ];
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'CdA reduction vs reference (m²)  →  more aero', color: '#bbb' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#aaa' },
          },
          y: {
            title: { display: true, text: 'Weight reduction vs reference (g)  →  lighter', color: '#bbb' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#aaa' },
          },
        },
      },
    };

    if (chart) { chart.destroy(); }
    chart = new Chart(document.getElementById('scatter'), config);
    document.getElementById('pointCount').textContent =
      `${points.length} data point${points.length === 1 ? '' : 's'} shown ` +
      `(frames at level ${level === 0 ? '0 / stock' : '5 / maxed'}, reference fixed at level 0).`;
  }

  function buildFrameList() {
    const box = document.getElementById('frameList');
    box.innerHTML = '';
    DATA.frames.forEach(frame => {
      const row = document.createElement('label');
      row.className = 'check-item';
      row.innerHTML =
        `<input type="checkbox" ${selectedFrames.has(frame.id) ? 'checked' : ''}>` +
        `<span class="swatch" style="background:${frameColor[frame.id]}"></span>${frame.name}`;
      row.querySelector('input').addEventListener('change', (e) => {
        if (e.target.checked) selectedFrames.add(frame.id); else selectedFrames.delete(frame.id);
        updateFrameCount();
        render();
      });
      box.appendChild(row);
    });
    updateFrameCount();
  }

  function buildWheelList() {
    const box = document.getElementById('wheelList');
    box.innerHTML = '';
    DATA.wheels.forEach(wheel => {
      const row = document.createElement('label');
      row.className = 'check-item';
      row.innerHTML =
        `<input type="checkbox" ${selectedWheels.has(wheel.id) ? 'checked' : ''}>` +
        `<span>${wheel.name}</span>`;
      row.querySelector('input').addEventListener('change', (e) => {
        if (e.target.checked) selectedWheels.add(wheel.id); else selectedWheels.delete(wheel.id);
        updateWheelCount();
        render();
      });
      box.appendChild(row);
    });
    updateWheelCount();
  }

  function updateFrameCount() {
    document.getElementById('frameCount').textContent =
      `${selectedFrames.size} of ${DATA.frames.length} selected`;
  }
  function updateWheelCount() {
    document.getElementById('wheelCount').textContent =
      `${selectedWheels.size} of ${DATA.wheels.length} selected`;
  }

  function wireControls() {
    document.getElementById('lvl0Btn').addEventListener('click', () => setLevel(0));
    document.getElementById('lvl5Btn').addEventListener('click', () => setLevel(1));

    document.getElementById('frameAll').addEventListener('click', () => {
      DATA.frames.forEach(f => selectedFrames.add(f.id));
      buildFrameList(); render();
    });
    document.getElementById('frameNone').addEventListener('click', () => {
      selectedFrames.clear();
      buildFrameList(); render();
    });
    document.getElementById('wheelAll').addEventListener('click', () => {
      DATA.wheels.forEach(w => selectedWheels.add(w.id));
      buildWheelList(); render();
    });
    document.getElementById('wheelNone').addEventListener('click', () => {
      selectedWheels.clear();
      buildWheelList(); render();
    });
  }

  function setLevel(lv) {
    level = lv;
    document.getElementById('lvl0Btn').classList.toggle('active', lv === 0);
    document.getElementById('lvl5Btn').classList.toggle('active', lv === 1);
    render();
  }

  fetch('/api/top_performers')
    .then(r => r.json())
    .then(data => {
      DATA = data;
      DATA.frames.forEach((f, i) => { frameColor[f.id] = PALETTE[i % PALETTE.length]; });
      DATA.frames.forEach(f => { if (f.default) selectedFrames.add(f.id); });
      DATA.wheels.forEach(w => { if (w.default) selectedWheels.add(w.id); });
      buildFrameList();
      buildWheelList();
      wireControls();
      render();
    })
    .catch(err => {
      document.getElementById('pointCount').textContent = 'Failed to load data: ' + err;
    });
})();
