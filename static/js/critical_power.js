let regressionChart = null;
let expectedPowerChart = null;

function parseDurationSeconds(input) {
    const raw = (input || "").trim();
    if (!raw) {
        return null;
    }

    if (raw.includes(":")) {
        const parts = raw.split(":").map(s => s.trim());
        if (parts.length !== 2) {
            return null;
        }

        const mins = Number(parts[0]);
        const secs = Number(parts[1]);
        if (!Number.isFinite(mins) || !Number.isFinite(secs) || mins < 0 || secs < 0 || secs >= 60) {
            return null;
        }
        return mins * 60 + secs;
    }

    const seconds = Number(raw);
    if (!Number.isFinite(seconds) || seconds <= 0) {
        return null;
    }
    return seconds;
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}:${String(secs).padStart(2, "0")}`;
}

function formatSigned(value, digits = 1) {
    const rounded = Number(value).toFixed(digits);
    return (value > 0 ? "+" : "") + rounded;
}

function createRows() {
    const dataRows = document.getElementById("dataRows");
    const defaults = [
        { duration: "300", power: "370" },
        { duration: "600", power: "330" },
        { duration: "1200", power: "295" },
        { duration: "1800", power: "275" }
    ];

    dataRows.innerHTML = "";
    for (let i = 0; i < 4; i += 1) {
        const row = document.createElement("div");
        row.className = "grid data-row";
        row.innerHTML = `
            <div class="index-pill">${i + 1}</div>
            <input id="duration-${i}" type="text" inputmode="numeric" placeholder="e.g. 300 or 5:00" value="${defaults[i].duration}">
            <input id="power-${i}" type="number" min="1" step="1" placeholder="Watts" value="${defaults[i].power}">
        `;
        dataRows.appendChild(row);
    }
}

function collectDataPoints() {
    const points = [];

    for (let i = 0; i < 4; i += 1) {
        const durationInput = document.getElementById(`duration-${i}`);
        const powerInput = document.getElementById(`power-${i}`);

        const rawDuration = durationInput.value.trim();
        const rawPower = powerInput.value.trim();

        if (!rawDuration && !rawPower) {
            continue;
        }

        const duration = parseDurationSeconds(rawDuration);
        const power = Number(rawPower);

        if (!duration || !Number.isFinite(power) || power <= 0) {
            return { error: `Invalid values on point ${i + 1}. Duration must be seconds or mm:ss, and power must be > 0.` };
        }

        points.push({
            index: i + 1,
            duration,
            power,
            work: duration * power
        });
    }

    if (points.length < 2) {
        return { error: "Enter at least 2 valid data points to run regression." };
    }

    return { points };
}

function runRegression(points) {
    const n = points.length;
    const x = points.map(p => p.duration);
    const y = points.map(p => p.work);

    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;

    let sxx = 0;
    let syy = 0;
    let sxy = 0;

    for (let i = 0; i < n; i += 1) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if (sxx === 0 || syy === 0) {
        return { error: "Regression could not be computed. Use varied durations and powers." };
    }

    const cp = sxy / sxx;
    const wPrime = meanY - cp * meanX;
    const r = sxy / Math.sqrt(sxx * syy);

    const modeled = points.map(p => {
        const expectedPower = cp + (wPrime / p.duration);
        const diff = p.power - expectedPower;
        return {
            ...p,
            expectedPower,
            diff
        };
    });

    return { cp, wPrime, r, modeled };
}

function renderChart(points, cp, wPrime) {
    const sortedPoints = [...points].sort((a, b) => a.duration - b.duration);
    const maxX = sortedPoints[sortedPoints.length - 1].duration;

    const lineData = [
        { x: 0, y: wPrime },
        { x: maxX, y: cp * maxX + wPrime }
    ];

    const scatterData = sortedPoints.map(p => ({ x: p.duration, y: p.work }));

    if (regressionChart) {
        regressionChart.destroy();
    }

    const ctx = document.getElementById("regressionChart").getContext("2d");
    regressionChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Actual Work",
                    data: scatterData,
                    pointRadius: 6,
                    pointHoverRadius: 7,
                    backgroundColor: "#58d3ff"
                },
                {
                    label: "Regression Line",
                    type: "line",
                    data: lineData,
                    borderColor: "#f5a623",
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#f4f7ff"
                    }
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const duration = context.parsed.x;
                            const work = context.parsed.y;
                            return `${context.dataset.label}: ${Math.round(work)} J at ${formatDuration(duration)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    title: {
                        display: true,
                        text: "Duration (seconds)",
                        color: "#9cb0d3"
                    },
                    ticks: {
                        color: "#9cb0d3"
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.08)"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Work (J)",
                        color: "#9cb0d3"
                    },
                    ticks: {
                        color: "#9cb0d3"
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.08)"
                    }
                }
            }
        }
    });
}

function renderExpectedPowerChart(points, cp, wPrime) {
    const sortedPoints = [...points].sort((a, b) => a.duration - b.duration);
    const minX = sortedPoints[0].duration;
    const maxX = sortedPoints[sortedPoints.length - 1].duration;

    const padding = Math.max(1, (maxX - minX) * 0.05);
    const start = Math.max(1, minX - padding);
    const end = maxX + padding;
    const steps = 100;

    const curveData = [];
    for (let i = 0; i <= steps; i += 1) {
        const t = start + ((end - start) * (i / steps));
        curveData.push({ x: t, y: cp + (wPrime / t) });
    }

    const actualData = sortedPoints.map(p => ({ x: p.duration, y: p.power }));

    if (expectedPowerChart) {
        expectedPowerChart.destroy();
    }

    const ctx = document.getElementById("expectedPowerChart").getContext("2d");
    expectedPowerChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Actual Power",
                    data: actualData,
                    pointRadius: 6,
                    pointHoverRadius: 7,
                    backgroundColor: "#58d3ff"
                },
                {
                    label: "Expected Power Curve",
                    type: "line",
                    data: curveData,
                    borderColor: "#43d587",
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#f4f7ff"
                    }
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const duration = context.parsed.x;
                            const power = context.parsed.y;
                            return `${context.dataset.label}: ${power.toFixed(1)} W at ${formatDuration(duration)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Duration (seconds)",
                        color: "#9cb0d3"
                    },
                    ticks: {
                        color: "#9cb0d3"
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.08)"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Power (W)",
                        color: "#9cb0d3"
                    },
                    ticks: {
                        color: "#9cb0d3"
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.08)"
                    }
                }
            }
        }
    });
}

function renderTable(modeled) {
    const tbody = document.getElementById("residualTableBody");
    tbody.innerHTML = "";

    modeled
        .slice()
        .sort((a, b) => a.duration - b.duration)
        .forEach(point => {
            const tr = document.createElement("tr");
            const diffClass = point.diff >= 0 ? "diff-positive" : "diff-negative";
            tr.innerHTML = `
                <td>${point.index}</td>
                <td>${formatDuration(point.duration)} (${point.duration}s)</td>
                <td>${point.power.toFixed(1)} W</td>
                <td>${point.expectedPower.toFixed(1)} W</td>
                <td class="${diffClass}">${formatSigned(point.diff)} W</td>
            `;
            tbody.appendChild(tr);
        });
}

function setMessage(text, type = "") {
    const el = document.getElementById("message");
    el.textContent = text;
    el.className = `message ${type}`.trim();
}

function calculate() {
    const { points, error } = collectDataPoints();
    if (error) {
        setMessage(error, "error");
        document.getElementById("results").classList.add("hidden");
        return;
    }

    const regression = runRegression(points);
    if (regression.error) {
        setMessage(regression.error, "error");
        document.getElementById("results").classList.add("hidden");
        return;
    }

    document.getElementById("cpValue").textContent = `${regression.cp.toFixed(1)} W`;
    document.getElementById("wPrimeValue").textContent = `${(regression.wPrime / 1000).toFixed(2)} kJ (${regression.wPrime.toFixed(0)} J)`;
    document.getElementById("rValue").textContent = regression.r.toFixed(4);

    renderChart(points, regression.cp, regression.wPrime);
    renderExpectedPowerChart(points, regression.cp, regression.wPrime);
    renderTable(regression.modeled);

    document.getElementById("results").classList.remove("hidden");
    setMessage(`Regression calculated from ${points.length} data point(s).`, "success");
}

function resetForm() {
    createRows();
    setMessage("");
    document.getElementById("results").classList.add("hidden");
    if (regressionChart) {
        regressionChart.destroy();
        regressionChart = null;
    }
    if (expectedPowerChart) {
        expectedPowerChart.destroy();
        expectedPowerChart = null;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    createRows();
    document.getElementById("calculateBtn").addEventListener("click", calculate);
    document.getElementById("resetBtn").addEventListener("click", resetForm);
    calculate();
});
