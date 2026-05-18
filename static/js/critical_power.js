let regressionChart = null;
let expectedPowerChart = null;
const STORAGE_KEY = "criticalPowerCalculatorInputsV1";

function loadStoredInputs() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return null;
        }
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function saveInputs() {
    const rows = [];
    for (let i = 0; i < 4; i += 1) {
        const durationEl = document.getElementById(`duration-${i}`);
        const powerEl = document.getElementById(`power-${i}`);
        rows.push({
            duration: durationEl ? durationEl.value : "",
            power: powerEl ? powerEl.value : ""
        });
    }

    const payload = {
        rows,
        whatIfDuration: (document.getElementById("whatIfDuration") || {}).value || "",
        whatIfPower: (document.getElementById("whatIfPower") || {}).value || ""
    };

    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
        // Ignore storage failures (private mode or blocked storage).
    }
}

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

function bindRecalcOnCommit(inputEl) {
    inputEl.addEventListener("input", saveInputs);
    inputEl.addEventListener("change", calculate);
    inputEl.addEventListener("keydown", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            calculate();
        }
    });
}

function createRows(storedRows = null) {
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
        const rowValues = (storedRows && storedRows[i]) || defaults[i];
        row.innerHTML = `
            <div class="index-pill">${i + 1}</div>
            <input id="duration-${i}" type="text" inputmode="numeric" placeholder="e.g. 300 or 5:00" value="${rowValues.duration || ""}">
            <input id="power-${i}" type="number" min="1" step="1" placeholder="Watts" value="${rowValues.power || ""}">
        `;
        dataRows.appendChild(row);

        const durationEl = row.querySelector(`#duration-${i}`);
        const powerEl = row.querySelector(`#power-${i}`);
        bindRecalcOnCommit(durationEl);
        bindRecalcOnCommit(powerEl);
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

function collectWhatIfPoint() {
    const rawDuration = document.getElementById("whatIfDuration").value.trim();
    const rawPower = document.getElementById("whatIfPower").value.trim();

    if (!rawDuration && !rawPower) {
        return { hasWhatIf: false };
    }

    if (!rawDuration || !rawPower) {
        return { error: "To run What-if, enter both duration and power." };
    }

    const duration = parseDurationSeconds(rawDuration);
    const power = Number(rawPower);

    if (!duration || !Number.isFinite(power) || power <= 0) {
        return { error: "What-if values are invalid. Duration must be seconds or mm:ss, and power must be > 0." };
    }

    return {
        hasWhatIf: true,
        point: {
            index: "W",
            duration,
            power,
            work: duration * power,
            isWhatIfPoint: true
        }
    };
}

function buildWhatIfPoints(basePoints, whatIfPoint) {
    const merged = basePoints.map(p => ({ ...p }));
    const existingIndex = merged.findIndex(p => p.duration === whatIfPoint.duration);

    if (existingIndex >= 0) {
        merged[existingIndex] = {
            ...merged[existingIndex],
            power: whatIfPoint.power,
            work: whatIfPoint.duration * whatIfPoint.power,
            isWhatIfPoint: true
        };
        return {
            points: merged,
            summary: `What-if point replaced duration ${formatDuration(whatIfPoint.duration)} with ${whatIfPoint.power.toFixed(1)} W.`
        };
    }

    merged.push(whatIfPoint);
    return {
        points: merged,
        summary: `What-if point added at ${formatDuration(whatIfPoint.duration)} and ${whatIfPoint.power.toFixed(1)} W.`
    };
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

function renderChart(points, cp, wPrime, whatIfRegression = null) {
    const sortedPoints = [...points].sort((a, b) => a.duration - b.duration);
    const baseMaxX = sortedPoints[sortedPoints.length - 1].duration;
    const whatIfMaxX = whatIfRegression
        ? Math.max(...whatIfRegression.points.map(p => p.duration))
        : baseMaxX;
    const maxX = Math.max(baseMaxX, whatIfMaxX);

    const lineData = [
        { x: 0, y: wPrime },
        { x: maxX, y: cp * maxX + wPrime }
    ];

    const scatterData = sortedPoints.map(p => ({ x: p.duration, y: p.work }));

    const datasets = [
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
    ];

    if (whatIfRegression) {
        const whatIfScatter = whatIfRegression.points
            .slice()
            .sort((a, b) => a.duration - b.duration)
            .map(p => ({ x: p.duration, y: p.work }));
        const whatIfLine = [
            { x: 0, y: whatIfRegression.wPrime },
            { x: maxX, y: whatIfRegression.cp * maxX + whatIfRegression.wPrime }
        ];

        datasets.push(
            {
                label: "What-if Work",
                data: whatIfScatter,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: "#101728",
                pointBorderColor: "#43d587",
                pointBorderWidth: 2,
                showLine: false
            },
            {
                label: "What-if Regression Line",
                type: "line",
                data: whatIfLine,
                borderColor: "#43d587",
                borderWidth: 2,
                borderDash: [8, 6],
                pointRadius: 0,
                tension: 0
            }
        );
    }

    if (regressionChart) {
        regressionChart.destroy();
    }

    const ctx = document.getElementById("regressionChart").getContext("2d");
    regressionChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets
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

function renderExpectedPowerChart(points, cp, wPrime, whatIfRegression = null) {
    const sortedPoints = [...points].sort((a, b) => a.duration - b.duration);
    const baseMinX = sortedPoints[0].duration;
    const baseMaxX = sortedPoints[sortedPoints.length - 1].duration;
    const whatIfMinX = whatIfRegression
        ? Math.min(...whatIfRegression.points.map(p => p.duration))
        : baseMinX;
    const whatIfMaxX = whatIfRegression
        ? Math.max(...whatIfRegression.points.map(p => p.duration))
        : baseMaxX;
    const minX = Math.min(baseMinX, whatIfMinX);
    const maxX = Math.max(baseMaxX, whatIfMaxX);

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

    const datasets = [
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
    ];

    if (whatIfRegression) {
        const whatIfActual = whatIfRegression.points
            .slice()
            .sort((a, b) => a.duration - b.duration)
            .map(p => ({ x: p.duration, y: p.power }));

        const whatIfCurve = [];
        for (let i = 0; i <= steps; i += 1) {
            const t = start + ((end - start) * (i / steps));
            whatIfCurve.push({ x: t, y: whatIfRegression.cp + (whatIfRegression.wPrime / t) });
        }

        datasets.push(
            {
                label: "What-if Power",
                data: whatIfActual,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: "#101728",
                pointBorderColor: "#f5a623",
                pointBorderWidth: 2,
                showLine: false
            },
            {
                label: "What-if Expected Curve",
                type: "line",
                data: whatIfCurve,
                borderColor: "#f5a623",
                borderWidth: 2,
                borderDash: [8, 6],
                pointRadius: 0,
                tension: 0.2
            }
        );
    }

    if (expectedPowerChart) {
        expectedPowerChart.destroy();
    }

    const ctx = document.getElementById("expectedPowerChart").getContext("2d");
    expectedPowerChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets
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

function renderTable(modeled, tbodyId = "residualTableBody") {
    const tbody = document.getElementById(tbodyId);
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
    saveInputs();
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

    const whatIfInput = collectWhatIfPoint();
    const whatIfInputError = whatIfInput.error || "";

    let whatIfRegression = null;
    let whatIfSummary = "";

    if (whatIfInput.hasWhatIf && !whatIfInputError) {
        const whatIfBuild = buildWhatIfPoints(points, whatIfInput.point);
        whatIfSummary = whatIfBuild.summary;
        const recalculated = runRegression(whatIfBuild.points);
        if (recalculated.error) {
            // Keep base regression visible; only disable what-if overlays.
            whatIfSummary = "";
            whatIfRegression = null;
        } else {
            whatIfRegression = {
                ...recalculated,
                points: whatIfBuild.points
            };
        }
    }

    renderChart(points, regression.cp, regression.wPrime, whatIfRegression);
    renderExpectedPowerChart(points, regression.cp, regression.wPrime, whatIfRegression);
    renderTable(regression.modeled);

    const whatIfMetrics = document.getElementById("whatIfMetrics");
    const whatIfSummaryEl = document.getElementById("whatIfSummary");
    const whatIfTableCard = document.getElementById("whatIfTableCard");

    if (whatIfRegression) {
        document.getElementById("whatIfCpValue").textContent = `${whatIfRegression.cp.toFixed(1)} W`;
        document.getElementById("whatIfWPrimeValue").textContent = `${(whatIfRegression.wPrime / 1000).toFixed(2)} kJ (${whatIfRegression.wPrime.toFixed(0)} J)`;
        document.getElementById("whatIfRValue").textContent = whatIfRegression.r.toFixed(4);
        whatIfSummaryEl.textContent = `${whatIfSummary} Dashed lines show the What-if model.`;
        whatIfSummaryEl.classList.remove("hidden");
        whatIfMetrics.classList.remove("hidden");
        whatIfTableCard.classList.remove("hidden");
        renderTable(whatIfRegression.modeled, "whatIfResidualTableBody");
    } else {
        whatIfSummaryEl.textContent = "";
        whatIfSummaryEl.classList.add("hidden");
        whatIfMetrics.classList.add("hidden");
        whatIfTableCard.classList.add("hidden");
    }

    document.getElementById("results").classList.remove("hidden");

    if (whatIfInputError) {
        setMessage(`Regression calculated from ${points.length} data point(s). What-if ignored: ${whatIfInputError}`, "success");
        return;
    }

    setMessage(
        whatIfRegression
            ? `Regression calculated from ${points.length} base point(s) plus What-if scenario.`
            : `Regression calculated from ${points.length} data point(s).`,
        "success"
    );
}

function resetForm() {
    createRows();
    document.getElementById("whatIfDuration").value = "";
    document.getElementById("whatIfPower").value = "";
    saveInputs();
    setMessage("");
    document.getElementById("results").classList.add("hidden");
    document.getElementById("whatIfSummary").textContent = "";
    document.getElementById("whatIfSummary").classList.add("hidden");
    document.getElementById("whatIfMetrics").classList.add("hidden");
    document.getElementById("whatIfTableCard").classList.add("hidden");
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
    const stored = loadStoredInputs();
    createRows(stored && Array.isArray(stored.rows) ? stored.rows : null);
    document.getElementById("calculateBtn").addEventListener("click", calculate);
    document.getElementById("resetBtn").addEventListener("click", resetForm);

    bindRecalcOnCommit(document.getElementById("whatIfDuration"));
    bindRecalcOnCommit(document.getElementById("whatIfPower"));

    if (stored) {
        document.getElementById("whatIfDuration").value = stored.whatIfDuration || "";
        document.getElementById("whatIfPower").value = stored.whatIfPower || "";
    }

    calculate();
});
