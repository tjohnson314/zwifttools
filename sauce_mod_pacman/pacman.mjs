import * as Common from '/pages/src/common.mjs';
import {SauceZwiftMap} from '/pages/src/map.mjs';

Common.enableSentry();

const SCOTLAND_COURSE_ID = 17;
const MAX_SPEED_MPS = 100 / 3.6;
const SAFETY_METERS = 4;
const COLLISION_MIN = 12;
const COLLISION_MAX = 25;
const PELLET_SPACING_METERS = 55;
const PELLET_COLLECT_METERS = 9;
const GHOST_COUNT = 4;
const GHOST_BASE_SPEED = 8.5;
const GLOBAL_GHOST_SPEED_MULT = 0.80;
const SIM_HZ = 10;
const GHOST_SPAWN_MIN_DIST_M = 500;
const CATCH_GRACE_PERIOD_MS = 12000;
const TELEPORT_MSG_HOLD_MS = 20000;

const DIFFICULTY_PRESETS = {
    easy: {downhillRel: 0.90, flatRel: 0.95, uphillRel: 0.98},
    normal: {downhillRel: 0.95, flatRel: 1.00, uphillRel: 1.05},
    hard: {downhillRel: 0.97, flatRel: 1.03, uphillRel: 1.10},
    pro: {downhillRel: 0.98, flatRel: 1.03, uphillRel: 1.16},
};

const PLAYER_REF_SPEEDS_KMH = [
    {grade: -12, speed: 106.66},
    {grade: -11, speed: 102.12},
    {grade: -10, speed: 97.39},
    {grade: -9, speed: 92.44},
    {grade: -8, speed: 87.24},
    {grade: -7, speed: 81.80},
    {grade: -6, speed: 76.10},
    {grade: -5, speed: 70.18},
    {grade: -4, speed: 64.11},
    {grade: -3, speed: 58.00},
    {grade: -2, speed: 52.00},
    {grade: -1, speed: 46.16},
    {grade: 0, speed: 40.42},
    {grade: 1, speed: 34.79},
    {grade: 2, speed: 29.55},
    {grade: 3, speed: 24.96},
    {grade: 4, speed: 21.17},
    {grade: 5, speed: 18.14},
    {grade: 6, speed: 15.75},
    {grade: 7, speed: 13.86},
    {grade: 8, speed: 12.35},
    {grade: 9, speed: 11.12},
    {grade: 10, speed: 10.10},
    {grade: 11, speed: 9.26},
    {grade: 12, speed: 8.54},
];

const MIN_GHOST_RELATIVE_FACTOR = 0.85;
const MAX_GHOST_RELATIVE_FACTOR = 1.20;

const game = {
    phase: 'boot', // boot|running|respawning|gameover
    lives: 3,
    score: 0,
    respawnDeadlineTs: null,
    lastPlayerSampleTs: null,
    collisionThresholdMeters: COLLISION_MIN,
    player: {
        state: null,
        graphPos: null,
        point: null,
        visualPoint: null,
        home: null,
    },
    roads: new Map(),
    nodes: [],
    nodeKeyMap: new Map(),
    adjacency: new Map(),
    pellets: [],
    ghosts: [],
    closestPelletHint: null,
    lastPelletHintTs: 0,
    difficulty: 'normal',
    catchGraceUntilTs: 0,
    respawnTeleportPending: false,
    componentCount: 0,
};

const seenPelletRoads = new Set();

let mapEl;
let statusEl;
let livesEl;
let scoreEl;
let pelletsEl;
let collisionEl;
let messageEl;
let difficultySelectEl;
let ghostDistanceEls = [];
let ghostGradeEls = [];
let ghostSpeedEls = [];
let simInterval;
let drawQueued = false;
let zwiftMap;
let mapWorldList;
let messageHoldUntilTs = 0;
const ghostEnts = new Map();
const pelletEnts = new Map();
const componentHighlightPaths = [];


export async function main() {
    Common.initInteractionListeners();

    mapEl = document.getElementById('map');
    statusEl = document.getElementById('status');
    livesEl = document.getElementById('lives');
    scoreEl = document.getElementById('score');
    pelletsEl = document.getElementById('pellets');
    collisionEl = document.getElementById('collision-threshold');
    messageEl = document.getElementById('message');
    difficultySelectEl = document.getElementById('difficulty-select');
    ghostDistanceEls = [1, 2, 3, 4].map(i => document.getElementById(`ghost-dist-${i}`));
    ghostGradeEls = [1, 2, 3, 4].map(i => document.getElementById(`ghost-grade-${i}`));
    ghostSpeedEls = [1, 2, 3, 4].map(i => document.getElementById(`ghost-speed-${i}`));

    const storedDifficulty = Common.settingsStore.get('difficulty');
    const initialDifficulty = DIFFICULTY_PRESETS[storedDifficulty] ? storedDifficulty : 'normal';
    game.difficulty = initialDifficulty;
    difficultySelectEl.value = initialDifficulty;
    difficultySelectEl.addEventListener('change', () => {
        const value = difficultySelectEl.value;
        game.difficulty = DIFFICULTY_PRESETS[value] ? value : 'normal';
        Common.settingsStore.set('difficulty', game.difficulty);
    });

    statusEl.textContent = 'Loading Scotland roads...';
    await initMap();
    await loadScotlandGraph();
    initGhosts();

    game.phase = 'running';
    statusEl.textContent = 'Waiting for rider state...';
    Common.subscribe('athlete/self', onSelfState);

    simInterval = setInterval(stepSimulation, 1000 / SIM_HZ);
    requestDraw();
}


function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}


function pointDistanceToSegmentCm(point, a, b) {
    const ax = a[0];
    const ay = a[1];
    const bx = b[0];
    const by = b[1];
    const px = point[0];
    const py = point[1];
    const vx = bx - ax;
    const vy = by - ay;
    const wx = px - ax;
    const wy = py - ay;
    const vv = vx * vx + vy * vy;
    if (vv < 1e-6) {
        return Math.hypot(px - ax, py - ay);
    }
    const t = clamp((wx * vx + wy * vy) / vv, 0, 1);
    const cx = ax + t * vx;
    const cy = ay + t * vy;
    return Math.hypot(px - cx, py - cy);
}


function onSelfState(data) {
    if (!data?.state) {
        return;
    }
    const state = data.state;
    if (state.courseId !== SCOTLAND_COURSE_ID) {
        statusEl.textContent = `Switch to Scotland (course ${SCOTLAND_COURSE_ID}) to play`;
        showMessage('This mode currently supports Scotland only.', true);
        return;
    }
    const now = Date.now();
    if (now >= messageHoldUntilTs) {
        clearMessage();
    }
    if (game.lastPlayerSampleTs) {
        const dt = clamp((now - game.lastPlayerSampleTs) / 1000, 0.10, 0.50);
        game.collisionThresholdMeters = clamp(MAX_SPEED_MPS * dt + SAFETY_METERS, COLLISION_MIN, COLLISION_MAX);
    }
    game.lastPlayerSampleTs = now;

    const prevVisualPoint = game.player.visualPoint ? [...game.player.visualPoint] : null;
    const pos = stateToGraphPos(state);
    if (!pos) {
        statusEl.textContent = 'Waiting for road projection...';
        return;
    }

    game.player.state = state;
    game.player.graphPos = pos;
    const visualPoint = Number.isFinite(state.x) && Number.isFinite(state.y) ?
        [state.x, state.y] : graphPosToPoint(pos);
    game.player.visualPoint = visualPoint;
    game.player.point = visualPoint;

    if (!zwiftMap.athleteId) {
        zwiftMap.setAthlete(state.athleteId);
        zwiftMap.setWatching(state.athleteId);
    }
    zwiftMap.renderAthleteStates([state]).catch(err => console.error('map render state failed', err));
    updateMapViewport();

    if (!game.player.home) {
        game.player.home = {...pos};
        respawnGhostsSafely();
        game.catchGraceUntilTs = Date.now() + CATCH_GRACE_PERIOD_MS;
    }

    statusEl.textContent = 'Running';
    checkPelletCollection(prevVisualPoint, visualPoint);
    requestDraw();
}


async function loadScotlandGraph() {
    const roads = await Common.getRoads(SCOTLAND_COURSE_ID);
    resetGraphState();
    clearComponentOverlays();
    for (const road of roads) {
        if (!road.isAvailable || road.oneWay) {
            continue;
        }
        const lengthM = road.curvePath.distance() / 100;
        if (!Number.isFinite(lengthM) || lengthM < 8) {
            continue;
        }

        const controlPoints = buildRoadControlPoints(road, lengthM);
        if (controlPoints.length < 2) {
            continue;
        }

        game.roads.set(road.id, {
            id: road.id,
            road,
            lengthM,
            controlPoints,
            startNodeId: controlPoints[0].nodeId,
            endNodeId: controlPoints[controlPoints.length - 1].nodeId,
        });

        addRoadInternalEdges(game.roads.get(road.id));
        placePelletsOnRoad(road.id, lengthM);
    }

    addNearbyControlPointEdges(10);

    computeConnectedComponents();

    pelletsEl.textContent = String(game.pellets.filter(p => !p.collected).length);
}


function resetGraphState() {
    game.roads.clear();
    game.nodes = [];
    game.nodeKeyMap.clear();
    game.adjacency.clear();
    game.pellets = [];
    seenPelletRoads.clear();
    game.componentCount = 0;
}


function buildRoadControlPoints(road, lengthM) {
    const pts = road.path.map(p => [Number(p[0]), Number(p[1])]);
    if (pts.length < 2) {
        return [];
    }
    let rawLenM = 0;
    const segLensM = [0];
    for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0] - pts[i - 1][0];
        const dy = pts[i][1] - pts[i - 1][1];
        rawLenM += Math.hypot(dx, dy) / 100;
        segLensM.push(rawLenM);
    }
    const scale = rawLenM > 0 ? lengthM / rawLenM : 0;
    return pts.map((point, index) => ({
        nodeId: createGraphNode(point, road.id, index, segLensM[index] * scale),
        point,
        index,
        offsetM: segLensM[index] * scale,
    }));
}


function createGraphNode(point, roadId, pointIndex, offsetM=0) {
    const id = game.nodes.length;
    game.nodes.push({id, roadId, pointIndex, offsetM, point: [point[0], point[1]]});
    return id;
}


function addRoadInternalEdges(roadInfo) {
    const controlPoints = roadInfo.controlPoints;
    for (let i = 1; i < controlPoints.length; i++) {
        const a = controlPoints[i - 1];
        const b = controlPoints[i];
        const segM = Math.max(0.001, b.offsetM - a.offsetM);
        addEdge(a.nodeId, b.nodeId, roadInfo.id, +1, segM);
        addEdge(b.nodeId, a.nodeId, roadInfo.id, -1, segM);
    }
}


function addNearbyControlPointEdges(maxDistanceM) {
    const roads = Array.from(game.roads.values());
    const bestByRoadPair = new Map();
    for (let i = 0; i < roads.length; i++) {
        const roadA = roads[i];
        for (let j = i + 1; j < roads.length; j++) {
            const roadB = roads[j];
            let best = null;
            for (const cpA of roadA.controlPoints) {
                for (const cpB of roadB.controlPoints) {
                    const dx = cpA.point[0] - cpB.point[0];
                    const dy = cpA.point[1] - cpB.point[1];
                    const distM = Math.hypot(dx, dy) / 100;
                    if (distM > maxDistanceM) {
                        continue;
                    }
                    if (!best || distM < best.distanceM) {
                        best = {
                            roadAId: roadA.id,
                            roadBId: roadB.id,
                            nodeAId: cpA.nodeId,
                            nodeBId: cpB.nodeId,
                            distanceM: distM,
                        };
                    }
                }
            }
            if (best) {
                bestByRoadPair.set(pickRoadPairKey(roadA.id, roadB.id), best);
            }
        }
    }

    for (const best of bestByRoadPair.values()) {
        addEdge(best.nodeAId, best.nodeBId, `x-${best.roadAId}-${best.roadBId}`, +1, best.distanceM);
        addEdge(best.nodeBId, best.nodeAId, `x-${best.roadAId}-${best.roadBId}`, -1, best.distanceM);
    }
}


function pickRoadPairKey(a, b) {
    return a < b ? `${a}:${b}` : `${b}:${a}`;
}


function computeConnectedComponents() {
    const nodeComponent = new Array(game.nodes.length).fill(-1);
    let compId = 0;
    for (let root = 0; root < game.nodes.length; root++) {
        if (nodeComponent[root] !== -1) {
            continue;
        }
        const queue = [root];
        nodeComponent[root] = compId;
        for (let qi = 0; qi < queue.length; qi++) {
            const u = queue[qi];
            const edges = game.adjacency.get(u) || [];
            for (const e of edges) {
                if (nodeComponent[e.to] === -1) {
                    nodeComponent[e.to] = compId;
                    queue.push(e.to);
                }
            }
        }
        compId++;
    }
    game.componentCount = compId;
    for (const roadInfo of game.roads.values()) {
        roadInfo.componentId = nodeComponent[roadInfo.startNodeId] ?? 0;
    }
}


function clearComponentOverlays() {
    for (const hl of componentHighlightPaths) {
        zwiftMap.removeHighlightPath(hl);
    }
    componentHighlightPaths.length = 0;
}


function getOrCreateNode(point) {
    throw new Error('getOrCreateNode is obsolete with control-point graph');
}


function addEdge(from, to, roadId, direction, lengthM) {
    if (!game.adjacency.has(from)) {
        game.adjacency.set(from, []);
    }
    game.adjacency.get(from).push({to, roadId, direction, lengthM});
}


function pelletRoadKey(roadInfo) {
    const a = roadInfo.startNodeId;
    const b = roadInfo.endNodeId;
    const n1 = Math.min(a, b);
    const n2 = Math.max(a, b);
    const mid = roadInfo.road.curvePath.pointAtDistance(roadInfo.road.curvePath.distance() / 2) || [0, 0];
    const mx = Math.round(mid[0] / 2);
    const my = Math.round(mid[1] / 2);
    const len = Math.round(roadInfo.lengthM);
    return `${n1}:${n2}:${len}:${mx}:${my}`;
}


function placePelletsOnRoad(roadId, lengthM) {
    const roadInfo = game.roads.get(roadId);
    const key = roadInfo ? pelletRoadKey(roadInfo) : `${roadId}`;
    if (seenPelletRoads.has(key)) {
        return;
    }
    seenPelletRoads.add(key);

    const count = Math.max(1, Math.ceil(lengthM / PELLET_SPACING_METERS));
    const step = lengthM / count;
    for (let i = 0; i < count; i++) {
        const offsetM = (i + 0.5) * step;
        const point = graphPosToPoint({roadId, offsetM});
        game.pellets.push({
            id: `${roadId}:${offsetM.toFixed(1)}`,
            roadId,
            offsetM,
            point,
            collected: false,
        });
    }
}


function initGhosts() {
    const personalities = [
        {chaseBias: 0.92, randomnessM: 18, spreadPenaltyM: 36},
        {chaseBias: 0.82, randomnessM: 34, spreadPenaltyM: 52},
        {chaseBias: 0.74, randomnessM: 48, spreadPenaltyM: 64},
        {chaseBias: 0.86, randomnessM: 26, spreadPenaltyM: 44},
    ];
    game.ghosts = new Array(GHOST_COUNT).fill(0).map((_, i) => ({
        id: `ghost-${i + 1}`,
        color: ['#ff5d61', '#5db8ff', '#ff92cb', '#ffb554'][i % 4],
        roadId: null,
        offsetM: 0,
        direction: i % 2 === 0 ? 1 : -1,
        speedMps: GHOST_BASE_SPEED + i * 0.35,
        chaseBias: personalities[i].chaseBias,
        randomnessM: personalities[i].randomnessM,
        spreadPenaltyM: personalities[i].spreadPenaltyM,
        targetNodeId: null,
        lastPathDistanceToPlayer: null,
        lastAdjustedDistanceToPlayer: null,
        lastNodeId: null,
    }));
}


function respawnGhostsNearHome() {
    if (!game.player.home) {
        return;
    }
    for (let i = 0; i < game.ghosts.length; i++) {
        const g = game.ghosts[i];
        g.roadId = game.player.home.roadId;
        g.offsetM = clamp(game.player.home.offsetM + (i - 1.5) * 30, 1, game.roads.get(g.roadId).lengthM - 1);
        g.direction = i % 2 === 0 ? 1 : -1;
    }
}


function respawnGhostsSafely() {
    if (!game.player.graphPos) {
        respawnGhostsNearHome();
        return;
    }
    for (let i = 0; i < game.ghosts.length; i++) {
        const g = game.ghosts[i];
        const spawn = findSpawnForGhost(i);
        g.roadId = spawn.roadId;
        g.offsetM = spawn.offsetM;
        g.direction = spawn.direction;
    }
}


function findSpawnForGhost(index) {
    const playerCompId = game.roads.get(game.player.graphPos?.roadId)?.componentId;
    const allRoads = Array.from(game.roads.values());
    const roads = playerCompId != null
        ? allRoads.filter(r => r.componentId === playerCompId)
        : allRoads;
    let best = null;
    const tries = Math.min(roads.length, 120);
    for (let i = 0; i < tries; i++) {
        const road = roads[(index * 31 + i * 17) % roads.length];
        const length = road.lengthM;
        const offsetM = clamp(length * (0.12 + (((index + i) % 7) / 8) * 0.76), 2, length - 2);
        const direction = (i + index) % 2 === 0 ? 1 : -1;
        const d = shortestPathInfo(game.player.graphPos, {roadId: road.id, offsetM}).distanceM;
        if (Number.isFinite(d) && d >= GHOST_SPAWN_MIN_DIST_M) {
            return {roadId: road.id, offsetM, direction};
        }
        if (!best || (Number.isFinite(d) && d > best.distanceM)) {
            best = {roadId: road.id, offsetM, direction, distanceM: d};
        }
    }
    return best || {
        roadId: game.player.home.roadId,
        offsetM: clamp(game.player.home.offsetM + 650 + index * 90, 2, game.roads.get(game.player.home.roadId).lengthM - 2),
        direction: index % 2 ? -1 : 1,
    };
}


function stateToGraphPos(state) {
    const roadInfo = game.roads.get(state.roadId);
    if (!roadInfo || state.roadTime == null) {
        return null;
    }
    const distCm = roadInfo.road.curvePath.distanceAtRoadTime(state.roadTime);
    const distanceM = clamp(Number.isFinite(distCm) ? distCm / 100 : 0, 0, roadInfo.lengthM);
    return {
        roadId: state.roadId,
        offsetM: distanceM,
    };
}


function graphPosToPoint(pos) {
    const roadInfo = game.roads.get(pos.roadId);
    if (!roadInfo) {
        return null;
    }
    const curve = roadInfo.road.curvePath;
    const targetDistCm = clamp(pos.offsetM * 100, 0, curve.distance());
    const p = curve.pointAtDistance(targetDistCm);
    return p ? [p[0], p[1]] : null;
}


function graphPosToEndpoints(pos) {
    const roadInfo = game.roads.get(pos.roadId);
    if (!roadInfo || !roadInfo.controlPoints || !roadInfo.controlPoints.length) {
        return null;
    }

    const target = clamp(pos.offsetM, 0, roadInfo.lengthM);
    const cps = roadInfo.controlPoints;
    let hi = cps.length;
    let lo = 0;
    while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (cps[mid].offsetM < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const left = cps[Math.max(0, lo - 1)];
    const right = cps[Math.min(cps.length - 1, lo)];
    const candidates = [];
    const pushCandidate = cp => {
        if (!cp) {
            return;
        }
        if (!candidates.some(x => x.nodeId === cp.nodeId)) {
            candidates.push({
                nodeId: cp.nodeId,
                attachDist: Math.abs(cp.offsetM - target),
                initialDirection: cp.offsetM >= target ? +1 : -1,
            });
        }
    };
    pushCandidate(left);
    pushCandidate(right);
    return {
        roadInfo,
        candidates,
    };
}


function shortestNodePaths(startNodeId) {
    const dist = new Array(game.nodes.length).fill(Infinity);
    const prev = new Array(game.nodes.length).fill(-1);
    const visited = new Array(game.nodes.length).fill(false);
    dist[startNodeId] = 0;

    for (;;) {
        let u = -1;
        let best = Infinity;
        for (let i = 0; i < dist.length; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u === -1) {
            break;
        }
        visited[u] = true;
        const edges = game.adjacency.get(u) || [];
        for (const e of edges) {
            const alt = dist[u] + e.lengthM;
            if (alt < dist[e.to]) {
                dist[e.to] = alt;
                prev[e.to] = u;
            }
        }
    }

    return {dist, prev};
}


function shortestPathInfo(fromPos, toPos) {
    const a = graphPosToEndpoints(fromPos);
    const b = graphPosToEndpoints(toPos);
    if (!a || !b) {
        return {distanceM: Infinity, initialDirection: null};
    }

    let best = {
        distanceM: Infinity,
        initialDirection: null,
    };

    for (const fc of a.candidates) {
        const sp = shortestNodePaths(fc.nodeId);
        for (const tc of b.candidates) {
            const middle = sp.dist[tc.nodeId];
            if (!Number.isFinite(middle)) {
                continue;
            }
            const total = fc.attachDist + middle + tc.attachDist;
            if (total < best.distanceM) {
                best = {
                    distanceM: total,
                    initialDirection: fc.initialDirection,
                };
            }
        }
    }

    if (fromPos.roadId === toPos.roadId) {
        const d = Math.abs(fromPos.offsetM - toPos.offsetM);
        if (d < best.distanceM) {
            best = {
                distanceM: d,
                initialDirection: toPos.offsetM >= fromPos.offsetM ? +1 : -1,
            };
        }
    }

    return best;
}


function tangentAt(pos, directionSign) {
    const roadInfo = game.roads.get(pos.roadId);
    if (!roadInfo) {
        return [1, 0];
    }
    const curve = roadInfo.road.curvePath;
    const curveDistCm = curve.distance();
    const baseCm = clamp(pos.offsetM * 100, 0, curveDistCm);
    const deltaCm = clamp(500, 50, Math.max(100, curveDistCm * 0.03));
    const p1 = curve.pointAtDistance(baseCm);
    const p2 = curve.pointAtDistance(clamp(baseCm + deltaCm * directionSign, 0, curveDistCm));
    if (!p1 || !p2) {
        return [1, 0];
    }
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    const mag = Math.hypot(dx, dy) || 1;
    dx /= mag;
    dy /= mag;
    return [dx, dy];
}


function stepSimulation() {
    if (!game.player.graphPos || !game.player.point) {
        updateHud();
        requestDraw();
        return;
    }

    const now = Date.now();

    if (game.phase === 'respawning') {
        const sec = Math.max(0, Math.ceil((game.respawnDeadlineTs - now) / 1000));
        if (!game.respawnTeleportPending) {
            showMessage(`Caught by ghost. Teleporting home in ${sec}s...`, true);
        }
        if (now >= game.respawnDeadlineTs && !game.respawnTeleportPending) {
            game.respawnTeleportPending = true;
            showMessage('Teleporting home...', true);
            attemptTeleportHome().then(result => {
                const usedGraceMs = result.ok ? CATCH_GRACE_PERIOD_MS : CATCH_GRACE_PERIOD_MS * 2;
                finishRespawn(Date.now(), usedGraceMs);
                if (result.ok) {
                    showMessage('Teleported home', false);
                } else {
                    holdMessage('Teleport unsupported in this Sauce build. Safe respawn applied (extra grace).',
                        7000);
                }
            }).catch(err => {
                const errMsg = err?.message || String(err);
                holdMessage(`Teleport failed: ${errMsg}`, TELEPORT_MSG_HOLD_MS);
                console.error('teleportHome failed', err);
                finishRespawn(Date.now());
            });
        }
        updateHud();
        requestDraw();
        return;
    }

    if (game.phase === 'gameover') {
        showMessage('Game Over', true);
        updateHud();
        requestDraw();
        return;
    }

    for (const ghost of game.ghosts) {
        stepGhost(ghost, 1 / SIM_HZ);
        const gPos = {roadId: ghost.roadId, offsetM: ghost.offsetM};
        const p = shortestPathInfo(game.player.graphPos, gPos);
        ghost.lastPathDistanceToPlayer = p.distanceM;
        let adjustedDistanceM;
        if (Number.isFinite(p.distanceM)) {
            adjustedDistanceM = p.distanceM - game.collisionThresholdMeters;
        } else {
            const gp = graphPosToPoint(gPos);
            if (gp && game.player.point) {
                const euclidM = Math.hypot(gp[0] - game.player.point[0], gp[1] - game.player.point[1]) / 100;
                adjustedDistanceM = euclidM - game.collisionThresholdMeters;
            } else {
                adjustedDistanceM = Infinity;
            }
        }
        ghost.lastAdjustedDistanceToPlayer = adjustedDistanceM;
        if (now >= game.catchGraceUntilTs && Number.isFinite(p.distanceM) && adjustedDistanceM <= 0) {
            game.lives -= 1;
            game.phase = 'respawning';
            game.respawnDeadlineTs = now + 10000;
            game.respawnTeleportPending = false;
            showMessage('Caught by ghost. Teleporting home in 10s...', true);
            break;
        }
    }

    if (now - game.lastPelletHintTs > 1000) {
        game.lastPelletHintTs = now;
        game.closestPelletHint = computeClosestPelletHint();
    }

    updateHud();
    requestDraw();
}


function finishRespawn(nowTs, graceMs=CATCH_GRACE_PERIOD_MS) {
    game.respawnTeleportPending = false;
    game.respawnDeadlineTs = null;
    if (game.lives <= 0) {
        game.phase = 'gameover';
        showMessage('Game Over', true);
        return;
    }
    game.phase = 'running';
    respawnGhostsSafely();
    game.catchGraceUntilTs = nowTs + graceMs;
}


async function attemptTeleportHome() {
    const errors = [];
    try {
        await Common.rpc.teleportHome();
        return {ok: true, via: 'rpc.teleportHome'};
    } catch (err) {
        errors.push(`teleportHome: ${err?.message || String(err)}`);
    }

    try {
        await Common.rpc.sendCommands({type: 'TELEPORT_TO_START'});
        return {ok: true, via: 'rpc.sendCommands'};
    } catch (err) {
        errors.push(`sendCommands: ${err?.message || String(err)}`);
    }

    const allMissing = errors.every(x => /Invalid handler name/i.test(x));
    if (allMissing) {
        return {ok: false, unavailable: true, reason: errors.join(' | ')};
    }
    throw new Error(errors.join(' | '));
}


function stepGhost(ghost, dt) {
    if (!ghost.roadId) {
        return;
    }
    const road = game.roads.get(ghost.roadId);
    if (!road) {
        return;
    }

    const effectiveSpeed = ghostEffectiveSpeedMps(ghost);
    let nextOffset = ghost.offsetM + ghost.direction * effectiveSpeed * dt;
    if (nextOffset > road.lengthM || nextOffset < 0) {
        const hitEnd = nextOffset > road.lengthM;
        const nodeId = hitEnd ? road.endNodeId : road.startNodeId;
        const edges = game.adjacency.get(nodeId) || [];

        if (!edges.length) {
            ghost.direction *= -1;
            ghost.offsetM = clamp(ghost.offsetM, 0, road.lengthM);
            return;
        }

        const choice = chooseGhostEdge(ghost, nodeId, edges);
        const destNode = game.nodes[choice.to];
        if (!destNode) {
            ghost.direction *= -1;
            return;
        }
        const destRoad = game.roads.get(destNode.roadId);
        if (!destRoad) {
            ghost.direction *= -1;
            return;
        }
        ghost.lastNodeId = nodeId;
        ghost.roadId = destNode.roadId;
        ghost.offsetM = destNode.offsetM;
        ghost.direction = choice.direction;
        return;
    }

    ghost.offsetM = clamp(nextOffset, 0, road.lengthM);
}


function ghostEffectiveSpeedMps(ghost) {
    const preset = DIFFICULTY_PRESETS[game.difficulty] || DIFFICULTY_PRESETS.normal;
    const grade = roadGradeAtOffset(ghost.roadId, ghost.offsetM);
    // grade from road data is forward-direction slope; invert when ghost is moving backward
    const movementGrade = grade * ghost.direction;
    const clampedGrade = clamp(movementGrade, -0.12, 0.12);
    const refPlayerKmh = referencePlayerSpeedKmh(clampedGrade);

    // Non-linear interpolation keeps Pro from exploding on flats/descents while
    // still preserving stronger uphill pressure.
    let relativeFactor;
    if (clampedGrade >= 0) {
        const t = Math.pow(clampedGrade / 0.12, 1.25);
        relativeFactor = preset.flatRel + (preset.uphillRel - preset.flatRel) * t;
    } else {
        const t = Math.pow((-clampedGrade) / 0.12, 1.10);
        relativeFactor = preset.flatRel + (preset.downhillRel - preset.flatRel) * t;
    }
    relativeFactor = clamp(relativeFactor, MIN_GHOST_RELATIVE_FACTOR, MAX_GHOST_RELATIVE_FACTOR);

    const personality = ghost.speedMps / GHOST_BASE_SPEED;
    const speedKmh = refPlayerKmh * relativeFactor * personality * GLOBAL_GHOST_SPEED_MULT;
    return speedKmh / 3.6;
}


function referencePlayerSpeedKmh(gradeDecimal) {
    const pct = clamp(gradeDecimal * 100, -12, 12);
    const floorG = Math.floor(pct);
    const ceilG = Math.ceil(pct);
    if (floorG === ceilG) {
        return PLAYER_REF_SPEEDS_KMH[floorG + 12].speed;
    }
    const a = PLAYER_REF_SPEEDS_KMH[floorG + 12].speed;
    const b = PLAYER_REF_SPEEDS_KMH[ceilG + 12].speed;
    const t = pct - floorG;
    return a + (b - a) * t;
}


function roadGradeAtOffset(roadId, offsetM) {
    const roadInfo = game.roads.get(roadId);
    if (!roadInfo) {
        return 0;
    }
    const distances = roadInfo.road.distances;
    const grades = roadInfo.road.grades;
    if (!distances || !grades || !distances.length || !grades.length) {
        return 0;
    }
    const target = clamp(offsetM, 0, roadInfo.lengthM);
    for (let i = 1; i < distances.length; i++) {
        if (target <= distances[i]) {
            return Number.isFinite(grades[i]) ? grades[i] : 0;
        }
    }
    const last = grades[grades.length - 1];
    return Number.isFinite(last) ? last : 0;
}


function chooseGhostEdge(ghost, nodeId, edges) {
    let options = edges.slice();
    if (!options.length) {
        return null;
    }
    if (ghost.lastNodeId != null && options.length > 1) {
        const nonBacktrack = options.filter(e => e.to !== ghost.lastNodeId);
        if (nonBacktrack.length) {
            options = nonBacktrack;
        }
    }

    if (!game.player.graphPos) {
        return options[Math.floor(Math.random() * options.length)];
    }

    const candidates = [];
    for (const edge of options) {
        const dest = game.nodes[edge.to];
        if (!dest) {
            continue;
        }
        const candidate = {
            roadId: dest.roadId,
            offsetM: dest.offsetM,
        };
        const d = shortestPathInfo(candidate, game.player.graphPos).distanceM;
        const congestion = game.ghosts.filter(g => g !== ghost && g.roadId === dest.roadId).length;
        const randomJitter = Math.random() * ghost.randomnessM;
        const score = d + congestion * ghost.spreadPenaltyM + randomJitter;
        candidates.push({edge, score});
    }

    if (!candidates.length) {
        return options[0];
    }

    candidates.sort((a, b) => a.score - b.score);
    if (Math.random() < ghost.chaseBias) {
        return candidates[0].edge;
    }

    const top = candidates.slice(0, Math.min(3, candidates.length));
    const invWeights = top.map(x => 1 / Math.max(1, x.score));
    const sum = invWeights.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < top.length; i++) {
        r -= invWeights[i];
        if (r <= 0) {
            return top[i].edge;
        }
    }
    return top[top.length - 1].edge;
}


function checkPelletCollection(prevPoint, curPoint) {
    if (!curPoint) {
        return;
    }
    prevPoint = prevPoint || curPoint;
    const collectRadiusCm = PELLET_COLLECT_METERS * 100;
    let changed = false;
    for (const pellet of game.pellets) {
        if (pellet.collected) {
            continue;
        }
        const pelletPoint = pellet.point || graphPosToPoint({roadId: pellet.roadId, offsetM: pellet.offsetM});
        pellet.point = pelletPoint;
        if (!pelletPoint) {
            continue;
        }
        const dCm = pointDistanceToSegmentCm(pelletPoint, prevPoint, curPoint);
        if (dCm <= collectRadiusCm) {
            pellet.collected = true;
            game.score += 10;
            changed = true;
        }
    }
    if (changed) {
        pelletsEl.textContent = String(game.pellets.filter(p => !p.collected).length);
    }
}


function computeClosestPelletHint() {
    if (!game.player.graphPos) {
        return null;
    }
    let best = null;
    for (const pellet of game.pellets) {
        if (pellet.collected) {
            continue;
        }
        const target = {roadId: pellet.roadId, offsetM: pellet.offsetM};
        const p = shortestPathInfo(game.player.graphPos, target);
        if (!best || p.distanceM < best.distanceM) {
            best = {
                distanceM: p.distanceM,
                direction: p.initialDirection,
            };
        }
    }
    return best;
}


function updateHud() {
    livesEl.textContent = String(Math.max(0, game.lives));
    scoreEl.textContent = String(game.score);
    collisionEl.textContent = `${Math.round(game.collisionThresholdMeters)}m`;
    for (let i = 0; i < game.ghosts.length; i++) {
        const el = ghostDistanceEls[i];
        if (!el) {
            continue;
        }
        const ghost = game.ghosts[i];
        const d = ghost.lastAdjustedDistanceToPlayer;
        el.textContent = Number.isFinite(d) ? `${Math.max(0, Math.round(d))}m` : '-';
        if (ghostGradeEls[i]) {
            const grade = ghost.roadId ? roadGradeAtOffset(ghost.roadId, ghost.offsetM) : null;
            ghostGradeEls[i].textContent = grade != null ? `${(grade * 100).toFixed(1)}%` : '-';
        }
        if (ghostSpeedEls[i]) {
            const speedKmh = ghost.roadId ? ghostEffectiveSpeedMps(ghost) * 3.6 : null;
            ghostSpeedEls[i].textContent = speedKmh != null ? `${speedKmh.toFixed(1)}` : '-';
        }
    }
    if (game.phase === 'running' && Date.now() < game.catchGraceUntilTs) {
        const sec = Math.ceil((game.catchGraceUntilTs - Date.now()) / 1000);
        statusEl.textContent = `Running (grace ${sec}s)`;
    }
}


function showMessage(text, sticky=false) {
    messageEl.textContent = text;
    messageEl.classList.remove('hidden');
    if (!sticky) {
        setTimeout(clearMessage, 1600);
    }
}


function holdMessage(text, holdMs) {
    messageHoldUntilTs = Date.now() + holdMs;
    showMessage(text, true);
}


function clearMessage() {
    messageEl.classList.add('hidden');
}


function requestDraw() {
    if (drawQueued) {
        return;
    }
    drawQueued = true;
    requestAnimationFrame(() => {
        drawQueued = false;
        render();
    });
}


async function initMap() {
    mapWorldList = await Common.getWorldList({all: true});
    zwiftMap = new SauceZwiftMap({
        el: mapEl,
        worldList: mapWorldList,
        autoCenter: true,
        autoHeading: false,
        style: 'default',
        fpsLimit: 30,
        quality: 0.7,
        zoomPriorityTilt: true,
        zoom: 1,
    });
    await zwiftMap.setCourse(SCOTLAND_COURSE_ID);
    zwiftMap.setTiltShift(0);
}


function updateMapViewport() {
    // autoCenter:true means renderAthleteStates handles centering automatically.
}


function render() {
    if (!game.player.point || !zwiftMap) {
        return;
    }
    updateMapViewport();
    renderGhostEntities();
    renderPelletEntities();
}


function renderGhostEntities() {
    for (const ghost of game.ghosts) {
        if (!ghost.roadId) {
            continue;
        }
        const gPos = {roadId: ghost.roadId, offsetM: ghost.offsetM};
        const p = graphPosToPoint(gPos);
        if (!p) {
            continue;
        }
        let ent = ghostEnts.get(ghost.id);
        if (!ent) {
            ent = zwiftMap.addPoint(p, 'pacman-ghost');
            ghostEnts.set(ghost.id, ent);
        }
        ent.setPosition(p);
        ent.el.style.setProperty('--fill', ghost.color);
        ent.el.style.setProperty('--border-color', 'rgba(0,0,0,0.7)');
    }
}


function renderPelletEntities() {
    const visible = new Set();
    for (const pellet of game.pellets) {
        if (pellet.collected) {
            const oldEnt = pelletEnts.get(pellet.id);
            if (oldEnt) {
                zwiftMap.removePoint(oldEnt);
                pelletEnts.delete(pellet.id);
            }
            continue;
        }
        const p = graphPosToPoint({roadId: pellet.roadId, offsetM: pellet.offsetM});
        if (!p) {
            continue;
        }
        visible.add(pellet.id);
        let ent = pelletEnts.get(pellet.id);
        if (!ent) {
            ent = zwiftMap.addPoint(p, 'pacman-pellet');
            pelletEnts.set(pellet.id, ent);
            ent.el.style.setProperty('--fill', '#f5d53b');
            ent.el.style.setProperty('--border-color', 'rgba(0,0,0,0.5)');
        }
        ent.setPosition(p);
    }
    for (const [pelletId, ent] of pelletEnts) {
        if (!visible.has(pelletId)) {
            zwiftMap.removePoint(ent);
            pelletEnts.delete(pelletId);
        }
    }
}
