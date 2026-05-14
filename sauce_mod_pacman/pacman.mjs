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
const TICK_PROFILE_LOG_LINES = 12;
const GHOST_REGRESSION_PENALTY_SCALE = 2.2;
const GHOST_REVISIT_PENALTY_M = 180;
const GHOST_RECENT_NODE_MEMORY = 6;
const GHOST_NODE_EPS_M = 0.25;
const GHOST_MIN_HOP_CONSUME_M = 0.35;
const GHOST_SPAWN_MIN_DIST_M = 500;
const CATCH_GRACE_PERIOD_MS = 12000;
const HUNTER_DEBUG_MAX_LINES = 18;

const DIFFICULTY_PRESETS = {
    easy: {downhillRel: 0.90, flatRel: 0.95, uphillRel: 0.98},
    normal: {downhillRel: 0.95, flatRel: 1.00, uphillRel: 1.05},
    hard: {downhillRel: 0.97, flatRel: 1.03, uphillRel: 1.10},
    pro: {downhillRel: 0.98, flatRel: 1.03, uphillRel: 1.16},
    max: {downhillRel: 1.00, flatRel: 1.00, uphillRel: 1.00},
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
        travelDir: 1,
        point: null,
        visualPoint: null,
        lastSimPoint: null,
        home: null,
    },
    roads: new Map(),
    nodes: [],
    nodeKeyMap: new Map(),
    adjacency: new Map(),
    reverseAdjacency: new Map(),
    pathNodes: [],
    pathAdjacency: new Map(),
    playerPathCache: null,
    playerNodeField: null,
    pellets: [],
    ghosts: [],
    closestPelletHint: null,
    lastPelletHintTs: 0,
    lastSimStepTs: null,
    lastSimDtMs: null,
    graphEdgeCount: 0,
    pathEdgeCount: 0,
    tickCounter: 0,
    currentTickProfile: null,
    tickProfileLog: [],
    difficulty: 'normal',
    catchGraceUntilTs: 0,
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
let ghostChipEls = [];
let ghostNameEls = [];
let hunterDebugLogEl;
let hunterDebugLines = [];
let simInterval;
let drawQueued = false;


function isHunterGhost(ghost) {
    return ghost?.behavior === 'pursue' || ghost?.name === 'Hunter' || ghost?.id === 'ghost-4';
}


function fmtDecisionNum(value) {
    return Number.isFinite(value) ? Number(value).toFixed(2) : 'inf';
}


function renderHunterDebugLog() {
    if (!hunterDebugLogEl) {
        return;
    }
    hunterDebugLogEl.textContent = hunterDebugLines.length ?
        hunterDebugLines.join('\n') :
        'Waiting for Hunter decisions...';
}


function pushHunterDecisionLog(event, payload={}) {
    const ts = new Date().toISOString().slice(11, 23);
    const line = `${ts} ${event} ${JSON.stringify(payload)}`;
    hunterDebugLines.push(line);
    if (hunterDebugLines.length > HUNTER_DEBUG_MAX_LINES) {
        hunterDebugLines.splice(0, hunterDebugLines.length - HUNTER_DEBUG_MAX_LINES);
    }
    renderHunterDebugLog();
    console.log(`[HunterDecision] ${event}`, payload);
}
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
    ghostChipEls = [1, 2, 3, 4].map(i => document.getElementById(`ghost-chip-${i}`));
    ghostNameEls = [1, 2, 3, 4].map(i => document.getElementById(`ghost-name-${i}`));
    hunterDebugLogEl = document.getElementById('hunter-debug-log');
    const hunterDebugCopyBtn = document.getElementById('hunter-debug-copy-btn');
    if (hunterDebugCopyBtn) {
        hunterDebugCopyBtn.addEventListener('click', () => {
            const logText = hunterDebugLogEl.textContent;
            navigator.clipboard.writeText(logText).then(() => {
                const originalText = hunterDebugCopyBtn.textContent;
                hunterDebugCopyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    hunterDebugCopyBtn.textContent = originalText;
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy logs:', err);
            });
        });
    }
    renderHunterDebugLog();

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


function minMovingPointDistanceM(a0, a1, b0, b1) {
    if (!a0 || !a1 || !b0 || !b1) {
        return Infinity;
    }
    const r0x = a0[0] - b0[0];
    const r0y = a0[1] - b0[1];
    const vx = (a1[0] - a0[0]) - (b1[0] - b0[0]);
    const vy = (a1[1] - a0[1]) - (b1[1] - b0[1]);
    const vv = vx * vx + vy * vy;
    let t = 0;
    if (vv > 1e-9) {
        t = clamp(-(r0x * vx + r0y * vy) / vv, 0, 1);
    }
    const rx = r0x + vx * t;
    const ry = r0y + vy * t;
    return Math.hypot(rx, ry) / 100;
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

    const prevPlayerPos = game.player.graphPos;
    game.player.state = state;
    game.player.graphPos = pos;
    if (prevPlayerPos && prevPlayerPos.roadId === pos.roadId) {
        const delta = pos.offsetM - prevPlayerPos.offsetM;
        if (Math.abs(delta) > 0.5) {
            game.player.travelDir = delta > 0 ? +1 : -1;
        }
    }
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
    mergeZeroLengthNodes();
    buildReverseAdjacency();
    buildCompressedPathGraph();

    computeConnectedComponents();
    game.graphEdgeCount = Array.from(game.adjacency.values()).reduce((sum, edges) => sum + edges.length, 0);

    pelletsEl.textContent = String(game.pellets.filter(p => !p.collected).length);
}


function resetGraphState() {
    game.roads.clear();
    game.nodes = [];
    game.nodeKeyMap.clear();
    game.adjacency.clear();
    game.reverseAdjacency.clear();
    game.pathNodes = [];
    game.pathAdjacency.clear();
    game.playerPathCache = null;
    game.playerNodeField = null;
    game.pellets = [];
    seenPelletRoads.clear();
    game.componentCount = 0;
    game.graphEdgeCount = 0;
    game.pathEdgeCount = 0;
    game.tickCounter = 0;
    game.currentTickProfile = null;
    game.tickProfileLog = [];
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
    const extraEndpointPairs = [];
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
                    const candidate = {
                        roadAId: roadA.id,
                        roadBId: roadB.id,
                        nodeAId: cpA.nodeId,
                        nodeBId: cpB.nodeId,
                        distanceM: distM,
                    };
                    if (!best || distM < best.distanceM) {
                        best = candidate;
                    }
                    const cpAIsEndpoint = cpA.index === 0 || cpA.index === roadA.controlPoints.length - 1;
                    const cpBIsEndpoint = cpB.index === 0 || cpB.index === roadB.controlPoints.length - 1;
                    if (cpAIsEndpoint && cpBIsEndpoint) {
                        extraEndpointPairs.push(candidate);
                    }
                }
            }
            if (best) {
                bestByRoadPair.set(pickRoadPairKey(roadA.id, roadB.id), best);
            }
        }
    }

    const emitted = new Set();
    const emitEdgePair = edge => {
        const key = `${edge.nodeAId}:${edge.nodeBId}`;
        const revKey = `${edge.nodeBId}:${edge.nodeAId}`;
        if (emitted.has(key) || emitted.has(revKey)) {
            return;
        }
        emitted.add(key);
        emitted.add(revKey);
        addEdge(edge.nodeAId, edge.nodeBId, `x-${edge.roadAId}-${edge.roadBId}`, +1, edge.distanceM);
        addEdge(edge.nodeBId, edge.nodeAId, `x-${edge.roadAId}-${edge.roadBId}`, -1, edge.distanceM);
    };

    for (const best of bestByRoadPair.values()) {
        emitEdgePair(best);
    }
    for (const edge of extraEndpointPairs) {
        emitEdgePair(edge);
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


function addPathEdge(from, to, lengthM) {
    if (!game.pathAdjacency.has(from)) {
        game.pathAdjacency.set(from, []);
    }
    game.pathAdjacency.get(from).push({to, lengthM});
}


function mergeZeroLengthNodes() {
    // Union-Find to track node merging
    const parent = new Array(game.nodes.length);
    for (let i = 0; i < parent.length; i++) {
        parent[i] = i;
    }
    
    function find(x) {
        if (parent[x] !== x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    function union(x, y) {
        const px = find(x);
        const py = find(y);
        if (px !== py) {
            parent[py] = px;
        }
    }
    
    // Find all zero-length edges and union their endpoints
    for (const [fromNodeId, edges] of game.adjacency) {
        for (const edge of edges) {
            if (edge.lengthM <= 1e-6) {
                union(fromNodeId, edge.to);
            }
        }
    }
    
    // Build mapping from old node id to canonical node id
    const nodeMap = new Array(game.nodes.length);
    for (let i = 0; i < nodeMap.length; i++) {
        nodeMap[i] = find(i);
    }
    
    // Rebuild adjacency with remapped node ids, removing self-loops and duplicate edges
    const newAdjacency = new Map();
    for (const [fromNodeId, edges] of game.adjacency) {
        const canonicalFrom = nodeMap[fromNodeId];
        if (!newAdjacency.has(canonicalFrom)) {
            newAdjacency.set(canonicalFrom, []);
        }
        for (const edge of edges) {
            const canonicalTo = nodeMap[edge.to];
            if (canonicalFrom === canonicalTo) {
                continue; // Skip self-loops
            }
            // Check if this edge already exists
            const existing = newAdjacency.get(canonicalFrom).find(e => e.to === canonicalTo && e.roadId === edge.roadId);
            if (!existing) {
                newAdjacency.get(canonicalFrom).push({...edge, to: canonicalTo});
            }
        }
    }
    
    // Rebuild nodes array with merged nodes
    const nodeToIndex = new Map();
    const newNodes = [];
    for (let i = 0; i < game.nodes.length; i++) {
        const canonical = nodeMap[i];
        if (!nodeToIndex.has(canonical)) {
            nodeToIndex.set(canonical, newNodes.length);
            newNodes.push(game.nodes[canonical]);
        }
    }
    
    // Update node ids to match new indices and rebuild maps
    const oldToNewNodeId = new Array(game.nodes.length);
    for (let oldId = 0; oldId < game.nodes.length; oldId++) {
        const canonical = nodeMap[oldId];
        oldToNewNodeId[oldId] = nodeToIndex.get(canonical);
    }
    
    // Update adjacency with new node ids
    const finalAdjacency = new Map();
    for (const [oldFromId, edges] of newAdjacency) {
        const newFromId = nodeToIndex.get(oldFromId);
        finalAdjacency.set(newFromId, edges.map(e => ({
            ...e,
            to: nodeToIndex.get(nodeMap[e.to])
        })));
    }
    
    // Update game state
    game.nodes = newNodes;
    game.adjacency = finalAdjacency;
    game.nodeKeyMap.clear();
    for (let i = 0; i < game.nodes.length; i++) {
        const node = game.nodes[i];
        game.nodeKeyMap.set(`${node.roadId}:${node.offsetM}`, i);
    }
    
    // Update road control points and start/end node ids
    for (const roadInfo of game.roads.values()) {
        for (const cp of roadInfo.controlPoints) {
            cp.nodeId = nodeToIndex.get(nodeMap[cp.nodeId]);
        }
        roadInfo.startNodeId = nodeToIndex.get(nodeMap[roadInfo.startNodeId]);
        roadInfo.endNodeId = nodeToIndex.get(nodeMap[roadInfo.endNodeId]);
    }
}


function buildReverseAdjacency() {
    game.reverseAdjacency = new Map();
    for (const [from, edges] of game.adjacency) {
        for (const e of edges) {
            if (!game.reverseAdjacency.has(e.to)) {
                game.reverseAdjacency.set(e.to, []);
            }
            game.reverseAdjacency.get(e.to).push({
                from,
                lengthM: e.lengthM,
            });
        }
    }
}


function buildCompressedPathGraph() {
    game.pathNodes = [];
    game.pathAdjacency = new Map();
    const pathNodeByGraphNode = new Map();
    const junctionGraphNodeIds = new Set();

    for (const [from, edges] of game.adjacency) {
        for (const e of edges) {
            if (typeof e.roadId === 'string' && e.roadId.startsWith('x-')) {
                junctionGraphNodeIds.add(from);
                junctionGraphNodeIds.add(e.to);
            }
        }
    }

    const getOrCreatePathNode = cp => {
        const existing = pathNodeByGraphNode.get(cp.nodeId);
        if (existing != null) {
            return existing;
        }
        const id = game.pathNodes.length;
        game.pathNodes.push({
            id,
            graphNodeId: cp.nodeId,
            roadId: cp.roadId,
            offsetM: cp.offsetM,
        });
        pathNodeByGraphNode.set(cp.nodeId, id);
        return id;
    };

    for (const roadInfo of game.roads.values()) {
        const cps = roadInfo.controlPoints;
        if (!cps || cps.length < 2) {
            roadInfo.pathAnchors = [];
            continue;
        }
        const anchors = [];
        for (let i = 0; i < cps.length; i++) {
            const cp = cps[i];
            if (i === 0 || i === cps.length - 1 || junctionGraphNodeIds.has(cp.nodeId)) {
                anchors.push(cp);
            }
        }

        if (anchors.length < 2) {
            anchors.push(cps[0], cps[cps.length - 1]);
        }

        const seen = new Set();
        const uniqueAnchors = [];
        for (const cp of anchors) {
            if (seen.has(cp.nodeId)) {
                continue;
            }
            seen.add(cp.nodeId);
            uniqueAnchors.push(cp);
        }

        roadInfo.pathAnchors = uniqueAnchors.map(cp => ({
            nodeId: cp.nodeId,
            pathNodeId: getOrCreatePathNode({
                ...cp,
                roadId: roadInfo.id,
            }),
            offsetM: cp.offsetM,
        }));

        for (let i = 1; i < roadInfo.pathAnchors.length; i++) {
            const a = roadInfo.pathAnchors[i - 1];
            const b = roadInfo.pathAnchors[i];
            const segM = Math.max(0.001, b.offsetM - a.offsetM);
            addPathEdge(a.pathNodeId, b.pathNodeId, segM);
            addPathEdge(b.pathNodeId, a.pathNodeId, segM);
        }
    }

    const seenCross = new Set();
    for (const [from, edges] of game.adjacency) {
        for (const e of edges) {
            if (!(typeof e.roadId === 'string' && e.roadId.startsWith('x-'))) {
                continue;
            }
            const fromPath = pathNodeByGraphNode.get(from);
            const toPath = pathNodeByGraphNode.get(e.to);
            if (fromPath == null || toPath == null) {
                continue;
            }
            const key = `${fromPath}>${toPath}`;
            if (seenCross.has(key)) {
                continue;
            }
            seenCross.add(key);
            addPathEdge(fromPath, toPath, e.lengthM);
        }
    }

    game.pathEdgeCount = Array.from(game.pathAdjacency.values()).reduce((sum, edges) => sum + edges.length, 0);
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
        {name: 'Drifter', behavior: 'wander', chaseBias: 0.92, randomnessM: 18, spreadPenaltyM: 36},
        {name: 'Looper', behavior: 'wander', chaseBias: 0.82, randomnessM: 34, spreadPenaltyM: 52},
        {name: 'Ambush', behavior: 'intercept', chaseBias: 1.00, randomnessM: 8, spreadPenaltyM: 24},
        {name: 'Hunter', behavior: 'pursue', chaseBias: 1.00, randomnessM: 0, spreadPenaltyM: 0},
    ];
    game.ghosts = new Array(GHOST_COUNT).fill(0).map((_, i) => ({
        id: `ghost-${i + 1}`,
        name: personalities[i].name,
        behavior: personalities[i].behavior,
        color: ['#ff5d61', '#5db8ff', '#ff92cb', '#ffb554'][i % 4],
        roadId: null,
        offsetM: 0,
        direction: i % 2 === 0 ? 1 : -1,
        speedMps: GHOST_BASE_SPEED + i * 0.35,
        chaseBias: personalities[i].chaseBias,
        randomnessM: personalities[i].randomnessM,
        spreadPenaltyM: personalities[i].spreadPenaltyM,
        targetNodeId: null,
        graphNodeId: null,
        edgeFromNodeId: null,
        edgeToNodeId: null,
        edgeLengthM: 0,
        edgeProgressM: 0,
        edgeRoadId: null,
        renderPoint: null,
        lastPathDistanceToPlayer: null,
        lastAdjustedDistanceToPlayer: null,
        lastNodeId: null,
        lastDirectionChangeTs: 0,
        recentNodeIds: [],
        animFromPos: null,
        animToPos: null,
        animStartTs: 0,
        animDurationMs: 0,
    }));
}


function snapGhostAnimation(ghost) {
    if (!ghost.roadId) {
        ghost.animFromPos = null;
        ghost.animToPos = null;
        ghost.animStartTs = 0;
        ghost.animDurationMs = 0;
        return;
    }
    const pos = {roadId: ghost.roadId, offsetM: ghost.offsetM};
    ghost.animFromPos = pos;
    ghost.animToPos = {...pos};
    ghost.animStartTs = Date.now();
    ghost.animDurationMs = 1;
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
        g.lastNodeId = null;
        g.lastDirectionChangeTs = Date.now();
        g.recentNodeIds = [];
        initGhostGraphState(g);
        snapGhostAnimation(g);
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
        g.lastNodeId = null;
        g.lastDirectionChangeTs = Date.now();
        g.recentNodeIds = [];
        initGhostGraphState(g);
        snapGhostAnimation(g);
    }
}


function initGhostGraphState(ghost) {
    ghost.graphNodeId = null;
    ghost.edgeFromNodeId = null;
    ghost.edgeToNodeId = null;
    ghost.edgeLengthM = 0;
    ghost.edgeProgressM = 0;
    ghost.edgeRoadId = null;
    ghost.renderPoint = null;
    if (!ghost?.roadId || (ghost.behavior !== 'pursue' && ghost.behavior !== 'intercept')) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('init-skip-not-pursue', {
                hasRoadId: !!ghost?.roadId,
                behavior: ghost?.behavior,
            });
        }
        return;
    }
    const endpoints = graphPosToEndpoints({roadId: ghost.roadId, offsetM: ghost.offsetM});
    if (!endpoints?.candidates?.length) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('init-no-endpoints', {
                roadId: ghost.roadId,
                offsetM: ghost.offsetM,
                hasEndpoints: !!endpoints,
                candidateCount: endpoints?.candidates?.length,
            });
        }
        return;
    }
    let best = endpoints.candidates[0];
    for (const c of endpoints.candidates) {
        if (c.attachDist < best.attachDist || (c.attachDist === best.attachDist && c.nodeId < best.nodeId)) {
            best = c;
        }
    }
    ghost.graphNodeId = best.nodeId;
    const node = game.nodes[best.nodeId];
    if (node) {
        ghost.roadId = node.roadId;
        ghost.offsetM = node.offsetM;
    }
    if (isHunterGhost(ghost)) {
        pushHunterDecisionLog('init-success', {
            nodeId: ghost.graphNodeId,
            attachDist: best.attachDist,
            roadId: ghost.roadId,
            offsetM: ghost.offsetM,
        });
    }
}


function syncGhostPoseFromGraphState(ghost) {
    ghost.renderPoint = null;
    if (ghost.edgeToNodeId != null && ghost.edgeFromNodeId != null && ghost.edgeLengthM > 0) {
        const fromNode = game.nodes[ghost.edgeFromNodeId];
        const toNode = game.nodes[ghost.edgeToNodeId];
        if (!fromNode || !toNode) {
            return;
        }
        const t = clamp(ghost.edgeProgressM / ghost.edgeLengthM, 0, 1);
        const sameRoad = fromNode.roadId === toNode.roadId && !(typeof ghost.edgeRoadId === 'string' && ghost.edgeRoadId.startsWith('x-'));
        if (sameRoad) {
            ghost.roadId = fromNode.roadId;
            ghost.offsetM = fromNode.offsetM + (toNode.offsetM - fromNode.offsetM) * t;
            ghost.direction = toNode.offsetM >= fromNode.offsetM ? +1 : -1;
        } else {
            ghost.renderPoint = [
                fromNode.point[0] + (toNode.point[0] - fromNode.point[0]) * t,
                fromNode.point[1] + (toNode.point[1] - fromNode.point[1]) * t,
            ];
            if (t < 0.5) {
                ghost.roadId = fromNode.roadId;
                ghost.offsetM = fromNode.offsetM;
                ghost.direction = fromNode.offsetM <= toNode.offsetM ? +1 : -1;
            } else {
                ghost.roadId = toNode.roadId;
                ghost.offsetM = toNode.offsetM;
                ghost.direction = fromNode.offsetM <= toNode.offsetM ? +1 : -1;
            }
        }
        return;
    }
    if (ghost.graphNodeId != null) {
        const node = game.nodes[ghost.graphNodeId];
        if (node) {
            ghost.roadId = node.roadId;
            ghost.offsetM = node.offsetM;
            ghost.renderPoint = [...node.point];
        }
    }
}


function rememberGhostNodeVisit(ghost, nodeId) {
    if (!Array.isArray(ghost.recentNodeIds)) {
        ghost.recentNodeIds = [];
    }
    ghost.recentNodeIds.push(nodeId);
    if (ghost.recentNodeIds.length > GHOST_RECENT_NODE_MEMORY) {
        ghost.recentNodeIds.shift();
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
    const controlPoints = roadInfo?.controlPoints;
    if (!roadInfo || !controlPoints || !controlPoints.length) {
        return null;
    }

    const target = clamp(pos.offsetM, 0, roadInfo.lengthM);
    let hi = controlPoints.length;
    let lo = 0;
    while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (controlPoints[mid].offsetM < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const left = controlPoints[Math.max(0, lo - 1)];
    const right = controlPoints[Math.min(controlPoints.length - 1, lo)];
    const candidates = [];
    const pushCandidate = cp => {
        if (!cp) {
            return;
        }
        const nodeId = cp.nodeId;
        if (!candidates.some(x => x.nodeId === nodeId)) {
            candidates.push({
                nodeId,
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
    const startMs = performance.now();
    let settled = 0;
    const dist = new Array(game.nodes.length).fill(Infinity);
    const prev = new Array(game.nodes.length).fill(-1);
    const visited = new Array(game.nodes.length).fill(false);
    if (startNodeId == null || startNodeId < 0 || startNodeId >= dist.length) {
        return {dist, prev};
    }
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
        settled++;
        const edges = game.adjacency.get(u) || [];
        for (const e of edges) {
            const alt = dist[u] + e.lengthM;
            if (alt < dist[e.to]) {
                dist[e.to] = alt;
                prev[e.to] = u;
            }
        }
    }

    if (game.currentTickProfile) {
        game.currentTickProfile.dijkstraCalls += 1;
        game.currentTickProfile.dijkstraMs += (performance.now() - startMs);
        game.currentTickProfile.dijkstraSettled += settled;
    }

    return {dist, prev};
}


function buildNodeDistanceFieldToPos(targetPos) {
    const target = graphPosToEndpoints(targetPos);
    if (!target?.candidates?.length) {
        return null;
    }
    const sp = shortestNodePathsToTargetCandidates(target.candidates);
    return {
        dist: sp.dist,
        nextHop: sp.nextHop,
        targetNodeId: target.candidates[0]?.nodeId ?? null,
    };
}


function shortestNodePathsToTargetCandidates(candidates) {
    const dist = new Array(game.nodes.length).fill(Infinity);
    const nextHop = new Array(game.nodes.length).fill(-1);
    const visited = new Array(game.nodes.length).fill(false);
    if (!Array.isArray(candidates) || !candidates.length) {
        return {dist, nextHop};
    }
    for (const tc of candidates) {
        const nodeId = tc?.nodeId;
        const attach = tc?.attachDist;
        if (nodeId == null || nodeId < 0 || nodeId >= dist.length || !Number.isFinite(attach)) {
            continue;
        }
        if (attach < dist[nodeId]) {
            dist[nodeId] = attach;
            nextHop[nodeId] = nodeId;
        }
    }

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
        const incoming = game.reverseAdjacency.get(u) || [];
        for (const e of incoming) {
            const alt = dist[u] + e.lengthM;
            if (alt + 1e-6 < dist[e.from]) {
                dist[e.from] = alt;
                nextHop[e.from] = u;
            } else if (Math.abs(alt - dist[e.from]) <= 1e-6) {
                if (nextHop[e.from] === -1 || u < nextHop[e.from]) {
                    nextHop[e.from] = u;
                }
            }
        }
    }

    return {dist, nextHop};
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

    return best;
}


function buildPathCache(fromPos) {
    const from = graphPosToEndpoints(fromPos);
    if (!from) {
        return null;
    }
    const paths = from.candidates.map(fc => ({
        fc,
        sp: shortestNodePaths(fc.nodeId),
    }));
    return {fromPos, from, paths};
}


function shortestPathInfoFromCache(pathCache, toPos) {
    if (game.currentTickProfile) {
        game.currentTickProfile.pathQueries += 1;
    }
    if (!pathCache) {
        return {distanceM: Infinity, initialDirection: null};
    }
    const b = graphPosToEndpoints(toPos);
    if (!b) {
        return {distanceM: Infinity, initialDirection: null};
    }

    let best = {
        distanceM: Infinity,
        initialDirection: null,
    };

    for (const {fc, sp} of pathCache.paths) {
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
    let dt = 1 / SIM_HZ;
    if (game.lastSimStepTs != null) {
        // Use wall-clock delta so movement stays true even if the sim loop lags.
        dt = clamp((now - game.lastSimStepTs) / 1000, 0.02, 0.35);
    }
    game.lastSimStepTs = now;
    game.lastSimDtMs = dt * 1000;

    const profile = {
        tick: ++game.tickCounter,
        dtMs: dt * 1000,
        startMs: performance.now(),
        cacheMs: 0,
        ghostsMs: 0,
        pelletMs: 0,
        hudMs: 0,
        totalMs: 0,
        dijkstraMs: 0,
        dijkstraCalls: 0,
        dijkstraSettled: 0,
        pathQueries: 0,
        phase: game.phase,
    };
    game.currentTickProfile = profile;

    const playerPointNow = game.player.point;
    const playerPointPrev = game.player.lastSimPoint || playerPointNow;

    const cacheStartMs = performance.now();
    const playerPathCache = buildPathCache(game.player.graphPos);
    const playerNodeField = buildNodeDistanceFieldToPos(game.player.graphPos);
    profile.cacheMs = performance.now() - cacheStartMs;
    game.playerPathCache = playerPathCache;
    game.playerNodeField = playerNodeField;

    if (game.phase === 'respawning') {
        const sec = Math.max(0, Math.ceil((game.respawnDeadlineTs - now) / 1000));
        showMessage(`Caught by ghost. Safe respawn in ${sec}s...`, true);
        if (now >= game.respawnDeadlineTs) {
            finishRespawn(now);
            showMessage('Safe respawn applied', false);
        }
        const hudStartMs = performance.now();
        updateHud();
        profile.hudMs = performance.now() - hudStartMs;
        requestDraw();
        finishTickProfile(profile);
        return;
    }

    if (game.phase === 'gameover') {
        showMessage('Game Over', true);
        const hudStartMs = performance.now();
        updateHud();
        profile.hudMs = performance.now() - hudStartMs;
        requestDraw();
        finishTickProfile(profile);
        return;
    }

    const ghostsStartMs = performance.now();
    for (const ghost of game.ghosts) {
        const beforePos = ghost.roadId ? {roadId: ghost.roadId, offsetM: ghost.offsetM} : null;
        stepGhost(ghost, dt, playerPathCache, playerNodeField);
        if (beforePos && ghost.roadId) {
            ghost.animFromPos = beforePos;
            ghost.animToPos = {roadId: ghost.roadId, offsetM: ghost.offsetM};
            ghost.animStartTs = now;
            ghost.animDurationMs = Math.max(1, dt * 1000);
        }
        const gPos = {roadId: ghost.roadId, offsetM: ghost.offsetM};
        const p = shortestPathInfoFromCache(playerPathCache, gPos);
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
        const ghostPointPrev = beforePos ? graphPosToPoint(beforePos) : graphPosToPoint(gPos);
        const ghostPointNow = ghost.renderPoint || graphPosToPoint(gPos);
        const sweptEuclidM = minMovingPointDistanceM(playerPointPrev, playerPointNow, ghostPointPrev, ghostPointNow);
        const sweptAdjustedDistanceM = sweptEuclidM - game.collisionThresholdMeters;
        const collisionAdjustedDistanceM = Math.min(adjustedDistanceM, sweptAdjustedDistanceM);
        ghost.lastAdjustedDistanceToPlayer = collisionAdjustedDistanceM;
        if (now >= game.catchGraceUntilTs && Number.isFinite(collisionAdjustedDistanceM) && collisionAdjustedDistanceM <= 0) {
            game.lives -= 1;
            game.phase = 'respawning';
            game.respawnDeadlineTs = now + 10000;
            showMessage('Caught by ghost. Safe respawn in 10s...', true);
            break;
        }
    }
    profile.ghostsMs = performance.now() - ghostsStartMs;
    game.player.lastSimPoint = game.player.point ? [...game.player.point] : null;

    if (now - game.lastPelletHintTs > 1000) {
        const pelletStartMs = performance.now();
        game.lastPelletHintTs = now;
        game.closestPelletHint = computeClosestPelletHint(playerPathCache);
        profile.pelletMs = performance.now() - pelletStartMs;
    }

    const hudStartMs = performance.now();
    updateHud();
    profile.hudMs = performance.now() - hudStartMs;
    requestDraw();
    finishTickProfile(profile);
}


function finishTickProfile(profile) {
    profile.totalMs = performance.now() - profile.startMs;
    game.currentTickProfile = null;
}


function finishRespawn(nowTs, graceMs=CATCH_GRACE_PERIOD_MS) {
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


function stepGhost(ghost, dt, playerPathCache=null, playerNodeField=null) {
    if (ghost.behavior === 'pursue' || ghost.behavior === 'intercept') {
        stepGraphGhost(ghost, dt, playerPathCache, playerNodeField);
        return;
    }
    if (!ghost.roadId) {
            return; // Early exit if ghost has no roadId
    }
    const nowTs = Date.now();
    let remainingM = Math.max(0, ghostEffectiveSpeedMps(ghost) * dt);
    let hops = 0;
    const visitedNodeIds = new Set();
    while (remainingM > 1e-6 && hops < 24) {
        hops++;
        const road = game.roads.get(ghost.roadId);
        if (!road) {
            return;
        }

        const stopNode = findNextStopNodeOnRoad(road, ghost.offsetM, ghost.direction);
        if (!stopNode) {
            return;
        }
        const distToStop = Math.max(0, Math.abs(stopNode.offsetM - ghost.offsetM));
        if (remainingM < distToStop) {
            ghost.offsetM = clamp(ghost.offsetM + ghost.direction * remainingM, 0, road.lengthM);
            return;
        }

        // Reach the next decision/terminal node and keep leftover distance for the next road/hop.
        ghost.offsetM = stopNode.offsetM;
        remainingM -= distToStop;
        if (distToStop <= GHOST_NODE_EPS_M) {
            remainingM = Math.max(0, remainingM - GHOST_MIN_HOP_CONSUME_M);
        }

        const nodeId = stopNode.nodeId;
        visitedNodeIds.add(nodeId);
        const edges = game.adjacency.get(nodeId) || [];
        if (!edges.length) {
            if (isHunterGhost(ghost)) {
                pushHunterDecisionLog('no-edges', {
                    ghostId: ghost.id,
                    nodeId,
                    roadId: ghost.roadId,
                    offsetM: ghost.offsetM,
                    direction: ghost.direction,
                });
            }
            ghost.direction *= -1;
            ghost.lastDirectionChangeTs = nowTs;
            continue;
        }

        const choice = chooseGhostEdge(ghost, nodeId, edges, playerPathCache, visitedNodeIds, playerNodeField);
        if (!choice) {
            if (isHunterGhost(ghost)) {
                pushHunterDecisionLog('no-choice', {
                    ghostId: ghost.id,
                    nodeId,
                    roadId: ghost.roadId,
                    offsetM: ghost.offsetM,
                    direction: ghost.direction,
                });
            }
            ghost.direction *= -1;
            ghost.lastDirectionChangeTs = nowTs;
            continue;
        }
        const destNode = game.nodes[choice.to];
        if (!destNode || !game.roads.get(destNode.roadId)) {
            ghost.direction *= -1;
            ghost.lastDirectionChangeTs = nowTs;
            continue;
        }
        ghost.lastNodeId = nodeId;
        rememberGhostNodeVisit(ghost, nodeId);
        const enteredViaCrossRoad = destNode.roadId !== road.id;
        if (enteredViaCrossRoad) {
            ghost.roadId = destNode.roadId;
            ghost.offsetM = destNode.offsetM;
        } else {
            ghost.roadId = road.id;
            ghost.offsetM = stopNode.offsetM;
        }
        const newRoad = game.roads.get(ghost.roadId);
        let newDirection = choice.direction;
        if (ghost.direction !== newDirection) {
            ghost.lastDirectionChangeTs = nowTs;
        }
        const preClampDirection = newDirection;
        ghost.direction = newDirection;
        if (newRoad) {
            ghost.offsetM = clamp(ghost.offsetM, 0, newRoad.lengthM);
        }
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('committed', {
                ghostId: ghost.id,
                fromNodeId: nodeId,
                toNodeId: choice.to,
                choiceRoadId: choice.roadId,
                enteredViaCrossRoad,
                choiceDirection: choice.direction,
                preClampDirection,
                finalDirection: ghost.direction,
                roadId: ghost.roadId,
                offsetM: ghost.offsetM,
                playerRoadId: game.player.graphPos?.roadId,
                playerOffsetM: game.player.graphPos?.offsetM,
            });
        }
    }
}


function stepGraphGhost(ghost, dt, playerPathCache=null, playerNodeField=null) {
    if (ghost.graphNodeId == null) {
        initGhostGraphState(ghost);
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('init-graph-state', {
                graphNodeId: ghost.graphNodeId,
                roadId: ghost.roadId,
                offsetM: ghost.offsetM,
            });
        }
    }
    if (ghost.graphNodeId == null && ghost.edgeToNodeId == null) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('early-return-no-state', {
                graphNodeId: ghost.graphNodeId,
                edgeToNodeId: ghost.edgeToNodeId,
            });
        }
        return;
    }
    const nowTs = Date.now();
    let remainingM = Math.max(0, ghostEffectiveSpeedMps(ghost) * dt);
    if (isHunterGhost(ghost)) {
        pushHunterDecisionLog('step-start', {
            dt,
            remainingM,
            graphNodeId: ghost.graphNodeId,
            edgeToNodeId: ghost.edgeToNodeId,
            edgeProgressM: ghost.edgeProgressM,
            edgeLengthM: ghost.edgeLengthM,
        });
    }
    let hops = 0;
    while (remainingM > 1e-6 && hops < 48) {
        hops++;
        if (ghost.edgeToNodeId == null) {
            const nodeId = ghost.graphNodeId;
            const edges = game.adjacency.get(nodeId) || [];
            if (!edges.length) {
                if (isHunterGhost(ghost)) {
                    pushHunterDecisionLog('no-edges-available', {
                        nodeId,
                        graphNodeId: ghost.graphNodeId,
                    });
                }
                break;
            }
            const choice = chooseGhostEdge(ghost, nodeId, edges, playerPathCache, null, playerNodeField);
            if (!choice) {
                if (isHunterGhost(ghost)) {
                    pushHunterDecisionLog('choose-edge-failed', {
                        nodeId,
                        edgeCount: edges.length,
                    });
                }
                break;
            }
            if (isHunterGhost(ghost)) {
                pushHunterDecisionLog('edge-chosen', {
                    fromNodeId: nodeId,
                    toNodeId: choice.to,
                    edgeLengthM: choice.lengthM,
                });
            }
            ghost.lastNodeId = nodeId;
            rememberGhostNodeVisit(ghost, nodeId);
            ghost.edgeFromNodeId = nodeId;
            ghost.edgeToNodeId = choice.to;
            ghost.edgeLengthM = Math.max(0.001, choice.lengthM);
            ghost.edgeProgressM = 0;
            ghost.edgeRoadId = choice.roadId;
            ghost.direction = choice.direction;
            if (isHunterGhost(ghost)) {
                pushHunterDecisionLog('committed', {
                    ghostId: ghost.id,
                    fromNodeId: nodeId,
                    toNodeId: choice.to,
                    choiceRoadId: choice.roadId,
                    enteredViaCrossRoad: typeof choice.roadId === 'string' && choice.roadId.startsWith('x-'),
                    choiceDirection: choice.direction,
                    preClampDirection: choice.direction,
                    finalDirection: choice.direction,
                    roadId: ghost.roadId,
                    offsetM: ghost.offsetM,
                    playerRoadId: game.player.graphPos?.roadId,
                    playerOffsetM: game.player.graphPos?.offsetM,
                });
            }
        }

        const remainingOnEdge = Math.max(0, ghost.edgeLengthM - ghost.edgeProgressM);
        const stepM = Math.min(remainingM, remainingOnEdge);
        ghost.edgeProgressM += stepM;
        remainingM -= stepM;
        syncGhostPoseFromGraphState(ghost);
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('edge-progress', {
                edgeProgressM: ghost.edgeProgressM,
                edgeLengthM: ghost.edgeLengthM,
                stepM,
                remainingM,
                roadId: ghost.roadId,
                offsetM: ghost.offsetM,
            });
        }

        if (ghost.edgeProgressM + 1e-6 >= ghost.edgeLengthM) {
            ghost.graphNodeId = ghost.edgeToNodeId;
            ghost.edgeFromNodeId = null;
            ghost.edgeToNodeId = null;
            ghost.edgeLengthM = 0;
            ghost.edgeProgressM = 0;
            ghost.edgeRoadId = null;
            syncGhostPoseFromGraphState(ghost);
        } else {
            break;
        }
        if (stepM <= 1e-6) {
            break;
        }
        ghost.lastDirectionChangeTs = nowTs;
    }
}


function findNextStopNodeOnRoad(roadInfo, offsetM, direction) {
    const cps = roadInfo?.controlPoints;
    if (!cps || !cps.length) {
        return null;
    }
    const eps = GHOST_NODE_EPS_M;
    if (direction >= 0) {
        for (let i = 0; i < cps.length; i++) {
            const cp = cps[i];
            if (cp.offsetM <= offsetM + eps) {
                continue;
            }
            return {nodeId: cp.nodeId, offsetM: cp.offsetM};
        }
        const last = cps[cps.length - 1];
        return {nodeId: last.nodeId, offsetM: last.offsetM};
    }

    for (let i = cps.length - 1; i >= 0; i--) {
        const cp = cps[i];
        if (cp.offsetM >= offsetM - eps) {
            continue;
        }
        return {nodeId: cp.nodeId, offsetM: cp.offsetM};
    }
    const first = cps[0];
    return {nodeId: first.nodeId, offsetM: first.offsetM};
}


function ghostEffectiveSpeedMps(ghost) {
    if (game.difficulty === 'max') {
        return 100 / 3.6;
    }
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


function chooseGhostEdge(ghost, nodeId, edges, playerPathCache=null, avoidNodeIds=null, playerNodeField=null) {
    let options = edges.slice();
    if (!options.length) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('empty-options', {ghostId: ghost.id, nodeId});
        }
        return null;
    }
    if (ghost.behavior === 'wander' && ghost.lastNodeId != null && options.length > 1) {
        const nonBacktrack = options.filter(e => e.to !== ghost.lastNodeId);
        if (nonBacktrack.length) {
            options = nonBacktrack;
        }
    }
    if (avoidNodeIds && ghost.behavior === 'wander' && options.length > 1) {
        const nonVisited = options.filter(e => !avoidNodeIds.has(e.to));
        if (nonVisited.length) {
            options = nonVisited;
        }
    }

    if (!game.player.graphPos) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('no-player-pos-random', {
                ghostId: ghost.id,
                nodeId,
                optionCount: options.length,
            });
        }
        return options[Math.floor(Math.random() * options.length)];
    }

    const interceptTarget = ghost.behavior === 'intercept' ? getPlayerInterceptTarget() : null;
    const interceptTargetPos = ghost.behavior === 'intercept' ? (interceptTarget || game.player.graphPos) : null;
    const interceptNodeField = ghost.behavior === 'intercept' && interceptTargetPos ?
        buildNodeDistanceFieldToPos(interceptTargetPos) : null;
    const currentNodeDistToPlayer = ghost.behavior === 'pursue' && playerNodeField?.dist ?
        playerNodeField.dist[nodeId] : Infinity;
    const currentNodeDistToIntercept = ghost.behavior === 'intercept' && interceptNodeField?.dist ?
        interceptNodeField.dist[nodeId] : Infinity;
    const pursueNextHop = ghost.behavior === 'pursue' && playerNodeField?.nextHop ?
        playerNodeField.nextHop[nodeId] : -1;
    const interceptNextHop = ghost.behavior === 'intercept' && interceptNodeField?.nextHop ?
        interceptNodeField.nextHop[nodeId] : -1;

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
        const dToPlayerCached = shortestPathInfoFromCache(playerPathCache, candidate).distanceM;
        const congestion = game.ghosts.filter(g => g !== ghost && g.roadId === dest.roadId).length;
        let score;
        let pursueFallbackScore = dToPlayerCached;
        if (ghost.behavior === 'pursue') {
            const dFromNextNode = playerNodeField?.dist ? playerNodeField.dist[edge.to] : Infinity;
            score = edge.lengthM + dFromNextNode;
            pursueFallbackScore = score;
        } else if (ghost.behavior === 'intercept') {
            const dFromNextNode = interceptNodeField?.dist ? interceptNodeField.dist[edge.to] : Infinity;
            score = edge.lengthM + dFromNextNode;
            pursueFallbackScore = score;
        } else {
            const randomJitter = Math.random() * ghost.randomnessM;
            score = dToPlayerCached + congestion * ghost.spreadPenaltyM + randomJitter;
        }
        candidates.push({edge, score, pursueFallbackScore});
    }

    if (!candidates.length) {
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('no-candidates', {
                ghostId: ghost.id,
                nodeId,
                optionCount: options.length,
            });
        }
        return options[0];
    }

    let sortedCandidates = candidates;
    if (ghost.behavior === 'pursue' && Number.isFinite(currentNodeDistToPlayer)) {
        const nonIncreasing = candidates.filter(c => c.score <= currentNodeDistToPlayer + 1e-6);
        if (nonIncreasing.length) {
            sortedCandidates = nonIncreasing;
        }
    } else if (ghost.behavior === 'intercept' && Number.isFinite(currentNodeDistToIntercept)) {
        const nonIncreasing = candidates.filter(c => c.score <= currentNodeDistToIntercept + 1e-6);
        if (nonIncreasing.length) {
            sortedCandidates = nonIncreasing;
        }
    }

    sortedCandidates = sortedCandidates.slice().sort((a, b) => (a.score - b.score) || (a.edge.to - b.edge.to));
    if (ghost.behavior === 'pursue') {
        const selected = sortedCandidates.find(c => c.edge.to === pursueNextHop) || sortedCandidates[0];
        if (isHunterGhost(ghost)) {
            pushHunterDecisionLog('choose-edge', {
                ghostId: ghost.id,
                nodeId,
                ghostRoadId: ghost.roadId,
                ghostOffsetM: fmtDecisionNum(ghost.offsetM),
                playerRoadId: game.player.graphPos?.roadId,
                playerOffsetM: fmtDecisionNum(game.player.graphPos?.offsetM),
                currentNodeDist: fmtDecisionNum(currentNodeDistToPlayer),
                nextHopNodeId: pursueNextHop,
                candidates: candidates.map(c => ({
                    toNodeId: c.edge.to,
                    edgeRoadId: c.edge.roadId,
                    edgeDirection: c.edge.direction,
                    edgeLengthM: fmtDecisionNum(c.edge.lengthM),
                    score: fmtDecisionNum(c.score),
                })),
                chosen: {
                    toNodeId: selected.edge.to,
                    edgeRoadId: selected.edge.roadId,
                    edgeDirection: selected.edge.direction,
                    score: fmtDecisionNum(selected.score),
                },
            });
        }
        return selected.edge;
    }
    if (ghost.behavior === 'intercept') {
        const selected = sortedCandidates.find(c => c.edge.to === interceptNextHop) || sortedCandidates[0];
        return selected.edge;
    }
    if (Math.random() < ghost.chaseBias) {
        return sortedCandidates[0].edge;
    }

    const top = sortedCandidates.slice(0, Math.min(3, sortedCandidates.length));
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


function getPlayerInterceptTarget() {
    const playerPos = game.player.graphPos;
    if (!playerPos) {
        return null;
    }
    const roadInfo = game.roads.get(playerPos.roadId);
    const anchors = roadInfo?.pathAnchors;
    if (!anchors || !anchors.length) {
        return null;
    }
    const dir = game.player.travelDir || +1;
    const eps = 0.5;
    if (dir >= 0) {
        const next = anchors.find(a => a.offsetM > playerPos.offsetM + eps);
        return next ? {roadId: playerPos.roadId, offsetM: next.offsetM} : null;
    }
    for (let i = anchors.length - 1; i >= 0; i--) {
        if (anchors[i].offsetM < playerPos.offsetM - eps) {
            return {roadId: playerPos.roadId, offsetM: anchors[i].offsetM};
        }
    }
    return null;
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


function computeClosestPelletHint(playerPathCache=null) {
    if (!game.player.graphPos) {
        return null;
    }
    const cache = playerPathCache || buildPathCache(game.player.graphPos);
    let best = null;
    for (const pellet of game.pellets) {
        if (pellet.collected) {
            continue;
        }
        const target = {roadId: pellet.roadId, offsetM: pellet.offsetM};
        const p = shortestPathInfoFromCache(cache, target);
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
        if (ghostChipEls[i]) {
            ghostChipEls[i].style.backgroundColor = ghost.color;
        }
        if (ghostNameEls[i]) {
            ghostNameEls[i].textContent = ghost.name || `${i + 1}`;
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
    const animating = renderGhostEntities();
    renderPelletEntities();
    if (animating) {
        requestDraw();
    }
}


function renderGhostEntities() {
    let animating = false;
    const now = Date.now();
    for (const ghost of game.ghosts) {
        if (!ghost.roadId) {
            continue;
        }
        let p = null;
        const fromPos = ghost.animFromPos;
        const toPos = ghost.animToPos;
        if (fromPos && toPos && ghost.animDurationMs > 0) {
            const t = clamp((now - ghost.animStartTs) / ghost.animDurationMs, 0, 1);
            if (t < 1) {
                animating = true;
            }
            if (fromPos.roadId === toPos.roadId) {
                const offsetM = fromPos.offsetM + (toPos.offsetM - fromPos.offsetM) * t;
                p = graphPosToPoint({roadId: toPos.roadId, offsetM});
            } else {
                const a = graphPosToPoint(fromPos);
                const b = graphPosToPoint(toPos);
                if (a && b) {
                    p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                }
            }
        }
        if (!p) {
            p = ghost.renderPoint || graphPosToPoint({roadId: ghost.roadId, offsetM: ghost.offsetM});
        }
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
    return animating;
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
