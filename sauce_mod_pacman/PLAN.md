# Sauce Pacman (Scotland) Design + Implementation Status

## Current implementation status

Implemented in `sauce_mod_pacman/`:

- World locked to Scotland (`courseId = 17`).
- 1 km x 1 km viewport centered on player.
- Simulated entities:
  - Player from Zwift `athlete/self` state.
  - 4 ghosts in map simulation.
  - Pellets distributed on roads.
- Start lives: 3.
- Catch flow:
  - Enter respawn state.
  - Show countdown message.
  - Wait 10 seconds.
  - Call `teleportHome`.
- Dynamic collision threshold based on sample gap.
- Ghost and pellet edge arrows with road-distance labels.
- Difficulty selector persisted in settings (`easy`, `normal`, `hard`, `pro`).
- Gradient-aware ghost speed model using player reference table.
- Global speed reduction (`GLOBAL_GHOST_SPEED_MULT = 0.80`).
- Junction AI with personality + randomness + anti-clumping behavior.

## Collision logic with low state update rate

### Constraints

- Maximum player speed: 100 km/h = 27.7778 m/s.
- Player state updates arrive only a few times per second.
- Collision detection must remain fair even if player crosses a long distance between updates.

### Dynamic collision threshold

Use a threshold based on observed update gap:

- Let dt be seconds since previous player state sample.
- Clamp dt to [0.10, 0.50] to avoid extreme spikes.
- Let vmax = 27.7778 m/s.
- Let safety = 4.0 m.
- collisionThresholdMeters = clamp(vmax * dt + safety, 12.0, 25.0).

Practical values:

- dt = 0.20s -> threshold ~= 9.56m -> clamped to 12m.
- dt = 0.33s -> threshold ~= 13.17m.
- dt = 0.50s -> threshold ~= 17.89m.

### Collision distance metric

Use road-network distance, not straight-line distance:

- Compute shortest-path road distance between player projected position and ghost projected position.
- A catch occurs when pathDistance <= collisionThresholdMeters.

### Temporal anti-jitter guard

To avoid repeated catches from noisy updates:

- After a catch, set state to `respawning`.
- Ignore all collisions until respawn completes.

## Player viewport rules

### Visible area

- Axis-aligned square centered on player:
  - width = 1000 m
  - height = 1000 m
  - x in [playerX - 500, playerX + 500]
  - y in [playerY - 500, playerY + 500]

### Rendering

- Render roads intersecting the square.
- Render pellets/items whose projected point is inside the square.
- Render ghosts whose projected point is inside the square.
- Keep player fixed at center.

## Ghost guidance arrows

Implemented behavior:

- For each ghost inside the visible square:
  - compute shortest road-path distance to player,
  - draw ghost,
  - draw edge arrow with meter label.

Algorithm:

1. Compute shortest path from player graph position to ghost graph position.
2. Determine first direction vector from player along that path.
3. Cast ray from player center in that direction.
4. Intersect ray with viewport square boundary.
5. Place arrow at that boundary point with text like `142m`.

Notes:

- Ghost is rendered when visible.
- One arrow is drawn per visible ghost.
- Edge-overlap avoidance is not implemented yet.

## Pellet guidance arrow fallback

Requirement:

- If no pellets are inside the visible square:
  - find nearest remaining pellet by shortest road-path distance.
  - show one arrow at viewport edge along shortest path direction.
  - label with road distance in meters.

Rules:

- If at least one pellet is visible, do not show pellet guidance arrow.
- `Level clear` state is not implemented yet.

## Lives, catch flow, and teleport

### Initial lives

- lives = 3 on new game.

### On catch

1. Enter `respawning` state immediately.
2. Decrement lives by 1.
3. Show message overlay:
   - `Caught by ghost. Teleporting home in 10s...`
4. Show 10-second countdown.
5. At countdown end:
   - Call RPC `teleportHome`.
   - Reset ghosts to spawn logic.
   - Resume normal play state.

### Game over

- If lives == 0 after decrement:
  - show `Game Over`.
  - stop ghost movement/collision checks.
  - restart action is not implemented yet.

## Data model snapshot (implemented)

```text
GameState
- phase: boot | running | respawning | gameover
- lives: number
- score: number
- lastPlayerSampleTs: ms
- collisionThresholdMeters: number
- player.graphPos: GraphPos
- ghosts: GhostState[]
- pellets: PelletState[]
- closestPelletHint: {distanceM, direction} | null
- difficulty: easy | normal | hard | pro
- respawnDeadlineTs: ms | null
- powerSamples: [{ts, watts}]
- frightenedActive: boolean
- frightenedUntilTs: ms
- frightenedActivationCount: number
- ghostsEatenThisSession: number
```

```text
GhostState
- id: string
- roadId: number | null
- offsetM: number
- speedMps: number  // per-ghost personality base
- chaseBias: number
- randomnessM: number
- spreadPenaltyM: number
- targetNodeId: number | null
- lastPathDistanceToPlayer: number
- eaten: boolean
- eatenAtTs: ms | null
- respawnAtTs: ms | null
```

## Performance notes

- Current implementation recomputes shortest paths frequently and does not yet cache by `(fromNode, toNode)`.
- Simulation tick runs at 10 Hz.
- Run simulation at fixed tick (e.g. 10 Hz), independent of render FPS.

## Power Pellet / Frightened Mode

### Activation

- Track player power (watts) in a rolling 30-second window.
- Compute 30-second rolling average power each state update.
- When average power reaches the activation threshold, trigger frightened mode for 30 seconds.
- Each subsequent activation requires 100W more than the previous one.

### Difficulty scaling

| Difficulty | Base threshold |
|------------|---------------|
| Easy       | 500W          |
| Normal     | 600W          |
| Hard       | 700W          |
| Pro        | 800W          |
| Max        | 900W          |

Formula: `threshold = 500 + difficultyIndex * 100 + activationCount * 100`

### Ghost behavior when frightened

- All ghosts reverse their direction preference at junctions.
- Scoring logic inverts: ghosts prefer edges that maximize distance from the player.
- Ghosts turn dark blue and gain a distinct visual appearance.

### Eating ghosts

- During frightened mode, collision with a ghost eats it instead of catching the player.
- First ghost eaten = 100 points (10× pellet value).
- Each subsequent ghost eaten during the same frightened activation doubles: 200, 400, 800.
- The `ghostsEatenThisSession` counter resets on each new frightened activation.

### Ghost respawn after being eaten

- Eaten ghosts disappear for 30 seconds.
- After 30 seconds they respawn at a safe distance from the player.
- If frightened mode is still active when they respawn, they re-enter frightened state.

### UI: Power Bar

- Blue progress bar displayed above the message area.
- During charging: shows `{avgPower} / {threshold}W` and fills proportionally.
- During frightened: bar shows remaining time as a countdown, pulsing animation.
- Bar fill uses CSS transition for smooth movement.

## Remaining work (not blockers for first playtest)

- `Level clear` and next-level flow.
- Restart button/action on game over.
- Pathfinding optimization (priority queue / cache) for larger worlds and denser pellets.
- Better edge-arrow de-overlap when multiple ghosts are visible.
- Optional gameplay polish:
  - spawn definitions beyond first observed home road,
  - scoring UI extras,
  - settings panel beyond in-HUD difficulty.

## Acceptance criteria for current playtest

- Collision threshold scales with real sample gap and catches high-speed pass-through cases.
- Viewport strictly shows 1 km x 1 km area centered on player.
- For each visible ghost, an edge arrow is shown with road-distance label.
- If no pellets are visible, nearest-pellet edge arrow appears with road-distance label.
- Player starts with 3 lives.
- Catch flow always displays message, waits 10 seconds, then calls `teleportHome`.
- Difficulty selection changes ghost behavior and persists.
- Ghosts do not all behave identically at junctions (randomized anti-clumping chase).
