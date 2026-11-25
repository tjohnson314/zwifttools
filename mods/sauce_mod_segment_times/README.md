# Segment Times

A Sauce for Zwift overlay that appends the rider, segment and estimated time
when a nearby rider completes a segment.

Copy this folder into Sauce's mods folder (shown in Settings > Mods), restart
Sauce, enable the mod and add a **Segment Times** window. The list starts empty
when the window opens and remains until it is closed.

The mod uses Sauce's course segment catalog and interpolates each rider's
`worldTime` between road-position updates at the start and finish. Times are
estimates, not official Zwift results. Only the local rider and riders visible
to Sauce can be tracked; the window must receive both crossings, and gaps over
15 seconds discard an unfinished attempt. Opening the window partway through a
segment will not produce a result for that traversal. Non-loop attempts are
also discarded when the rider changes roads or reverses direction.

For loop segments the same line starts and finishes the clock. A result requires
the rider's distance to advance by at least 80% and at most 200% of Sauce's
declared loop length, so incomplete loops and missing distances are skipped.
Segments unavailable in Sauce's catalog cannot be shown.
