# Haunted Fetch — Godot presentation

Run `./membrane/run_halloween.command` from L1R6, or double-click the command
file in Finder. Default URL: http://127.0.0.1:8080. Use `PORT=8082` when the
original server occupies 8080. Godot 4.5.1 is installed locally in membrane/.tools;
set GODOT_BIN to use another executable. Binaries/import caches are Git-ignored.

## Architecture and scope

MuJoCo remains authoritative for physics, pose, depth, collisions, pickup/drop,
state transitions, locks and time. Godot is a visual renderer without physics
bodies or robot controls. Authenticated private loopback snapshots drive all
three cameras from the same world. The public agent observations remain pixels,
single-ray depth and sanitized robot state. Observer data is never passed to the
agent. The original server.py and console.html are not modified.

The later request for navigation, height and timing corrections DOES change
agent/navigation/perception, planner, sim.py, world.py and recorder.py. Therefore
this final version must not be described as byte-for-byte unchanged behavior.
Changes are bounded to those requested corrections; the MuJoCo engine and
existing model-provider workflow are retained.

## Views and appearance

- FPV: 640×480, horizontal FOV 90°, height 0.5 m, downward pitch 10°,
  existing forward lens offset 0.18 m. Enlarged robot forearms and two-finger
  grippers are visible in FPV and follow the held object's actual side surface.
- Top: 1012×760 orthographic Godot image, trajectory, local click annotations.
- Room: 1440×1080 cinematic observer, larger dashboard panel. Foreground walls
  are hidden only for the dollhouse views; FPV includes all walls.

Purple paneled walls, webs, bats, pumpkins, tiled floor and an enlarged orange capsule
robot are procedural 3D reconstructions of the supplied references. The shared
white 1.5 m square drop marker carries a black skull and skeleton design. The fixed
demo scene contains two equal-size cubes in different colours, one tall cone, and
two spheres. These are
not the original meshes or an exact photorealistic reproduction. The white skeleton platform preserves a measurable visual delivery marker. Test barriers use
neutral material for camera-based detection. There is no normal-room fallback
if Godot fails.

## Timing and mathematics

Physics remains 250 Hz (4 ms). Recording samples integer crossings of 30×sim
clock, avoiding the old 250/8 ≈ 31.25 Hz cadence. Godot schedules explicit draws at 30 FPS (transport polling runs faster);
HTTP streams publish at a 30 Hz target. A cap is not a hardware throughput
promise; duplicate static frames and slower wall-clock execution are possible.
The local encoded clip was measured at 30.0 FPS. Video letterboxes the 4:3
camera into 1280×720 instead of stretching physical proportions. Clips outside
12–18 seconds are explicitly resampled to that duration; competition acceptance
of time compression needs organizer clarification.

Ground-contact rays and calibrated camera pitch recover metric range and
height. Box height uses visible vertical sides where available so its distant
top face does not inflate the estimate. Clipped tops cannot win a superlative.
Thin colored outlines are rejected as objects. These are estimates from images,
not perfect measurements under arbitrary occlusion or lighting.

Navigation uses inflated 10 cm occupancy, 8-connected A* with sqrt(2) diagonals,
line-of-sight simplification and 36 release-side candidates. It minimizes the
sampled candidate routes, not exact continuous global distance. Barriers are
inferred from camera silhouettes. Missing routes no longer cause a direct hop
through an obstacle. Occluded pads trigger alternate-side observation. Action
budget is at most 60, including deterministic surveys.

Physics corrections include per-object cone meshes, random height/geometry
consistency, no lowering tall carried objects into the floor on drop, and
carried-object checks during turns. This remains the original kinematic robot
and idealized grasp, not a torque-controlled manipulation simulator.

## Blockage UI

Load clear, horizontal-front, left-open, right-open or sealed scenarios from the
console. A private engine diagnostic checks complete enclosure by rectangular
barriers and displays a centered red 75vw × 75vh error dialog only when enclosed.
It does not expose coordinates or scene metadata to the agent. A partial wall
or an uncertain camera route is not mislabeled as complete enclosure. The
verdict is specific to the configured rectangular barriers; it is not a general
mesh reachability proof. A gap narrower than the robot/held object is physically
unusable even though it is not a mathematical 360° enclosure.

## Verification

Run from membrane:

    ../.venv/bin/python -m pytest tests -m 'not integration' -q
    MEMBRANE_ENGINE=http://127.0.0.1:8082 ../.venv/bin/python scripts/verify_halloween.py

The renderer check is read-only; run while idle. The suite has 150 passing
non-live checks, including camera-height inversion across five distances,
clipped-height rejection, enclosure vs partial barriers, alternate approaches,
unsafe fallback rejection and exactly 30 captured frames per simulation second.
See FINAL_AUDIT.md for run-specific results and remaining limitations. End-to-end
checks use the free camera-driven mock policy. No paid model evaluation was
performed in these final checks. Do not claim universal success from these runs.

Godot references checked:
- https://docs.godotengine.org/en/stable/classes/class_camera3d.html
- https://docs.godotengine.org/en/stable/classes/class_engine.html

## Current dashboard and voice entry

The larger FPV and logs occupy the upper row. Rectangular trajectory and room
views are below. Fullscreen controls on FPV and observer return with the button
or Escape. The page background is black; seed/clutter selectors and the
randomisation explanation are hidden/removed. Manual trajectory markers and their click/double-click handlers are removed.
The white recorded robot path remains visible.

Dictate beside the command box uses browser speech recognition in English
(en-IN). The first speech result replaces the draft. Stop to review or edit;
Enter or Run agent starts the existing agent pipeline. Permission, silence,
network and unsupported-browser errors retain manual text entry. Some browsers
use a remote speech service, as noted in the button tooltip. No microphone
recording starts automatically. Actual microphone capture remains untested here
because browser automation is disabled; unsupported embedded browsers show a
message to open the page in Chrome or Edge.

The renderer now flushes every newly rebuilt gripper part before drawing and
resets hand animation when simulation time resets. Gold edge-connected forearms
are excluded from object ranking using camera pixels. Model/provider logic is
still covered by the existing tests; the live validation uses the free mock
policy and does not certify paid-provider responses.
