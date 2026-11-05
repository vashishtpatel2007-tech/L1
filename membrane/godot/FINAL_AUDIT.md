# Final audit — 5 November 2026

## Fullscreen, language clarification and light theme

“All views fullscreen” places the existing FPV, trajectory and room-observer panels side by side, with logs below. Exit, Escape and browser fullscreen exit restore the original panel positions. It reuses existing live images/canvas rather than opening duplicate streams. The viewport fallback is covered by JavaScript state tests. Normal layout now fills the FPV panel width, reduces map height and enlarges the observer panel. The interface has a white background, black text, legible system fonts, monochrome controls and shorter view labels. Halloween scene artwork is unchanged.

Every command now starts with eight 45-degree in-place turns before any translation or pickup. Resolution uses only surveyed camera detections. Duplicate colour/shape matches require clarification; nearest distance is not an implicit tie-break. Size adjectives and superlatives compare metric estimates; close ties, missing matches and clipped size measurements stop with an explanation. A dialog lists alternatives and returns focus to the command box. This covers the six scene colours, not arbitrary unimplemented colours or guaranteed discovery of completely hidden objects.

Both `run_selection_confirmed_ambiguous_mock` and `run_selection_confirmed_ambiguous_hosted` found two purple cones, performed eight turns then stopped, with no translation, pickup or external model call. `run_selection_confirmed_small_purple` delivered the small cone in 21 actions with zero collisions after a specific command. A separate Godot camera sweep detected all six palette colours (red, blue, yellow, purple, orange, teal). Purple segmentation now separates bright fetch objects from dim wall panels while preserving illuminated cone tips; two rendered frame fixtures protect this correction.

Live dashboard-backend mock checks delivered “round red” (19 actions), “tallest” (23), “shortest” (18) and “tall blue” (22), all with zero collisions. Browser automation remains disabled in Work settings: CSS/served HTML, JavaScript syntax, fullscreen state restoration and clarification payload wiring were checked, but native browser visual inspection and microphone hardware remain unverified. No paid-model accuracy claim is made.

## Latest reported turn-loop defect

The interrupted hosted-model run `run_ui911137f102f7` repeatedly alternated model alignment turns with obstacle-avoidance turns. Normal language commands had no comparison selection, so the fetch target was included in the obstacle list and blocked its own grasp approach. Target selection now happens before obstacle construction, and that same target supplies the approach goal. Other objects and barriers remain obstacles.

The actual camera-derived estimates from failing step 10 are stored in `tests/fixtures/hosted_target_obstacle.json`. The regression reproduces the old empty route and verifies a collision-free approach after correction. Three failed replans without more than 15 cm of position change stop with `route_unverified`; this is not represented as proof of full enclosure.

Additional hosted-path checks exposed two related conflicts: the copilot could override an existing navigator waypoint, and a successfully carried object's old floor track could block the carry route's starting cell. Navigator actions now retain route ownership; confirmed pickup retires that object's floor track and nearby carried-object detections are not re-added as floor obstacles.

Manual numbered trajectory dots, their storage, and their click handler are removed from the served dashboard. Double-click marking/clearing is also absent. The actual white robot trajectory remains.

## Automated verification

189 non-integration Python tests pass; eight live integration tests requiring a separate setup are deselected. Coverage includes the recorded target-obstacle regression, neighbouring-object retention, bounded replan failures, hosted-loop controller ownership, carried-floor-track removal, calibrated height comparisons, partial/complete barriers, safe routes, pickup/drop and exactly 30 recorded frames per simulation second.

Console JavaScript syntax and voice-input state tests pass. Served HTML contains no manual-marker storage or double-click handler. Git whitespace checks pass. Browser automation is disabled in Work settings, so final browser layout, microphone hardware and browser permissions were not exercised directly.

The full hosted-model control branch was also exercised with an offline camera-driven vision substitute (no external API calls): `run_hosted_target_regression_verified` delivered the red sphere around the horizontal barrier in **31 actions, zero collisions**, with a passing engine outcome. This tests hosted-path controller integration but does not evaluate a new paid-model response. Two intermediate failing regression runs are preserved as evidence, not counted as successes.

## Earlier completed scenario checks

These runs used Godot imagery, MuJoCo physics and the free camera-driven mock policy. They are evidence for the listed scenarios, not universal model reliability.

| Scenario | Run directory under L1R6/runs | Actions | Result | Collisions |
|---|---|---:|---|---:|
| Cube | run_verified_release_cube | 22 | Delivered | 0 |
| Tallest via dashboard backend | run_ui475af4cecd0a | 23 | Yellow cone delivered | 0 |
| Shortest immediately after reset | run_ui3373bfb3d66f | 18 | Blue sphere delivered | 0 |
| Horizontal front barrier | run_handoff_front | 23 | Red sphere delivered | 0 |
| Left approach open | run_handoff_left | 51 | Red sphere delivered | 0 |
| Right approach open | run_handoff_right | 24 | Red sphere delivered | 0 |

## Rendering and interface

Active motion measured 60 snapshots over 2.0058 seconds: **29.91 FPS**, against the 30 FPS target. Physics runs at 250 Hz. Idle images are cached; HTTP response rate is not GPU frame rate. FPV is 640×480, top view 1012×760 and observer 1440×1080.

The larger robot, visible picking/holding hands, and shared white 1.5 m drop pad with black skeleton are implemented. Hand transforms are flushed before rendering and reset between scenes. Fixed objects are two equally sized coloured cubes, one tall cone and two spheres. FPV and logs occupy the upper row; rectangular top and observer views sit below. FPV/observer have fullscreen and return controls. The dashboard uses a white background, black text and monochrome controls.

Voice dictation writes into the command box for review; Enter or Run agent starts the task. Tests cover interim/final transcript, explicit submission, denied permission, silence, unsupported browsers, locks and late results. Actual speech recognition depends on the browser and microphone; some browsers use an online service.

## Maths, obstruction and limits

Height uses calibrated floor-ray distance and top-row projection rather than apparent pixel size; clipped heights are excluded from comparisons. Geometry tests invert the camera equations at five distances. Real estimates still depend on visibility, calibration, segmentation and the primitive shape palette.

Routing uses collision-inflated 10 cm grid A*, sqrt(2) diagonal cost and sampled release approaches. It does not prove a globally shortest continuous path. Clear/front/left-open/right-open scenarios return non-enclosed; sealed returns enclosed. The diagnostic drives a centered 75vw × 75vh “ERROR — DROP LOCATION BLOCKED” dialog, restyled in white and black per the latest request. Narrow unusable gaps and unverified routes are distinct from full enclosure.

No finite test suite can guarantee every scene or every paid-model response succeeds. No new paid API evaluation was initiated for these fixes. MuJoCo remains authoritative; user-requested navigation, physics and perception corrections mean the original engine/agent are not byte-for-byte unchanged. The original server.py and console.html remain unchanged. Artwork is a procedural reconstruction rather than an exact photorealistic reproduction.

## Launch

Dashboard: http://127.0.0.1:8082/
From L1R6: `PORT=8082 ./membrane/run_halloween.command`.
Reload existing browser tabs to remove the old in-memory click handler and marks. The port 8080 service was not stopped or changed.
