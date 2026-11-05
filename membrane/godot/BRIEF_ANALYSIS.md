# Membrane problem and USP analysis

The brief asks for an image-driven fetch robot: understand a natural-language
request, distinguish coloured primitive objects, approach and grasp the right
one, then deliver it to a marked zone. Duplicate colours and shapes make a
colour-only lookup insufficient. Tallest/shortest requires physical height
estimation because image size changes with distance.

The strongest defensible USP is **one authoritative simulation with an
image-driven robot and three synchronized ways to inspect it**. FPV shows the
agent's actual information; the top view explains its route; the cinematic room
makes the same task understandable to an audience. It is not a separate
animation imitating success. The Halloween theme is a presentation advantage,
not evidence of better reasoning accuracy or a claim of market uniqueness.

Supporting advantages are metric size estimation, alternative delivery
approaches, trajectory annotations, reproducible image/action logs, and a strict
separation between operator diagnostics and agent observations. The engine may
know object positions to render them; the agent must infer them from pixels.

## Requirements and implementation

| Requirement | Implementation / constraint |
|---|---|
| Flat room, four walls, 4–6 primitives, duplicate colours/shapes | Existing MuJoCo scene generator retained; Halloween decoration is visual |
| Only camera, optional single-ray depth, own pose to agent | Public schemas retained; private renderer snapshots never passed to agent |
| 640×480 JPEG, 90° horizontal FOV, camera height 0.5 m, downward tilt 10° | Calibrated Godot camera; original 0.18 m lens offset; synchronous transforms |
| Move 1 m/s, turn 90°/s | MuJoCo 4 ms integration retained |
| Pickup within 0.5 m and ±30° | Existing pickup checks retained |
| Collision backoff 0.1 m, bounded retries | Existing movement recovery; added carried-shape turn checks |
| Maximum 60 actions | Deterministic survey actions also count |
| Simulation frozen during model reasoning | Godot cannot independently step physics |
| MP4 at least 720p, 10–20 seconds, command overlay | 1280×720 letterbox, 30 FPS; explicit duration resampling needs organizer interpretation |
| Halloween, three screens, 30 FPS | User additions; procedural room/robot, synchronized cameras and scheduled rendering |
| Complete enclosure error | Operator-only rectangular-barrier enclosure test; red centered 75% dialog |

## Mathematics

At 640 pixels and 90° horizontal FOV, focal length is 320 pixels. Horizontal
bearing near the centre row is atan((u−320)/320), not a linear degrees-per-pixel
formula. Tilt and pixel row determine the ray's ground-plane intersection. The
base fixes range; the top ray then estimates physical height. Box side edges
avoid confusing a visible top face with extra height. Clipped tops cannot win
superlative comparisons. Occlusion and pixel noise still limit accuracy.

Navigation inflates obstacles by robot radius and clearance, uses a 10 cm
8-connected A* grid, simplifies clear segments, and samples 36 release angles.
It minimizes sampled route lengths, not exact continuous global distance.
Barrier silhouettes and delivery-pad geometry are inferred from pixels.

A complete enclosure is different from an unsuccessful visual search. A narrow
gap may remain physically unusable for the robot and held object even though it
is not a mathematical 360° enclosure. No finite demo set proves success in every
possible layout.

## Demonstration and limitations

Show a normal fetch, tallest/shortest with misleading apparent image sizes,
a horizontal barrier, both single-side openings, and complete enclosure. Show
actual outcome and action logs, including failure rather than fabricated success.

The supplied references contain no original production meshes. This is a
procedural 3D reconstruction, not an exact photorealistic reproduction. Visual
recognition is calibrated for the primitive palette and neutral barriers.
Decorative walls extend above the original collision walls. Test barriers are
an optional challenge scene. The retained robot/grasp are idealized, not a
joint/torque-controlled manipulator. Free mock-policy tests do not establish
paid language-model performance. Clip time compression and themed-scene
eligibility need organizer interpretation; this is not full certification.

Pitch: “A camera-driven fetch robot that explains its journey: see what it sees,
inspect its route, and watch the same task unfold in a fully 3D haunted room.”
