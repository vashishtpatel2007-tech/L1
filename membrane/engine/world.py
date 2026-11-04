"""Scene definition and MuJoCo XML generation.

The engine owns EVERYTHING about the scene. None of it may reach Claude.
Object data stops at this boundary -- see agent/firewall.py.

Scene contract from the brief:
  - one flat room, four walls, no furniture (furniture is a stretch-goal toggle)
  - 4-6 primitive shapes in different colours
  - at least two share a colour, at least two share a shape
  - one marked drop zone
  - one wheeled robot with a forward-facing camera
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- geometry --
ROOM_HALF = 4.0          # metres: room spans -4..+4 in x and y
WALL_H = 1.2
WALL_T = 0.1

CAM_HEIGHT = 0.5         # brief: camera 0.5 m above floor
CAM_TILT_DEG = 10.0      # brief: tilted 10 degrees down
CAM_FWD_OFFSET = 0.18    # at the robot front
FOV_H_DEG = 90.0         # brief: 90 deg horizontal FOV
IMG_W, IMG_H = 640, 480  # brief: JPEG 640x480

ROBOT_RADIUS = 0.14
PICKUP_REACH = 0.5       # brief: reach 0.5 m
PICKUP_CONE_DEG = 30.0   # brief: within +/-30 deg of heading
DROP_ZONE_HALF = 0.55

#: Vertical FOV derived from the required horizontal FOV at 4:3.
#: tan(h/2) = aspect * tan(v/2)
FOV_V_DEG = math.degrees(2 * math.atan(math.tan(math.radians(FOV_H_DEG) / 2) / (IMG_W / IMG_H)))


@dataclass
class SceneObject:
    """A fetchable object. `label` and `colour` exist ONLY inside the engine --
    they are used for scoring a run, never sent anywhere near the agent."""
    name: str                 # engine-internal id, e.g. "obj0"
    kind: str                 # sphere | box | cylinder | cone
    colour: str               # engine-internal colour word
    rgba: tuple[float, float, float, float]
    pos: tuple[float, float]
    size: tuple[float, ...]   # mujoco geom size
    height: float             # top of the object above the floor, for "tallest"
    label: str                # human description, for engine-side scoring only


def _cone_vertices(radius: float, height: float, segments: int = 24) -> str:
    """Inline convex-hull vertices for a cone. MuJoCo has no cone primitive, and
    the brief explicitly lists cones among the expected shapes."""
    pts = [f"0 0 {height / 2:.4f}"]            # apex
    for i in range(segments):
        a = 2 * math.pi * i / segments
        pts.append(f"{radius * math.cos(a):.4f} {radius * math.sin(a):.4f} {-height / 2:.4f}")
    return " ".join(pts)


@dataclass
class Scene:
    objects: list[SceneObject]
    drop_zone: tuple[float, float]
    robot_start: tuple[float, float, float]   # x, y, heading(deg)
    obstacles: list[tuple[float, float, float, float]] = field(default_factory=list)


def default_scene(with_obstacles: bool = False) -> Scene:
    """Fixed, deterministic scene.

    Colour sharing : red {sphere, box}, blue {box, cylinder}
    Shape sharing  : box {red, blue}
    Height ordering: every height is DISTINCT (0.26 < 0.30 < 0.34 < 0.40 < 0.50)
    so "tallest" and "shortest" each resolve to exactly one object. With ties,
    a judge's superlative command becomes a coin flip.
    The drop zone is green and no object is green, so "the green square" is
    unambiguous for the multi-view stretch goal.
    """
    objects = [
        SceneObject("obj0", "sphere",   "red",    (0.85, 0.12, 0.12, 1), (2.6, 1.7),
                    (0.15,), 0.30, "red sphere"),
        SceneObject("obj1", "box",      "red",    (0.85, 0.12, 0.12, 1), (-2.2, 2.3),
                    (0.17, 0.17, 0.17), 0.34, "red cube"),
        SceneObject("obj2", "box",      "blue",   (0.15, 0.35, 0.90, 1), (2.9, -1.9),
                    (0.13, 0.13, 0.13), 0.26, "blue cube"),
        SceneObject("obj3", "cylinder", "blue",   (0.15, 0.35, 0.90, 1), (-2.8, -1.4),
                    (0.14, 0.20), 0.40, "blue cylinder"),
        SceneObject("obj4", "cone",     "yellow", (0.95, 0.80, 0.10, 1), (0.4, 3.0),
                    (0.20, 0.50), 0.50, "yellow cone"),
    ]
    obstacles = [(0.0, 1.1, 1.3, 0.12)] if with_obstacles else []
    return Scene(objects=objects, drop_zone=(0.0, -2.8), robot_start=(0.0, 0.0, 0.0), obstacles=obstacles)


def build_xml(scene: Scene) -> str:
    """Generate the MuJoCo model. Objects get free joints so they can be carried
    and dropped; the robot is a mocap body (kinematic -- the brief says grasping
    physics is not required and pick-up is a snap on contact)."""
    cam_z = CAM_HEIGHT
    tilt = math.radians(CAM_TILT_DEG)
    # Camera frame: -z is the view direction, +y is image up.
    # View along body +x, tilted down by CAM_TILT_DEG; image-right is body -y.
    cam_x_axis = (0.0, -1.0, 0.0)
    cam_y_axis = (math.sin(tilt), 0.0, math.cos(tilt))
    xyaxes = " ".join(f"{v:.5f}" for v in (*cam_x_axis, *cam_y_axis))

    body_xml, asset_xml = [], []
    if any(o.kind == "cone" for o in scene.objects):
        cone = next(o for o in scene.objects if o.kind == "cone")
        asset_xml.append(
            f'<mesh name="cone_mesh" vertex="{_cone_vertices(cone.size[0], cone.size[1])}"/>'
        )

    for o in scene.objects:
        if o.kind == "cone":
            geom = f'<geom type="mesh" mesh="cone_mesh" rgba="{" ".join(map(str, o.rgba))}" mass="0.5"/>'
            z = o.size[1] / 2
        elif o.kind == "sphere":
            geom = f'<geom type="sphere" size="{o.size[0]}" rgba="{" ".join(map(str, o.rgba))}" mass="0.5"/>'
            z = o.size[0]
        elif o.kind == "box":
            geom = f'<geom type="box" size="{" ".join(map(str, o.size))}" rgba="{" ".join(map(str, o.rgba))}" mass="0.5"/>'
            z = o.size[2]
        else:  # cylinder
            geom = f'<geom type="cylinder" size="{" ".join(map(str, o.size))}" rgba="{" ".join(map(str, o.rgba))}" mass="0.5"/>'
            z = o.size[1]
        body_xml.append(
            f'<body name="{o.name}" pos="{o.pos[0]} {o.pos[1]} {z}">'
            f'<freejoint name="{o.name}_j"/>{geom}</body>'
        )

    walls = []
    for i, (cx, cy, sx, sy) in enumerate([
        (0, ROOM_HALF, ROOM_HALF, WALL_T),
        (0, -ROOM_HALF, ROOM_HALF, WALL_T),
        (ROOM_HALF, 0, WALL_T, ROOM_HALF),
        (-ROOM_HALF, 0, WALL_T, ROOM_HALF),
    ]):
        walls.append(
            f'<geom name="wall{i}" type="box" pos="{cx} {cy} {WALL_H / 2}" '
            f'size="{sx} {sy} {WALL_H / 2}" rgba="0.80 0.82 0.86 1"/>'
        )
    for i, (cx, cy, sx, sy) in enumerate(scene.obstacles):
        walls.append(
            f'<geom name="obstacle{i}" type="box" pos="{cx} {cy} 0.3" '
            f'size="{sx} {sy} 0.3" rgba="0.45 0.33 0.22 1"/>'
        )

    dz_x, dz_y = scene.drop_zone
    rx, ry, rh = scene.robot_start

    return f"""
<mujoco model="membrane_fetch">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <quality shadowsize="2048"/>
    <!-- Claude identifies objects from these pixels, so the room must be evenly
         and generously lit. Unlit corners directly cost identification accuracy. -->
    <headlight ambient="0.34 0.34 0.34" diffuse="0.26 0.26 0.26" specular="0 0 0"/>
    <map znear="0.004" zfar="40"/>
  </visual>
  <asset>
    {"".join(asset_xml)}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.85 0.92"
             rgb2="0.94 0.96 1.0" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.90 0.90 0.92"
             rgb2="0.80 0.80 0.84" width="512" height="512"/>
    <!-- specular 0: the highlight blows out the centre of the overhead view
         (which judges read as directness evidence) and adds glare to the POV. -->
    <material name="floormat" texture="floortex" texrepeat="12 12" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <!-- Directional, not positional. A point light overhead falls off with
         distance and burns a bright pool into the middle of the floor, which
         washes out the centre of the overhead view the judges read as
         directness evidence. Directional light has no attenuation, so the room
         is lit evenly and object colours stay consistent wherever they sit.
         Levels chosen by measuring OBJECT VIVIDNESS, not mean frame brightness:
         Claude identifies items by colour, so a frame that is correctly exposed
         on average but renders a saturated object as muted is a worse frame.
         At these values a red object peaks at 245 with no channel clipping. -->
    <light directional="true" dir="0 0 -1" diffuse="0.42 0.42 0.42"
           specular="0 0 0" castshadow="true"/>
    <light directional="true" dir="-0.6 -0.6 -0.5" diffuse="0.18 0.18 0.18" castshadow="false"/>
    <light directional="true" dir="0.6 0.6 -0.5" diffuse="0.15 0.15 0.15" castshadow="false"/>

    <geom name="floor" type="plane" size="{ROOM_HALF} {ROOM_HALF} 0.1" material="floormat"/>
    <!-- A low PLATFORM, not a floor decal. The brief allows "a coloured square
         or flat platform". A 12 mm decal is a few pixels tall from across the
         room through a camera 0.5 m up, so a carrying agent cannot find it and
         drives blind; raising it to 70 mm gives it visible vertical faces while
         keeping it something the robot drives onto rather than over. -->
    <geom name="dropzone" type="box" pos="{dz_x} {dz_y} 0.035"
          size="{DROP_ZONE_HALF} {DROP_ZONE_HALF} 0.035" rgba="0.10 0.78 0.25 1"/>
    <geom name="dropzone_rim" type="box" pos="{dz_x} {dz_y} 0.074"
          size="{DROP_ZONE_HALF} {DROP_ZONE_HALF} 0.004" rgba="0.92 0.97 0.30 1"
          contype="0" conaffinity="0"/>
    {"".join(walls)}
    {"".join(body_xml)}

    <body name="robot" mocap="true" pos="{rx} {ry} 0.2" euler="0 0 {-rh}">
      <!-- Collision shell is a single sphere of radius ROBOT_RADIUS. Contact with
           an object then happens at ~0.32 m; after the 0.1 m back-off the robot
           sits at ~0.42 m, comfortably inside the 0.5 m pickup reach. A larger
           or forward-extended shell makes objects physically unreachable. -->
      <geom name="robot_body" type="sphere" size="{ROBOT_RADIUS}" rgba="0.18 0.18 0.22 1"/>
      <!-- Purely visual: shows which way the robot faces in the POV clip.
           contype/conaffinity 0 keeps it out of collision detection. -->
      <geom name="robot_nose" type="box" pos="0.16 0 0.01" size="0.04 0.09 0.04"
            rgba="0.95 0.45 0.10 1" contype="0" conaffinity="0"/>
      <geom name="robot_stripe" type="box" pos="0 0 0.12" size="0.10 0.10 0.02"
            rgba="0.85 0.85 0.90 1" contype="0" conaffinity="0"/>
      <camera name="rgb" pos="{CAM_FWD_OFFSET} 0 {cam_z - 0.2}" xyaxes="{xyaxes}" fovy="{FOV_V_DEG:.4f}"/>
    </body>

    <camera name="overhead" pos="0 -0.5 9" xyaxes="1 0 0 0 1 0" fovy="60"/>
  </worldbody>
</mujoco>
"""
