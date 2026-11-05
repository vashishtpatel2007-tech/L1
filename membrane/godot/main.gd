extends Node
# All geometry and transforms stay on the ENGINE side of the boundary.
# No controller, physics, task logic, or model calls exist in this project.
var server := TCPServer.new()
var peer: StreamPeerTCP
var incoming := ""
var busy := false
var last_draw_usec: int = 0
var world := Node3D.new()
var robot := Node3D.new()
var zone := Node3D.new()
var arms := Node3D.new()
var arm_signature := ""
var previous_held := -1
var release_clock := -10.0
var last_grip := Vector3(0.48,0.22,0)
var last_radius := 0.15
var last_arm_clock := -1.0
var items := Node3D.new()
var obstacles := Node3D.new()
var trail := MeshInstance3D.new()
var views: Dictionary = {}
var cameras: Dictionary = {}
var signature := ""
var state_label: Label
var material_cache: Dictionary = {}
var stone_cache: Dictionary = {}
var orange := material(Color("e97a19"), 0.38, 0.28)
var purple := material(Color("211c28"), 0.5, 0.1)
var black := material(Color("14101c"), 0.4, 0.45)
var gold := material(Color("ffc66a"), 0.23, 0.65)
var glow := material(Color("ffd38a"), 0.3, 0.2, 1.8)

func material(c: Color, rough: float = 0.65, metal: float = 0.0, emission: float = 0.0) -> StandardMaterial3D:
	var key := str(c)+":"+str(rough)+":"+str(metal)+":"+str(emission)
	if material_cache.has(key): return material_cache[key]
	var m := StandardMaterial3D.new()
	material_cache[key] = m
	m.albedo_color = c
	m.roughness = rough
	m.metallic = metal
	if emission > 0:
		m.emission_enabled = true
		m.emission = c
		m.emission_energy_multiplier = emission
	return m

func stone(c: Color) -> ShaderMaterial:
	var key := str(c)
	if stone_cache.has(key): return stone_cache[key]
	var shader := Shader.new()
	shader.code = """shader_type spatial;
uniform vec4 tint : source_color;
varying vec3 wp;
float noise(vec3 p){return fract(sin(dot(p,vec3(12.9898,78.233,37.719)))*43758.5453);}
void vertex(){wp=(MODEL_MATRIX*vec4(VERTEX,1.0)).xyz;}
void fragment(){float grain=noise(floor(wp*950.0));float mottling=noise(floor(wp*17.0));ALBEDO=tint.rgb*(0.90+grain*0.06+mottling*0.04);ROUGHNESS=0.62+grain*0.22;METALLIC=0.08;EMISSION=vec3(0.035);}
"""
	var m := ShaderMaterial.new()
	m.shader = shader
	m.set_shader_parameter("tint",c)
	stone_cache[key] = m
	return m

func mesh(parent: Node3D, shape: Mesh, pos: Vector3, mat: Material, layer: int = 1) -> MeshInstance3D:
	var n := MeshInstance3D.new()
	n.mesh = shape
	n.material_override = mat
	n.position = pos
	n.layers = layer
	parent.add_child(n)
	return n

func box(parent: Node3D, pos: Vector3, size: Vector3, mat: Material, layer: int = 1) -> MeshInstance3D:
	var b := BoxMesh.new()
	b.size = size
	return mesh(parent, b, pos, mat, layer)

func ball(parent: Node3D, pos: Vector3, radius: float, mat: Material, layer: int = 1) -> MeshInstance3D:
	var b := SphereMesh.new()
	b.radius = radius
	b.height = radius * 2
	b.radial_segments = 32
	b.rings = 16
	return mesh(parent, b, pos, mat, layer)

func cylinder(parent: Node3D, pos: Vector3, radius: float, height: float, mat: Material, top: float = -1.0, layer: int = 1) -> MeshInstance3D:
	var b := CylinderMesh.new()
	b.bottom_radius = radius
	b.top_radius = radius if top < 0 else top
	b.height = height
	b.radial_segments = 32
	return mesh(parent, b, pos, mat, layer)

func line(parent: Node3D, a: Vector3, b: Vector3, width: float, mat: Material, layer: int = 1):
	var n := cylinder(parent, (a+b)/2, width, a.distance_to(b), mat, -1, layer)
	n.quaternion = Quaternion(Vector3.UP, (b-a).normalized())

func _ready():
	# Poll the private transport promptly; explicit draws are independently 30 Hz.
	# A 30 Hz polling loop adds an avoidable frame of round-trip delay.
	Engine.max_fps = 240
	add_child(world)
	world.add_child(robot)
	world.add_child(zone)
	world.add_child(arms)
	world.add_child(items)
	world.add_child(obstacles)
	world.add_child(trail)
	trail.layers = 8 # trajectory is visible ONLY in the overhead observer
	build_room()
	batch_static_room()
	build_robot()
	robot.scale = Vector3(1.45,1.65,1.45)
	build_zone()
	build_views()
	var port := int(OS.get_environment("FETCH_RENDER_PORT"))
	if port == 0: port = 8091
	if server.listen(port, "127.0.0.1") != OK:
		push_error("Cannot start private rendering transport")
		get_tree().quit(1)

func batch_static_room():
	# Merge static triangles by material/layer to reduce three-view draw calls.
	# Dynamic objects and camera visibility layers remain independent.
	var groups: Dictionary = {}
	var pending: Array = world.get_children()
	var originals: Array = []
	while not pending.is_empty():
		var node: Node = pending.pop_back()
		if node in [robot,zone,arms,items,obstacles,trail]: continue
		pending.append_array(node.get_children())
		if node is MeshInstance3D and node.mesh != null:
			var key := str(node.material_override.get_instance_id())+":"+str(node.layers)
			if not groups.has(key):
				var surface := SurfaceTool.new()
				surface.begin(Mesh.PRIMITIVE_TRIANGLES)
				groups[key] = [surface,node.material_override,node.layers]
			groups[key][0].append_from(node.mesh,0,world.global_transform.affine_inverse()*node.global_transform)
			originals.append(node)
	for key in groups:
		var group: Array = groups[key]
		var merged := MeshInstance3D.new()
		merged.mesh = group[0].commit()
		merged.material_override = group[1]
		merged.layers = group[2]
		world.add_child(merged)
	for node in originals: node.free()

func barrier_material() -> StandardMaterial3D:
	var m := material(Color("777777"),0.85)
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return m

func build_room():
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color("0b0611")
	env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.environment.ambient_light_color = Color("d4c5e4")
	env.environment.ambient_light_energy = 0.65
	env.environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	world.add_child(env)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-65, -25, 0)
	sun.light_color = Color("fff1df")
	sun.light_energy = 1.25
	sun.shadow_enabled = true
	world.add_child(sun)
	var rng := RandomNumberGenerator.new()
	rng.seed = 1066
	var grout := material(Color("55465e"),1.0)
	grout.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	box(world,Vector3(0,-0.07,0),Vector3(8,0.06,8),grout)
	for x in range(16):
		for z in range(16):
			var c := Color("211c29").lightened(rng.randi_range(0,7)*0.005)
			box(world, Vector3(-3.75+x*0.5,-0.025,-3.75+z*0.5), Vector3(0.492,0.05,0.492), stone(c))
	for side in range(4):
		var wall := Node3D.new()
		world.add_child(wall)
		wall.rotation.y = side*PI/2
		# Two foreground walls remain in FPV but are omitted from the dollhouse view.
		var layer := 4 if side == 0 or side == 1 else 1
		box(wall, Vector3(0,1.6,4), Vector3(8.1,3.2,0.1), purple,layer)
		for i in range(17):
			box(wall, Vector3(-4+i*0.5,1.6,3.94),Vector3(0.035,3.2,0.025),material(Color("342a3d"),0.65),layer)
		box(wall,Vector3(0,0.08,3.91),Vector3(8,0.16,0.07),black,layer)
		box(wall,Vector3(0,3.08,3.9),Vector3(8,0.12,0.09),gold,layer)
		for i in range(3):
			var px := -2.7+i*2.7
			pumpkin(wall,Vector3(px,2.28,3.82),layer)
			line(wall,Vector3(px,2.5,3.82),Vector3(px,3.05,3.82),0.009,black,layer)
			for j in range(3):
				box(wall,Vector3(px+0.4+j*0.12,2.8-j*0.13,3.93),Vector3(0.023,0.55+j*0.2,0.012),material(Color("671a2b")),layer)
			bat(wall,Vector3(px+0.55,1.9,3.87),layer)
		web(wall,Vector3(-3.97,3.03,3.88),layer)
		var lamp := OmniLight3D.new()
		lamp.position = Vector3(0,2.5,3.4)
		lamp.light_color = Color("ff8e42") if side%2==0 else Color("b385ff")
		lamp.light_energy = 0.55
		lamp.omni_range = 5
		wall.add_child(lamp)

func pumpkin(parent: Node3D, p: Vector3, layer: int):
	var body := ball(parent,p,0.18,orange,layer)
	body.scale = Vector3(1.1,0.85,0.4)
	for sign_x in [-1,1]:
		var eye := box(parent,p+Vector3(sign_x*0.065,0.035,-0.077),Vector3(0.05,0.05,0.012),black,layer)
		eye.rotation.z = PI/4
	box(parent,p+Vector3(0,-0.065,-0.079),Vector3(0.15,0.035,0.012),black,layer)
	box(parent,p+Vector3(0,-0.043,-0.09),Vector3(0.025,0.03,0.012),gold,layer)
	cylinder(parent,p+Vector3(0,0.17,0),0.025,0.08,black,-1,layer)

func bat(parent: Node3D, p: Vector3, layer: int):
	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_TRIANGLES)
	for s in [-1,1]:
		for tri in [[Vector3(0,0,0),Vector3(s*0.25,0.09,0),Vector3(s*0.12,-0.06,0)], [Vector3(s*0.12,-0.06,0),Vector3(s*0.25,0.09,0),Vector3(s*0.18,0.0,0)]]:
			for v in tri: im.surface_add_vertex(v)
	im.surface_end()
	var m := material(Color("0b0710"))
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	mesh(parent,im,p,m,layer)

func web(parent: Node3D, p: Vector3, layer: int):
	var silk := material(Color("8b7c96"),0.8)
	for i in range(9):
		var a := i*PI/16
		line(parent,p,p+Vector3(cos(a),-sin(a),0)*1.25,0.0025,silk,layer)
	for ring in range(1,7):
		for i in range(8):
			var a := i*PI/16
			var b := (i+1)*PI/16
			line(parent,p+Vector3(cos(a),-sin(a),0)*ring*0.2,p+Vector3(cos(b),-sin(b),0)*ring*0.2,0.002,silk,layer)

func build_robot():
	# Visual shell is excluded from the calibrated front camera. MuJoCo's
	# original collision sphere and pickup envelope are never changed.
	var capsule := CapsuleMesh.new()
	capsule.radius = 0.14
	capsule.height = 0.57
	mesh(robot,capsule,Vector3(0,0.34,0),orange,2)
	cylinder(robot,Vector3(0,0.32,0),0.141,0.009,black,-1,2)
	for s in [-1,1]:
		var wheel := cylinder(robot,Vector3(0,0.095,s*0.13),0.085,0.045,black,-1,2)
		wheel.rotation.x = PI/2
	var lens := cylinder(robot,Vector3(0.139,0.47,0),0.087,0.035,black,-1,2)
	lens.rotation.z = PI/2
	var ring := TorusMesh.new()
	ring.inner_radius = 0.06
	ring.outer_radius = 0.077
	var r := mesh(robot,ring,Vector3(0.163,0.47,0),glow,2)
	r.rotation.z = PI/2
	var glass := ball(robot,Vector3(0.16,0.47,0),0.055,material(Color("1e1235"),0.08,0.7),2)
	glass.scale.x = 0.3
	var hat := Node3D.new()
	robot.add_child(hat)
	hat.position = Vector3(-0.025,0.635,0)
	hat.rotation.z = -0.18
	cylinder(hat,Vector3.ZERO,0.09,0.012,purple,-1,2)
	cylinder(hat,Vector3(0,0.075,0),0.065,0.15,purple,0.005,2)
	cylinder(hat,Vector3(0,0.025,0),0.055,0.02,orange,0.05,2)
	box(hat,Vector3(0.05,0.025,0),Vector3(0.007,0.025,0.028),gold,2)
	var badge := cylinder(robot,Vector3(0.137,0.27,0),0.05,0.008,gold,-1,2)
	badge.rotation.z = PI/2
	ball(robot,Vector3(-0.145,0.25,0.07),0.07,material(Color("9a7244")),2).scale = Vector3(0.5,1.2,0.8)

func build_zone():
	var ink := material(Color("080808"),1)
	ink.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	var white := material(Color.WHITE,1)
	white.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	box(zone,Vector3(0,0.035,0),Vector3(1.5,0.07,1.5),ink)
	box(zone,Vector3(0,0.074,0),Vector3(1.5,0.008,1.5),white)
	# One shared physical decal mesh is seen by all three cameras.
	var skull := ball(zone,Vector3(0,0.085,-0.22),0.29,ink)
	skull.scale = Vector3(1,0.025,0.85)
	box(zone,Vector3(0,0.086,-0.02),Vector3(0.32,0.009,0.18),ink)
	for side in [-1,1]:
		var eye := ball(zone,Vector3(side*0.105,0.096,-0.25),0.075,white)
		eye.scale.y = 0.025
		line(zone,Vector3(0,0.088,0.12),Vector3(side*0.32,0.088,0.30),0.025,ink)
		line(zone,Vector3(0,0.088,0.37),Vector3(side*0.22,0.088,0.61),0.032,ink)
		for rib in range(3):
			line(zone,Vector3(0,0.088,0.10+rib*0.075),Vector3(side*(0.22-rib*0.025),0.088,0.15+rib*0.075),0.02,ink)
	line(zone,Vector3(0,0.088,0.05),Vector3(0,0.088,0.42),0.03,ink)
	for tooth in range(4):
		box(zone,Vector3(-0.105+tooth*0.07,0.098,0.005),Vector3(0.025,0.006,0.07),white)

func sync_arms(d: Dictionary):
	if float(d.clock) < last_arm_clock:
		previous_held = -1
		release_clock = -10.0
		last_grip = Vector3(0.48,0.22,0)
		last_radius = 0.15
		arm_signature = ""
	last_arm_clock = float(d.clock)
	arms.position = robot.position
	arms.rotation.y = robot.rotation.y
	var centre := Vector3(0.48,0.22,0)
	var radius := 0.15
	var held := int(d.get("held_index",-1))
	if held >= 0:
		var object: Dictionary = d.objects[held]
		centre = arms.to_local(mapped(object.pos))
		radius = float(object.size[0])
		if object.kind == "box":
			# Find the actual side surface along the robot's gripper axis. The
			# held cube keeps its world orientation, so its projected half-width
			# changes as the robot turns.
			var held_node: Node3D = items.get_child(held)
			# Grip the near, upper side so both wrists remain visible to the lens.
			centre += Vector3(-0.10,0.06,0)
			var local_axis := held_node.global_basis.inverse() * (arms.global_basis * Vector3.FORWARD)
			var dims := Vector3(float(object.size[0]),float(object.size[2]),float(object.size[1]))
			var denominator := maxf(absf(local_axis.x)/dims.x, maxf(absf(local_axis.y)/dims.y, absf(local_axis.z)/dims.z))
			radius = 1.0/denominator if denominator > 0.0001 else dims.x
		elif object.kind == "cone":
			radius = float(object.size[0]) * 0.5
	if held >= 0:
		last_grip = centre
		last_radius = radius
	elif previous_held >= 0:
		release_clock = float(d.clock)
	var releasing := held < 0 and float(d.clock)-release_clock < 0.35
	if releasing:
		var t := clampf((float(d.clock)-release_clock)/0.35,0,1)
		centre = last_grip.lerp(centre,t)
		radius = lerpf(last_radius,0.23,t)
	previous_held = held
	var signature_now := str(held)+str(centre.snapped(Vector3.ONE*0.002))+str(radius)
	if signature_now == arm_signature: return
	arm_signature = signature_now
	for child in arms.get_children(): child.free()
	for side in [-1,1]:
		var shoulder := Vector3(0.04,0.46,side*0.21)
		var elbow := Vector3(0.30,0.29,side*0.30)
		var grip := centre+Vector3(0,0,side*(radius+0.018 if held>=0 or releasing else 0.23))
		line(arms,shoulder,elbow,0.035,black,16)
		for band in range(5):
			ball(arms,shoulder.lerp(elbow,(band+0.5)/5.0),0.037,orange if band%2==0 else black,16)
		line(arms,elbow,grip,0.027,gold,16)
		ball(arms,grip,0.038,black,16)
		for finger in [-1,1]:
			var tip := centre+Vector3(finger*0.055,0,side*(radius*0.99 if held>=0 else (radius+0.04 if releasing else 0.17)))
			if held >= 0 and d.objects[held].kind == "box":
				var node: Node3D = items.get_child(held)
				var origin := node.to_local(arms.to_global(centre+Vector3(finger*0.035,0,0)))
				var direction := node.global_basis.inverse()*(arms.global_basis*Vector3(0,0,side))
				var size: Array = d.objects[held].size
				var half := Vector3(size[0],size[2],size[1])
				var reach := 10.0
				for axis in range(3):
					if absf(direction[axis]) > 0.0001:
						reach = minf(reach,(signf(direction[axis])*half[axis]-origin[axis])/direction[axis])
				tip = centre+Vector3(finger*0.035,0,side*maxf(0,reach))
			line(arms,grip,tip,0.018,gold,16)

func build_views():
	var layout := VBoxContainer.new()
	layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(layout)
	state_label = Label.new()
	state_label.text = "  HAUNTED FETCH  /  waiting for engine"
	state_label.add_theme_font_size_override("font_size",24)
	state_label.add_theme_color_override("font_color",Color("ffad58"))
	layout.add_child(state_label)
	var row := HBoxContainer.new()
	row.size_flags_vertical = Control.SIZE_EXPAND_FILL
	layout.add_child(row)
	for key in ["rgb","overhead","cinematic"]:
		var col := VBoxContainer.new()
		col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(col)
		var title := Label.new()
		title.text = {"rgb":"01 / ROBOT FPV", "overhead":"02 / TRAJECTORY", "cinematic":"03 / HAUNTED ROOM"}[key]
		col.add_child(title)
		var vp := SubViewport.new()
		vp.size = Vector2i(640,480) if key != "cinematic" else Vector2i(1440,1080)
		if key == "overhead": vp.size = Vector2i(1012,760)
		vp.world_3d = world.get_world_3d()
		vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		vp.msaa_3d = Viewport.MSAA_2X
		add_child(vp)
		views[key] = vp
		var camera := Camera3D.new()
		vp.add_child(camera)
		camera.current = true
		camera.near = 0.004 if key == "rgb" else 0.1
		camera.far = 40
		camera.fov = 73.739795
		camera.cull_mask = 1|4|16 if key == "rgb" else (1|2|8|16 if key == "overhead" else 1|2|16)
		if key == "overhead":
			camera.projection = Camera3D.PROJECTION_ORTHOGONAL
			camera.size = 8
			camera.position = Vector3(0,10,0)
			camera.rotation_degrees.x = -90
		elif key == "cinematic":
			camera.position = Vector3(8.5,8.5,10)
			camera.look_at(Vector3(0,0.65,0))
			camera.fov = 40
		cameras[key] = camera
		var screen := TextureRect.new()
		screen.texture = vp.get_texture()
		screen.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		screen.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		screen.size_flags_vertical = Control.SIZE_EXPAND_FILL
		col.add_child(screen)

func mapped(p: Array) -> Vector3:
	return Vector3(p[0],p[2],-p[1])

func sync_scene(d: Dictionary):
	var sig := JSON.stringify([d.objects.map(func(o): return [o.kind,o.size,o.rgba]),d.obstacles])
	if sig != signature:
		signature = sig
		for n in items.get_children(): n.free()
		for n in obstacles.get_children(): n.free()
		for o in d.objects:
			var c := Color(o.rgba[0],o.rgba[1],o.rgba[2],1)
			var m := material(c,0.45,0.05)
			match o.kind:
				"sphere": ball(items,Vector3.ZERO,o.size[0],m)
				"box": box(items,Vector3.ZERO,Vector3(o.size[0],o.size[2],o.size[1])*2,m)
				"cylinder": cylinder(items,Vector3.ZERO,o.size[0],o.size[1]*2,m)
				"cone": cylinder(items,Vector3.ZERO,o.size[0],o.size[1],m,0)
		for o in d.obstacles:
			box(obstacles,Vector3(o[0],0.3,-o[1]),Vector3(o[2]*2,0.6,o[3]*2),barrier_material())
	for i in range(d.objects.size()):
		var o: Dictionary = d.objects[i]
		var n: Node3D = items.get_child(i)
		n.position = mapped(o.pos)
		var q: Array = o.quat
		n.quaternion = Quaternion(q[1],q[3],-q[2],q[0])
	robot.position = Vector3(d.robot[0],0,-d.robot[1])
	robot.rotation.y = -deg_to_rad(d.robot[2])
	zone.position = Vector3(d.zone[0],0,-d.zone[1])
	sync_arms(d)
	var f := Vector3(cos(deg_to_rad(d.robot[2])),0,sin(deg_to_rad(d.robot[2])))
	var cam: Camera3D = cameras.rgb
	cam.position = robot.position+f*0.18+Vector3(0,0.5,0)
	cam.look_at(cam.position+f*cos(deg_to_rad(10))+Vector3.DOWN*sin(deg_to_rad(10)))
	state_label.text = "  HAUNTED FETCH  /  %s  /  simulation %.2f s" % [d.state,d.clock]
	var im := ImmediateMesh.new()
	if d.path.size() > 1:
		im.surface_begin(Mesh.PRIMITIVE_LINES)
		for i in range(1,d.path.size()):
			for p in [d.path[i-1],d.path[i]]: im.surface_add_vertex(Vector3(p[0],0.1,-p[1]))
		im.surface_end()
	trail.mesh = im
	trail.material_override = glow

func _process(_delta):
	if peer == null:
		if server.is_connection_available():
			peer = server.take_connection()
			peer.set_no_delay(true)
		return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		peer = null
		incoming = ""
		return
	if busy: return
	var count := peer.get_available_bytes()
	if count > 0: incoming += peer.get_utf8_string(count)
	var end := incoming.find("\n")
	if end < 0: return
	if Time.get_ticks_usec()-last_draw_usec < 33333: return
	var packet := incoming.substr(0,end)
	incoming = incoming.substr(end+1)
	var d = JSON.parse_string(packet)
	if not d is Dictionary or d.get("token", "") != OS.get_environment("FETCH_RENDER_TOKEN"):
		peer.disconnect_from_host()
		return
	busy = true
	sync_scene(d)
	# Synchronize queued Node3D transforms before drawing this exact snapshot.
	world.force_update_transform()
	for camera in cameras.values(): camera.force_update_transform()
	for node in items.get_children(): node.force_update_transform()
	robot.force_update_transform()
	arms.force_update_transform()
	for part in arms.get_children(): part.force_update_transform()
	zone.force_update_transform()
	var now := Time.get_ticks_usec()
	last_draw_usec = now if now-last_draw_usec > 66666 else last_draw_usec+33333
	RenderingServer.force_draw(false)
	for key in ["rgb","overhead","cinematic"]:
		var img: Image = views[key].get_texture().get_image()
		img.convert(Image.FORMAT_RGB8)
		var bytes := img.get_data()
		peer.put_u32(img.get_width())
		peer.put_u32(img.get_height())
		peer.put_u32(bytes.size())
		peer.put_data(bytes)
	busy = false
