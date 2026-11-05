import math
import pytest
from agent.navigation import ground_point
from agent.perception import height_at, Detection, WorldModel, choose, parse_request
from engine.drop_access import enclosed
from scripts import planner


def project(x, height):
    a=math.radians(10);dx=x-.18;dz=height-.5
    depth=dx*math.cos(a)-dz*math.sin(a)
    up=dx*math.sin(a)+dz*math.cos(a)
    return 240-320*up/depth

@pytest.mark.parametrize('distance',[.8,1.5,3,5,7])
@pytest.mark.parametrize('height',[.22,.4,.56])
def test_distance_invariant_height(distance,height):
    base=ground_point(320,project(distance,0))
    assert base[0]==pytest.approx(distance,abs=1e-8)
    assert height_at(project(distance,height),base[0])==pytest.approx(height,abs=1e-8)


def test_clipped_height_stays_invalid_in_memory():
    d=Detection('red','cone',.2,.2,1,0,(1,0),100,height_valid=False)
    m=WorldModel();m.observe([d])
    seen=m.objects(dict(x=0,y=0,heading=0))
    assert not seen[0].height_valid
    assert choose(parse_request('tallest'),seen) is None


def ring():
    return [(0,.95,1.05,.1),(0,-.95,1.05,.1),(-.95,0,.1,1.05),(.95,0,.1,1.05)]


def test_enclosure_is_not_one_blocked_side():
    assert not enclosed((0,0),ring()[:1])
    assert not enclosed((0,0),ring()[:3])
    assert enclosed((0,0),ring())


def test_tiny_physical_opening_is_not_falsely_called_sealed():
    walls=ring()[:3]+[(.95,-.5,.1,.54),(.95,.6,.1,.54)]
    assert not enclosed((0,0),walls)


def test_no_unsafe_fallback_for_sealed_goal():
    obstacles=planner.barrier_circles(ring())
    assert planner.plan((2,0),(0,0),obstacles)==[]
    assert planner.delivery_route((2,0),(0,0),obstacles) is None

@pytest.mark.parametrize('side',[-1,1])
def test_alternate_side_release_route(side):
    walls=ring()[:2]+[(side*.95,0,.1,1.05)]
    obstacles=planner.barrier_circles(walls)
    result=planner.delivery_route((0,2.6),(0,0),obstacles)
    assert result is not None
    route,stop=result
    assert stop[0]*side<0
    pts=[(0,2.6)]+route
    assert all(not planner.blockers(a,b,obstacles) for a,b in zip(pts,pts[1:]))


def test_exact_thirty_frame_schedule(sim,monkeypatch):
    import numpy as np
    monkeypatch.setattr(sim,'_render',lambda *args,**kwargs:np.zeros((2,2,3),dtype=np.uint8))
    sim.record_video=True
    start=sim.sim_clock
    sim.turn(90)
    assert sim.sim_clock-start==pytest.approx(1)
    assert len(sim.video_frames)==30
    sim.turn(90)
    assert len(sim.video_frames)==60
    sim.record_video=False


def test_pale_marker_projection_and_partial_occlusion():
    import io
    import numpy as np
    from PIL import Image, ImageDraw
    from agent.navigation import marker_from_image
    # A 1.5 m square viewed from the origin, aligned with forward +x.
    def pixel(x,y,z):
        a=math.radians(10);dx=x-.18;dz=z-.5
        depth=dx*math.cos(a)-dz*math.sin(a)
        return (320-320*y/depth,project(x,z))
    im=Image.new('RGB',(640,480),(65,40,85));draw=ImageDraw.Draw(im)
    draw.polygon([pixel(x,y,.078) for x,y in [(2.25,-.75),(3.75,-.75),(3.75,.75),(2.25,.75)]],fill=(245,240,225))
    def encode(image):
        b=io.BytesIO();image.save(b,format='PNG');return b.getvalue()
    goal=marker_from_image(encode(im),dict(x=0,y=0,heading=0))
    assert goal is not None
    assert goal == pytest.approx((3,0),abs=.08)
    draw.rectangle((0,0,320,480),fill=(65,40,85))
    assert marker_from_image(encode(im),dict(x=0,y=0,heading=0)) is None


def test_mock_does_not_treat_dark_walls_as_a_delivery_marker():
    """An exhausted marker search must never trigger a blind move or drop."""
    import io
    from PIL import Image
    from agent.loop import _mock_decide
    buffer = io.BytesIO()
    Image.new('RGB', (640, 480), (15, 10, 20)).save(buffer, format='JPEG')
    tool, args, _ = _mock_decide(buffer.getvalue(), {'holding': True}, None, {}, 'Bring the red sphere')
    assert tool == 'finish'
    assert 'marker' in args['outcome']


def test_visible_fingertips_are_not_ranked_as_fetch_objects():
    from pathlib import Path
    from agent.perception import detect
    jpg = (Path(__file__).parent / 'fixtures/visible_grippers.png').read_bytes()
    objects = detect(jpg, {'x': 0., 'y': 0., 'heading': 315.})
    assert not any(o.colour == 'orange' for o in objects)
    assert any(o.colour == 'yellow' and o.shape == 'cone' for o in objects)
    assert any(o.colour == 'red' for o in objects)


def test_hosted_target_is_reachable_in_recorded_turn_loop():
    """Replay the failing user's camera estimates, without a paid model call."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from agent.loop import _route_obstacles
    data = json.loads((Path(__file__).parent / 'fixtures/hosted_target_obstacle.json').read_text())
    objects = [Detection(**o) for o in data['objects']]
    navigator = SimpleNamespace(prepared=True, barrier_obstacles=lambda: data['barriers'])
    state = data['pose']
    target, obstacles = _route_obstacles(navigator, objects, parse_request('Bring that red object'), None, state)
    assert target.colour == 'red'
    here = (state['x'], state['y'])
    distance = math.dist(here, target.world)
    goal = tuple(target.world[i] + (here[i]-target.world[i])*.4/distance for i in range(2))
    # Reproduce the old failure: including the fetch target seals its grasp goal.
    old = obstacles + [{'label': 'target', 'pos': target.world, '_radius': target.width/2}]
    assert planner.plan(here, goal, old) == []
    path = planner.plan(here, goal, obstacles)
    assert path
    points = [here] + path
    assert all(not planner.blockers(a, b, obstacles) for a, b in zip(points, points[1:]))
    assert all(barrier in obstacles for barrier in data['barriers'])
    assert any(o['pos'] == obj.world for obj in objects if obj.colour == 'blue' for o in obstacles)


def test_fetch_target_exclusion_keeps_neighbouring_obstacle():
    from types import SimpleNamespace
    from agent.loop import _route_obstacles
    red = Detection('red', 'sphere', .3, .3, 1, 0, (1, 0), 100)
    blue = Detection('blue', 'sphere', .3, .3, 1, 0, (1, .3), 100)
    nav = SimpleNamespace(prepared=True, barrier_obstacles=lambda: [])
    _, obstacles = _route_obstacles(nav, [red, blue], parse_request('red sphere'), None,
                                    {'x': 0, 'y': 0, 'holding': False})
    assert len(obstacles) == 1
    assert obstacles[0]['pos'] == blue.world
    assert planner.blockers((0, .3), (1, .3), obstacles)


def test_failed_replans_stop_despite_heading_oscillation():
    from agent.loop import _RouteFailureGuard
    guard = _RouteFailureGuard()
    assert not guard.failed((2, 1.2))
    assert not guard.failed((2.01, 1.2))
    assert guard.failed((2, 1.2))
    assert not guard.failed((2.3, 1.2))  # actual translation permits new evidence


def test_hosted_loop_keeps_one_owner_for_navigator_moves(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from pathlib import Path
    from agent import loop
    state = dict(x=0., y=0., heading=0., holding=False, retry_count=0, state='Locate')
    image = (Path(__file__).parent / 'fixtures/visible_grippers.png').read_bytes()
    actions = iter([('move', {'distance': .8}, 'inspect another side'),
                    ('finish', {'outcome': 'test complete'}, 'done')])
    nav = SimpleNamespace(prepared=False, barrier_points=[], phase='survey',
                          prepare=lambda *args: next(actions))
    copilot = SimpleNamespace(set_command=Mock(), observe=Mock(), note_result=Mock(),
                             safe_action=Mock(return_value=('turn', {'degrees':45}, 'competing route')))
    posts = []
    monkeypatch.setattr(loop, 'ReturnNavigator', lambda: nav)
    monkeypatch.setattr(loop, 'Copilot', lambda: copilot)
    from agent import perception
    monkeypatch.setattr(perception, 'detect', lambda *args: [Detection('red', 'sphere', .3, .3, 2, 0, (2, 0), 500)])
    monkeypatch.setattr(loop, 'RUNS', tmp_path)
    monkeypatch.setattr(loop, 'build_vlm', lambda *args, **kwargs: SimpleNamespace(name='offline', model='offline'))
    for name in ('wait_for_engine', 'load_env', '_claim', '_release'):
        monkeypatch.setattr(loop, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(loop, '_read_spend', lambda: 0)
    monkeypatch.setattr(loop, '_commit_spend', lambda amount: 0)
    monkeypatch.setattr(loop, '_get', lambda path: {'/look': image, '/state': state,
                       '/depth': {'distance': 3}, '/admin/outcome': {'result': ''}}[path])
    def post(path, body=None):
        posts.append((path, body))
        return {'collided': False, 'distance_travelled': .8, 'state': 'GoToObject', 'retry_count': 0}
    monkeypatch.setattr(loop, '_post', post)
    loop.run('Bring that red object', None, False, 12, False, provider='anthropic')
    copilot.safe_action.assert_not_called()
    assert ('/move', {'distance': .8}) in posts
    assert [body for path, body in posts if path == '/turn'] == [{'degrees': 45.0}] * 8


def test_successful_grasp_removes_only_carried_floor_track(monkeypatch):
    from agent.copilot import Copilot
    from agent import perception
    c = Copilot()
    c.set_command('red sphere')
    red = Detection('red', 'sphere', .3, .34, .37, 0, (.37, 0), 100)
    blue = Detection('blue', 'sphere', .3, .3, .5, 0, (.37, .3), 100)
    c.world.observe([red, blue])
    c._observed_pose = dict(x=0, y=0, heading=0, holding=False)
    c.note_result('pickup', {'success': True})
    assert [t['colour'] for t in c.world.tracks] == ['blue']
    monkeypatch.setattr(perception, 'detect', lambda *args: [red, blue])
    from agent import barriers
    monkeypatch.setattr(barriers, 'observe_barriers', lambda *args: [])
    c.observe(b'', dict(x=0, y=0, heading=0, holding=True))
    assert [t['colour'] for t in c.world.tracks] == ['blue']
    c.note_result('drop', {'success': True})
    assert c._held_target is None
