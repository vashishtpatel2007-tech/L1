import pytest
from agent.perception import Detection, parse_request, COLOUR_ANCHORS
from agent.selection import resolve


def obj(colour='purple', shape='cone', height=.3, distance=2, x=1, valid=True):
    return Detection(colour, shape, height, height*.6, distance, 0, (x, 1), 500, height_valid=valid)


@pytest.mark.parametrize('colour', list(COLOUR_ANCHORS))
@pytest.mark.parametrize('shape', ['cone', 'sphere', 'box', 'cylinder'])
def test_duplicate_matches_need_clarification_for_every_scene_colour(colour, shape):
    result = resolve(parse_request(f'Bring that {colour} {shape}'),
                     [obj(colour, shape, .3, x=1), obj(colour, shape, .7, x=3)])
    assert result.target is None
    assert result.code == 'ambiguous'
    assert '2 matching' in result.message
    assert len(result.candidates) == 2


@pytest.mark.parametrize('command,expected', [('tallest object', .7), ('shortest object', .3),
    ('shorted obj', .3), ('big purple cone', .7), ('small purple cone', .3),
    ('taller purple cone', .7), ('smaller purple cone', .3)])
def test_comparisons_use_metric_size_not_distance(command, expected):
    result = resolve(parse_request(command), [obj(height=.3, distance=1), obj(height=.7, distance=5, x=3)])
    assert result.target.height == expected


def test_round_red_and_tall_blue_and_purple_thing():
    red_ball = obj('red', 'sphere'); red_box = obj('red', 'box', x=3)
    assert resolve(parse_request('that round red object'), [red_ball, red_box]).target is red_ball
    tall_blue = obj('blue', 'cone', .7, 5)
    assert resolve(parse_request('tall blue object'), [tall_blue, obj('blue', height=.3)]).target is tall_blue
    purple = obj()
    assert resolve(parse_request('that purple thing'), [red_ball, purple]).target is purple


def test_colour_only_ambiguity_does_not_silently_pick_nearest():
    assert resolve(parse_request('red thing'), [obj('red', distance=1), obj('red', 'sphere', distance=4)]).code == 'ambiguous'


def test_tied_or_clipped_comparisons_do_not_guess():
    assert resolve(parse_request('tallest'), [obj(height=.4), obj(height=.405, x=3)]).code == 'ambiguous'
    assert resolve(parse_request('shortest'), [obj(height=.3), obj(height=.7, valid=False)]).code == 'measurement_uncertain'


def test_wrong_shape_missing_colour_or_unspecified_are_not_substituted():
    assert resolve(parse_request('purple cone'), [obj('purple', 'box')]).target is None
    assert resolve(parse_request('blue sphere'), [obj('red', 'sphere')]).target is None
    assert resolve(parse_request('fetch that thing'), [obj()]).code == 'more_specific'


@pytest.mark.parametrize('filename,heading,height', [('purple_cone_lighting.png',315,.30), ('purple_cone_wall.png',135,.56)])
def test_purple_cones_remain_separate_from_room_lighting(filename, heading, height):
    from pathlib import Path
    from agent.perception import detect
    seen = detect((Path(__file__).parent/'fixtures'/filename).read_bytes(), dict(x=0,y=0,heading=heading))
    cones = [o for o in seen if o.colour == 'purple' and o.shape == 'cone']
    assert len(cones) == 1
    assert cones[0].height == pytest.approx(height, abs=.04)


@pytest.mark.parametrize('mock', [True, False])
def test_ambiguous_run_scans_then_stops_without_translation_or_model_call(monkeypatch, tmp_path, mock):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent import loop, perception
    image = (Path(__file__).parent/'fixtures/purple_cone_lighting.png').read_bytes()
    state = dict(x=0., y=0., heading=0., holding=False, retry_count=0, state='Locate')
    model = SimpleNamespace(name='offline', model='offline', decide=Mock(side_effect=AssertionError('No model call permitted')))
    nav = SimpleNamespace(phase='scan', prepare=Mock(side_effect=AssertionError('No navigation before clarification')))
    copilot = SimpleNamespace(set_command=Mock(), observe=Mock(), note_result=Mock(), safe_action=Mock())
    monkeypatch.setattr(perception, 'detect', lambda *a: [obj(height=.3, x=1), obj(height=.7, x=-2)])
    monkeypatch.setattr(loop, 'ReturnNavigator', lambda: nav)
    monkeypatch.setattr(loop, 'Copilot', lambda: copilot)
    monkeypatch.setattr(loop, 'build_vlm', lambda *a, **kw: model)
    monkeypatch.setattr(loop, 'RUNS', tmp_path)
    for name in ('wait_for_engine', 'load_env', '_claim', '_release'):
        monkeypatch.setattr(loop, name, lambda *a, **kw: None)
    monkeypatch.setattr(loop, '_read_spend', lambda: 0)
    monkeypatch.setattr(loop, '_commit_spend', lambda amount: 0)
    monkeypatch.setattr(loop, '_get', lambda path: {'/look': image, '/state': state,
                        '/depth': {'distance': 3}, '/admin/outcome': {'result': ''}}[path])
    posts = []
    def post(path, body=None):
        posts.append((path, body))
        if path == '/turn': state['heading'] = (state['heading'] + body['degrees']) % 360
        return {'heading': state['heading'], 'state': 'Locate'}
    monkeypatch.setattr(loop, '_post', post)
    result = loop.run('Bring that purple cone', None, mock, 60, False, provider='mock' if mock else 'anthropic')
    assert result['reason'] == 'clarification_required'
    assert len(result['clarification']['candidates']) == 2
    assert [p for p, body in posts if p != '/admin/reset'] == ['/turn'] * 8
    assert state['heading'] == 0
    model.decide.assert_not_called()
    nav.prepare.assert_not_called()
