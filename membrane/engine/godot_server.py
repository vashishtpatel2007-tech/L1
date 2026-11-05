"""Halloween renderer entry point. MuJoCo remains the simulation authority."""
import io
from pathlib import Path
from PIL import Image
from fastapi.responses import Response, StreamingResponse
import asyncio, threading, time
from .sim import Sim
from .godot_renderer import GodotRenderer

_renderer = None

def _godot_render(self, camera='rgb'):
    global _renderer
    if _renderer is None or _renderer.proc.poll() is not None:
        _renderer = GodotRenderer()
    try:
        return _renderer.render(self, camera)
    except (OSError, RuntimeError):
        _renderer.close()
        _renderer = GodotRenderer()
        return _renderer.render(self, camera)

# Replace only the rendering method before the existing server constructs Sim.
Sim._render = _godot_render
Sim.live_render = True
from .server import app, worker  # noqa: E402

@app.get('/admin/cinematic')
def cinematic():
    arr = worker.call(lambda w: w.sim._render('cinematic'))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=90)
    return Response(buf.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

@app.get('/admin/overhead')
def overhead():
    arr = worker.call(lambda w: w.sim._render('overhead'))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=90)
    return Response(buf.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

# Override the presentation route without changing any existing control routes.
for route in app.routes:
    if getattr(route, 'path', None) == '/':
        app.routes.remove(route)
        break

@app.get('/', include_in_schema=False)
def haunted_console():
    html = (Path(__file__).parent / 'console.html').read_text()
    css = '''<style>
:root{color-scheme:light;--bg:#fff;--panel:#fff;--panel2:#f6f6f6;--line:#d5d5d5;--txt:#111;--muted:#333;--accent:#111;--good:#111;--warn:#111;--bad:#111}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
header{background:#fff;color:#111}h1{font-size:17px;font-weight:650}
.panel h2{font-size:12px;text-transform:none;letter-spacing:.02em;font-weight:650;color:#111}
.sub{font-size:12px;color:#333}
button,button.go,button.warn{background:#fff;color:#111;border:1px solid #777;border-radius:7px;font-weight:600}
button.go{border:2px solid #111}button:hover:not(:disabled){background:#eee;color:#000;border-color:#111}
button:disabled{opacity:.45}input,select{background:#fff;color:#111;border-color:#999;border-radius:7px}
input::placeholder{color:#555}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #111;outline-offset:3px}
.chip,.stat{background:#f6f6f6;color:#111}.t-turn,.t-move,.t-pickup,.t-drop,.t-err,.t-sys{color:#111}.t-err{font-weight:700}
.log,.term{background:#fafafa;color:#111;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.term .cmd,.term .ok,.term .no,.term .dim{color:#111}.term .blk{border-color:#ddd}
img,canvas{background:#fff}
main{max-width:none;width:100%;gap:8px;padding:8px;grid-template-columns:repeat(12,minmax(0,1fr))}
.panel{min-width:0;margin:0;padding:10px;border-radius:10px;grid-column:span 6}.panel.wide{grid-column:1/-1}
.panel:has(#pov){grid-row:2;grid-column:1/8}.panel:has(#map){grid-row:3;grid-column:1/6}.cinematic{grid-row:3;grid-column:6/13}.panel:has(#alog){grid-row:2;grid-column:8/13;display:flex;flex-direction:column;min-height:0}
.panel:has(#pov),.panel:has(#map),.cinematic{position:relative;display:flex;flex-direction:column;align-self:start;padding:0;overflow:hidden}
.panel:has(#pov) h2,.panel:has(#map) h2,.cinematic h2{height:54px;flex-shrink:0;margin:0;padding:10px 12px;display:flex;align-items:center}
.panel:has(#pov) h2,.cinematic h2{padding-right:110px}
.panel:has(#map)>.sub{position:absolute;top:36px;left:12px;font-size:9px}
.panel:has(#map) h2{align-items:center;font-size:12px}
#pov{width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;border-radius:0;background:var(--panel)}
#cinematic{width:100%;height:clamp(300px,48vh,580px);object-fit:contain;border-radius:0;background:var(--panel)}
.panel:has(#map) #map{width:100%;height:clamp(240px,38vh,420px);object-fit:contain;align-self:center;border-radius:0}
.panel:has(#pov) .stats,.panel:has(#map) .stats{margin:0;padding:8px;gap:4px}.stat{padding:5px 6px}.stat .v{font-size:13px}
.cinematic>.sub{margin:0;padding:12px}
.panel:has(#alog) .log{height:0;min-height:160px;flex:1;overflow:auto}
.view-toggle{position:absolute;top:10px;right:8px;z-index:2;font-size:11px;padding:6px 8px}
.panel:fullscreen,.panel.expanded{background:#fff;padding:12px;width:100vw;height:100vh;box-sizing:border-box}.panel.expanded{position:fixed;inset:0;z-index:90}
.panel:fullscreen #pov,.panel.expanded #pov,.panel:fullscreen #cinematic,.panel.expanded #cinematic{width:100%;height:calc(100vh - 140px);object-fit:contain;aspect-ratio:auto}
@media(max-width:600px){main{grid-template-columns:1fr}.panel,.panel:has(#pov),.panel:has(#map),.panel:has(#alog),.cinematic{grid-row:auto;grid-column:1/-1}}
label:has(#seed),#clutter{display:none}
#allViews[hidden]{display:none}
body.all-views-open{overflow:hidden}
#allViews{position:fixed;inset:0;z-index:95;background:#fff;padding:8px;box-sizing:border-box;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:44px minmax(0,1fr) minmax(100px,22vh);gap:8px}
#allViews .all-views-toolbar{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between}
#allViews .panel{grid-area:auto;min-width:0;min-height:0;height:100%;align-self:stretch;box-sizing:border-box;overflow:hidden}
#allViews .panel:has(#alog){grid-column:1/-1;display:flex;flex-direction:column}
#allViews .panel:has(#alog) .log{height:auto;min-height:0;flex:1;overflow:auto}
#allViews #pov,#allViews #cinematic,#allViews #map{width:100%;height:0;min-height:0;flex:1 1 0;object-fit:contain;aspect-ratio:auto;max-width:100%}
#allViews .view-toggle{display:none}
#allViews .panel h2{padding:10px;font-size:12px;height:44px;flex-shrink:0}
#allViews .stats,#allViews .sub{flex-shrink:0}
#allViewsButton{white-space:nowrap;font-size:12px}
#selectionIssue{width:min(580px,85vw);background:#fff;color:#111;border:2px solid #111;border-radius:12px;padding:28px;box-sizing:border-box;line-height:1.6}
#selectionIssue::backdrop{background:#000b}
#blockedDrop{background:#fff!important;color:#111!important;border:2px solid #111!important;border-radius:12px!important}
#blockedDrop *{color:#111!important}
</style>'''
    html = html.replace('</head>', css+'</head>')
    import re
    html = re.sub(r'<p[^>]*>\s*Randomising always keeps the rules:.*?</p>', '', html, flags=re.S)
    html = html.replace('value="32"','value="60"')
    html = html.replace('if(locked || agentTimer){ refreshPov();', '{ refreshPov();')
    html = html.replace("function refreshPov(){ $('#pov').src = '/look?t='+Date.now(); }", "function refreshPov(){ if(!$('#pov').src.includes('/admin/live/rgb')) $('#pov').src='/admin/live/rgb'; }")
    html = html.replace('if(p.trace && p.trace.length > 1){', 'if(p.trace){')
    html = html.replace('Membrane Fetch — Operator Console', 'Haunted Fetch — Mission Control')
    html = html.replace('Robot camera &nbsp;·&nbsp; GET /look &nbsp;·&nbsp; 640×480, 90° FOV', 'Robot camera')
    html = html.replace('Scene &amp; trajectory &nbsp;·&nbsp; operator view, never sent to the model', 'Top view &amp; trajectory')
    html = html.replace('Log &nbsp;·&nbsp; every step and the model\'s stated reason', 'Activity log')
    html = html.replace('const f = s.final||{}, eo = f.engine_outcome||{};',
                        'const f = s.final||{}, eo = f.engine_outcome||{}; if(f.clarification) window.showSelectionIssue(f.clarification);')
    marker = '  <!-- ─────────── scene setup ─────────── -->'
    panel = '''<section class="panel cinematic"><h2>Haunted room · live observer</h2>
<img id="cinematic" alt="Live Godot Halloween room">
<p class="sub">Cinematic observer · same simulation · never sent to the agent</p></section>'''
    panel += '<section class="panel wide"><h2>Barrier test</h2><select id="barrierMode"><option value="clear">Clear drop location</option><option value="front">Horizontal front barrier</option><option value="left_open">Left approach open</option><option value="right_open">Right approach open</option><option value="sealed">Fully enclosed drop location</option></select><button id="applyBarrier">Load scenario</button><span class="sub"> 30 FPS renderer · 250 Hz physics</span></section>'
    html = html.replace(marker, panel+marker)
    html = html.replace("g.fillStyle = '#eef1f7'; g.fillRect(0,0,W,H);", "g.fillStyle = '#211a2c'; g.fillRect(0,0,W,H); if(window.hauntedTop && window.hauntedTop.complete && window.hauntedTop.naturalWidth) g.drawImage(window.hauntedTop,0,0,W,H);")
    html = html.replace("g.fillStyle = '#333'", "g.fillStyle = '#f4dfc5'")
    html = html.replace('<p class="sub" style="margin:9px 0 0">Randomising', '<p hidden class="sub" style="margin:9px 0 0">Randomising')
    html = html.replace('</body>',  '''<script>
const blocked=document.createElement('dialog');blocked.id='blockedDrop';blocked.style.cssText='width:75vw;height:75vh;box-sizing:border-box;background:radial-gradient(ellipse at top,#63131f,#1d050b);color:#fff1f2;border:1px solid #ef5364;border-radius:24px;box-shadow:0 30px 100px #000b;text-align:center;padding:9vh 5vw;z-index:100';blocked.innerHTML='<div style="font:60px system-ui;color:#ff6679">⚠</div><p style="letter-spacing:4px;color:#ff8f9c">DELIVERY UNAVAILABLE</p><h1 style="font-size:clamp(28px,4vw,60px)">ERROR — DROP LOCATION BLOCKED</h1><p style="font-size:20px">The drop location is enclosed on all sides. Cannot place the object.</p><p>Open an approach and reset the scenario to continue.</p><button id="dismissBlocked">Review scene</button>';document.body.appendChild(blocked);
let dismissedBlock=false;document.getElementById('dismissBlocked').onclick=()=>{dismissedBlock=true;blocked.close()};
async function accessPoll(){try{const v=await jget('/admin/drop-access');if(v.enclosed&&!dismissedBlock&&!blocked.open)blocked.showModal();if(!v.enclosed){dismissedBlock=false;if(blocked.open)blocked.close()}}catch(e){}setTimeout(accessPoll,1000)}accessPoll();
document.getElementById('applyBarrier').onclick=async()=>{try{await jpost('/admin/barrier-scenario',{mode:document.getElementById('barrierMode').value});dismissedBlock=false;trail=[];refreshPov();await refreshLayout();await refreshState()}catch(e){alog(String(e),'t-err')}};
window.hauntedTop=new Image();
(function(){const top=window.hauntedTop;top.src='/admin/live/overhead';function paint(){if(!document.hidden&&top.naturalWidth)drawMap();requestAnimationFrame(paint)}requestAnimationFrame(paint);
document.getElementById('map').width=1012;document.getElementById('map').height=760;drawMap=function(){const c=document.getElementById('map'),g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,c.width,c.height);if(top.naturalWidth)g.drawImage(top,0,0,c.width,c.height)};})();
document.getElementById('cinematic').src='/admin/live/cinematic';
for(const id of ['pov','cinematic']){
 const panel=document.getElementById(id).closest('section');
 const button=document.createElement('button');button.className='view-toggle';button.textContent='⛶ Full screen';panel.prepend(button);
 const restore=()=>{panel.classList.remove('expanded');button.textContent='⛶ Full screen'};
 button.onclick=async()=>{if(document.fullscreenElement===panel){await document.exitFullscreen();restore()}else if(panel.classList.contains('expanded'))restore();else{try{if(!panel.requestFullscreen)throw Error();await panel.requestFullscreen();button.textContent='↙ Exit full screen'}catch(e){panel.classList.add('expanded');button.textContent='↙ Back to all views'}}};
 document.addEventListener('fullscreenchange',()=>{if(document.fullscreenElement!==panel)restore()});
 document.addEventListener('keydown',async e=>{if(e.key==='Escape'){if(document.fullscreenElement===panel)await document.exitFullscreen();restore()}});
}

</script></body>''')
    html = html.replace('</body>', '<script>'+ (Path(__file__).parent / 'voice_input.js').read_text() + '</script></body>')
    html = html.replace('</body>', '<script>'+ (Path(__file__).parent / 'all_views.js').read_text() + '</script></body>')
    return Response(html, media_type='text/html')

# Diagnostic-only verdict. It is deliberately absent from the agent tool schema.
@app.get('/admin/drop-access')
def drop_access():
    from .drop_access import enclosed
    return worker.call(lambda w: {'enclosed': enclosed(w.sim.scene.drop_zone,w.sim.scene.obstacles),
                                  'message':'Drop location blocked', 'fps_target':30})

from pydantic import BaseModel
from typing import Literal
from fastapi import Depends
from .server import guard

class BarrierScenario(BaseModel):
    mode: Literal['clear','front','left_open','right_open','sealed'] = 'clear'

@app.post('/admin/barrier-scenario')
def barrier_scenario(body: BarrierScenario, _: None = Depends(guard)):
    from .world import default_scene
    def change(w):
        scene=default_scene()
        # Centre the pad so all four sides can be tested independently of room walls.
        scene.drop_zone=(0.,-1.8)
        x,y=scene.drop_zone
        front=(x,y+.95,1.05,.10)
        back=(x,y-.95,1.05,.10)
        left=(x-.95,y,.10,1.05)
        right=(x+.95,y,.10,1.05)
        scene.obstacles={'clear':[], 'front':[front], 'left_open':[front,back,right],
                         'right_open':[front,back,left], 'sealed':[front,back,left,right]}[body.mode]
        w.sim.close()
        w.sim=Sim(scene=scene,record_video=False,realtime=w.realtime)
        w.obstacles=bool(scene.obstacles)
        w.clutter='none';w.seed=None
        return {'ok':True,'mode':body.mode}
    return worker.call(change)


@app.get('/admin/live/{view}')
async def live_view(view: Literal['rgb','overhead','cinematic']):
    async def frames():
        while True:
            renderer=_renderer
            frame=renderer.jpeg_frames.get(view) if renderer else None
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(frame)).encode()+b'\r\n\r\n'+frame+b'\r\n'
            await asyncio.sleep(1/30)
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})

_stream_stop=threading.Event()
def _idle_frames():
    while not _stream_stop.is_set():
        start=time.monotonic()
        try:
            worker.call(lambda w:w.sim._render('rgb'))
        except Exception:
            _stream_stop.wait(.25)
        _stream_stop.wait(max(0,1/30-(time.monotonic()-start)))

@app.on_event('startup')
def start_live_frames():
    def set_realtime(w):
        w.realtime=1.0
        w.sim.realtime=1.0
    worker.call(set_realtime)
    threading.Thread(target=_idle_frames,name='godot-live-frames',daemon=True).start()

@app.on_event('shutdown')
def stop_live_frames():
    _stream_stop.set()
    # Serialize cleanup with the worker so no in-flight draw uses a closed
    # socket, and server restarts do not leave GPU renderer processes behind.
    if _renderer is not None:
        worker.call(lambda w: _renderer.close())

@app.get('/admin/render-performance')
def render_performance():
    r=_renderer
    if r is None or not r.generation:return {'ready':False}
    return {'ready':True,'target_fps':30,'snapshots':r.generation,
            'mean_render_ms':round(1000*r.render_seconds/r.generation,2),
            'effective_snapshot_fps':round(r.generation/max(.001,time.perf_counter()-r.started_at),2)}
