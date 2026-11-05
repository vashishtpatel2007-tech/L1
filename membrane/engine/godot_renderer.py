"""Engine-private Godot rendering transport. Never imported by the agent.
MuJoCo remains authoritative for every transform, collision and action.
"""
from __future__ import annotations
import atexit, io, json, os, socket, struct, subprocess, time, threading
from PIL import Image
from pathlib import Path
import numpy as np

class GodotRenderer:
    def __init__(self):
        self.frames = {}
        self._last_snapshot = None
        self.jpeg_frames = {}
        self.generation = 0
        self.render_seconds = 0.0
        self.started_at = time.perf_counter()
        self._encode_stop = threading.Event()
        self._encode_ready = threading.Event()
        self._encode_thread = threading.Thread(target=self._encode,daemon=True)
        self._encode_thread.start()
        root = Path(__file__).resolve().parents[1]
        executable = os.environ.get('GODOT_BIN', str(root / '.tools/Godot.app/Contents/MacOS/Godot'))
        if not Path(executable).is_file():
            raise RuntimeError('Godot missing. Set GODOT_BIN to your Godot 4 executable.')
        # An ephemeral loopback port and per-process token keep the private scene
        # transport separate from the public agent HTTP interface.
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        import secrets
        self.token = secrets.token_hex(24)
        env = dict(os.environ, FETCH_RENDER_PORT=str(port), FETCH_RENDER_TOKEN=self.token)
        self.log = open(root / '.tools/godot-renderer.log', 'w')
        self.proc = subprocess.Popen([executable, '--disable-render-loop', '--path', str(root / 'godot')], env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        for _ in range(200):
            if self.proc.poll() is not None:
                raise RuntimeError('Godot exited; see membrane/.tools/godot-renderer.log')
            try:
                self.sock = socket.create_connection(('127.0.0.1', port), timeout=1)
                self.sock.settimeout(30)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                break
            except OSError:
                time.sleep(.1)
        else:
            self.close()
            raise RuntimeError('Godot renderer startup timed out')

    def _read(self, n):
        result = bytearray()
        while len(result) < n:
            chunk = self.sock.recv(n - len(result))
            if not chunk:
                raise RuntimeError('Godot rendering connection closed')
            result.extend(chunk)
        return bytes(result)

    def render(self, sim, camera='rgb'):
        render_start=time.perf_counter()
        objects = []
        for o in sim.scene.objects:
            bid = sim.obj_bids[o.name]
            objects.append(dict(kind=o.kind, size=o.size, rgba=o.rgba,
                                pos=sim.data.xpos[bid].tolist(),
                                quat=sim.data.xquat[bid].tolist()))
        packet = dict(token=self.token, camera='all', robot=[sim.x, sim.y, sim.heading],
                      objects=objects, zone=sim.scene.drop_zone,
                      held_index=next((i for i,o in enumerate(sim.scene.objects) if o.name==sim.holding),-1),
                      obstacles=sim.scene.obstacles, clock=sim.sim_clock,
                      path=sim.path_trace, state=sim.state)
        payload=json.dumps(packet,separators=(',', ':'))
        if payload == self._last_snapshot and self.frames:
            return self.frames[camera]
        self.sock.sendall((payload+'\n').encode())
        self._last_snapshot=payload
        frames={}
        for key in ('rgb','overhead','cinematic'):
            width,height,size=struct.unpack('<III',self._read(12))
            if not (0<width<=1920 and 0<height<=1080 and size==width*height*3):
                raise RuntimeError('Invalid Godot frame dimensions')
            frames[key]=np.frombuffer(self._read(size),dtype=np.uint8).reshape(height,width,3).copy()
        self.frames=frames
        self.generation += 1
        self.render_seconds += time.perf_counter()-render_start
        self._encode_ready.set()
        return frames[camera]

    def _encode(self):
        while not self._encode_stop.is_set():
            self._encode_ready.wait(.2)
            self._encode_ready.clear()
            snapshot=self.frames
            if not snapshot:
                continue
            encoded={}
            for key,arr in snapshot.items():
                buf=io.BytesIO()
                Image.fromarray(arr).save(buf,format='JPEG',quality=85)
                encoded[key]=buf.getvalue()
            self.jpeg_frames=encoded # atomic publication of one coherent generation

    def close(self):
        self._encode_stop.set()
        self._encode_ready.set()
        if hasattr(self, 'sock'):
            self.sock.close()
        if hasattr(self, 'proc') and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if hasattr(self, 'log'):
            self.log.close()
