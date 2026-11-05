"""Read-only renderer verification. Run while the robot is idle."""
import io, os, time
import httpx
from PIL import Image
base = os.environ.get('MEMBRANE_ENGINE', 'http://127.0.0.1:8080')
with httpx.Client(base_url=base, timeout=60) as c:
    state = c.get('/state').json()
    assert set(state) == {'state','x','y','heading','holding','retry_count'}
    assert set(c.get('/depth').json()) == {'distance'}
    for path, shape in [('/look',(640,480)),('/admin/overhead',(1012,760)),('/admin/cinematic',(1440,1080))]:
        r=c.get(path);r.raise_for_status()
        image=Image.open(io.BytesIO(r.content))
        assert image.size == shape, (path,image.size)
        print(path, image.size, 'OK')
    start=time.perf_counter()
    for _ in range(15): c.get('/look').raise_for_status()
    print('FPV HTTP responses/sec:',round(15/(time.perf_counter()-start),1))
    assert c.get('/state').json() == state, 'State changed during read-only check.'
    print('Renderer contract verified; agent success requires separate task runs.')
