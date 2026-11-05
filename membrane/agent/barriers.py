"""Obstacle samples inferred exclusively from camera pixels and own odometry.
No simulator imports, masks or scene metadata cross the engine boundary.
"""
import io, math
import cv2
import numpy as np
from PIL import Image
from .navigation import ground_point


def observe_barriers(jpg, pose):
    a = np.asarray(Image.open(io.BytesIO(jpg)).convert('RGB')).astype(float)
    hi, lo = a.max(axis=2), a.min(axis=2)
    mask = ((hi-lo < 4) & (hi > 145) & (hi < 165)).astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    result=[]
    angle=math.radians(pose['heading'])
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area < 200 or w < 18 or h < 8 or y < 2:
            continue
        for u in range(x+2,min(x+w-2,638),6):
            rows=np.flatnonzero(lab[:,u]==i)
            if len(rows)<8 or rows[-1]>=477:
                continue
            base=ground_point(u,float(rows[-1]))
            if base is None or not .25<base[0]<8:
                continue
            # Reject floor/grout, slab, highlights and high wall decorations.
            q=(float(rows[0])-240)/320
            s,c=math.sin(math.radians(10)),math.cos(math.radians(10))
            height=.5-(base[0]-.18)*(s+q*c)/(c-q*s)
            if not .32 < height < .95:
                continue
            f,r=base
            result.append((pose['x']+f*math.cos(angle)-r*math.sin(angle),
                           pose['y']-f*math.sin(angle)-r*math.cos(angle)))
    return result
