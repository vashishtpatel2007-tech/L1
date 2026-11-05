"""Engine-only topological enclosure check for the operator error overlay.
Not a sensor or navigation input. No coordinates are returned to the agent.
Axis-aligned rectangle arrangement is exact, including very narrow openings;
it does not mistake conservative planner inflation for a sealed enclosure.
"""
from collections import deque


def enclosed(goal, barriers, room_half=4.0):
    if not barriers:
        return False
    xg,yg=goal
    if any(abs(xg-x)<=hx and abs(yg-y)<=hy for x,y,hx,hy in barriers):
        return False  # occupied centre is a different condition
    xs=sorted(set([-room_half,room_half]+[max(-room_half,min(room_half,v)) for x,y,hx,hy in barriers for v in (x-hx,x+hx)]))
    ys=sorted(set([-room_half,room_half]+[max(-room_half,min(room_half,v)) for x,y,hx,hy in barriers for v in (y-hy,y+hy)]))
    nx,ny=len(xs)-1,len(ys)-1
    free=set()
    for i in range(nx):
        for j in range(ny):
            x,y=(xs[i]+xs[i+1])/2,(ys[j]+ys[j+1])/2
            if not any(abs(x-cx)<hx and abs(y-cy)<hy for cx,cy,hx,hy in barriers):
                free.add((i,j))
    starts=[(i,j) for i,j in free if xs[i]<=xg<=xs[i+1] and ys[j]<=yg<=ys[j+1]]
    todo=deque(starts);seen=set(starts)
    while todo:
        i,j=todo.popleft()
        if i in (0,nx-1) or j in (0,ny-1):
            return False
        for p in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if p in free and p not in seen:
                seen.add(p);todo.append(p)
    return bool(starts)
