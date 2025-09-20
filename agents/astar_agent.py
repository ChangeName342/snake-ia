from env.snake_env import DIRECTIONS
from heapq import heappush, heappop

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_path(env):
    start = env.snake[0]
    goal = env.apple
    if goal is None:
        return None
    blocked = set(env.snake)
    open_set = []
    heappush(open_set, (manhattan(start, goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    visited = set()
    while open_set:
        f, g, current, parent = heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal:
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        cx, cy = current
        for d in DIRECTIONS:
            nx, ny = cx + d.dx, cy + d.dy
            neighbor = (nx, ny)
            if not (0 <= nx < env.size and 0 <= ny < env.size):
                continue
            tail = env.snake[-1]
            if neighbor in blocked and neighbor != tail:
                continue
            tentative_g = g + 1
            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                gscore[neighbor] = tentative_g
                heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor, current))
    return None

def path_to_direction(path):
    if path is None or len(path)<2:
        return None
    (x0,y0), (x1,y1) = path[0], path[1]
    dx, dy = x1-x0, y1-y0
    for d in DIRECTIONS:
        if d.dx==dx and d.dy==dy:
            return d
    return None

def expert_agent(obs, env):
    path = astar_path(env)
    d = path_to_direction(path)
    if d is None:
        acts = env.available_actions()
        d = acts[0] if acts else None
    return d