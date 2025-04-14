import math

def point_in_polygon(point, pg):
    """
    Check if point is inside polygon

    Args:
        point (tuple): (x, y) point to check
        pg (list of tuples): list of (x, y) vertices of polygon

    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    inside = False
    n = len(pg)
    for i in range(n):
        xi, yi = pg[i]
        xj, yj = pg[(i + 1) % n]
        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
    return inside

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """
    Returns the shortest distance from point (px, py) to the line segment (x1, y1)-(x2, y2)

    Args:
        px (float): x coordinate of point
        py (float): y coordinate of point
        x1 (float): x coordinate of first point
        y1 (float): y coordinate of first point
        x2 (float): x coordinate of second point
        y2 (float): y coordinate of second point

    Returns:
        float: distance from point (px, py) to line segment (x1, y1)-(x2, y2)
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        # Line segment is a point
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)

def circle_intersect(center, radius, polygon):
    """
    Checks if a circle is intersecting with polygon

    Args:
        center (tuple): (x, y) center of circle
        radius (float): radius of the circle
        polygon (list of tuples): list of (x, y) vertices of polygon

    Returns:
        bool: True if circle is inside polygon, False otherwise
    """
    cx, cy = center

    # 1. If the circle's center is inside the polygon, it intersects
    if point_in_polygon(center, polygon):
        return True

    # 2. Check if circle intersects any polygon edge
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        dist = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if dist <= radius:
            return True

    return False

# Example usage:
poly = [(0, 0), (10, 0), (10, 10), (5, 15), (0,10)]
circle_center = (12, 5)
circle_radius = 2

if circle_intersect(circle_center, circle_radius, poly):
    print("Circle Intersection found")
else:
    print("No Intersection found")
