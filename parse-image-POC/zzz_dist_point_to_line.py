import numpy as np
import numpy.linalg as la


def distance_to_line(a, b, point):
    """ 
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    To fine the distance from line segment AB and point C:
        Form a parallelogram with vectors AC and AB.
        Calculate the area of the parallelogram.
        Then the distance is equal to the area divided by the length of base.
    """
    a, b, point = np.array(a), np.array(b), np.array(point)
    v_base = b - a
    v_side = point - a
    area = la.norm(np.cross(v_base, v_side))
    base_length = la.norm(v_base)
    return area / base_length

# -----------------------------------------------------------

from collections import namedtuple
from math import sqrt

Point = namedtuple('Point', ['x', 'y'])
Line = namedtuple('Line', ['m', 'b'])
LineSegment = namedtuple('LineSegment', ['p1', 'p2'])

def get_line(line_seg):
    p1, p2 = line_seg
    if p1.x == p2.x:
        return Line(None, None)
    else:
        m = (p2.y - p1.y) / float(p2.x - p1.x)
        b = p1.y - m * p1.x
        return Line(m, b)

def distance(p1, p2):
    return sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def intersection(line1, line2):
    if line1.m == line2.m:
        return None
    x = (line2.b - line1.b) / (line1.m - line2.m)
    y = line1.m * x + line1.b
    return Point(x, y)

def is_on_line_segment(line_seg, p):
    p1, p2 = line_seg
    min_x, max_x = min(p1.x, p2.x), max(p1.x, p2.x)
    min_y, max_y = min(p1.y, p2.y), max(p1.y, p2.y)
    return min_x <= p.x <= max_x and min_y <= p.y <= max_y

def distance_to_line_v2(line_seg, target_point):
    p1, p2 = line_seg
    tp = target_point
    line = get_line(line_seg)

    if p1 == p2:
        return distance(p1, tp)

    if line.m is None:
        intersect_point = Point(p1.x, tp.y)
    elif line.m == 0:
        intersect_point = Point(tp.x, p1.y)
    else:
        m = -1 / line.m
        intersect_line = Line(m, tp.y - m * tp.x)
        intersect_point = intersection(line, intersect_line)

    if is_on_line_segment(line_seg, intersect_point):
        return distance(intersect_point, tp)
    else:
        # distance to nearest endpoint of line_seg
        return min(distance(p1, tp), distance(p2, tp))

# -----------------------------------------------------------

def calc_dist_and_print(p1, p2, target_point):
    dist = distance_to_line(p1, p2, target_point)
    dist_v2 = distance_to_line_v2(
        LineSegment(Point(*p1), Point(*p2)),
        Point(*target_point),
    )
    print(
        f'distance_to_line({p1}, {p2}, {target_point}) =',
        f'{dist:.3f} (v1), {dist_v2:.3f} (v2)',
    )

# -----------------------------------------------------------

def main():
    p1, p2, target_point = (0, 0), (1, 1), (1, 0)
    calc_dist_and_print(p1, p2, target_point)
    calc_dist_and_print((0, 0), (1, 0), (1, 0))
    calc_dist_and_print((-1, 0), (0, 0), (2, 2))

def main2():
    test_cases = [
        # Horizontal line, target point directly above
        dict(input=[(1, 2), (4, 2), (2, 3)], output=1.0),

        # Vertical line, target point directly to the right
        dict(input=[(2, 1), (2, 4), (3, 3)], output=1.0),

        # Diagonal line, target point at perpendicular distance
        dict(input=[(0, 0), (2, 2), (2, 0)], output=1.414),

        # Target point closest to an endpoint
        dict(input=[(0, 0), (2, 2), (3, 3)], output=1.414),

        # Line and target point at the same point (zero distance)
        dict(input=[(0, 0), (2, 2), (2, 2)], output=0.0),
    ]
    # run test cases
    for test_case in test_cases:
        p1, p2, target_point = test_case['input']
        expected = test_case['output']

        dist = distance_to_line(p1, p2, target_point)
        dist_v2 = distance_to_line_v2(
            LineSegment(Point(*p1), Point(*p2)),
            Point(*target_point),
        )
        print(
            f'distance_to_line({p1}, {p2}, {target_point}) =',
            f'{dist:.3f} (v1), {dist_v2:.3f} (v2)',
            f'--- expected: {expected})',
        )

if __name__ == '__main__':
    # main()
    main2()
