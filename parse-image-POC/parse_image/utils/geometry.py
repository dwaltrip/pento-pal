import numpy as np
import numpy.linalg as la


def dist_from_point_to_line(point, line_seg):
    """ 
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    To fine the distance from line segment AB and point C:
        Form a parallelogram with vectors AC and AB.
        Calculate the area of the parallelogram.
        Then the distance is equal to the area divided by the length of base.
    """
    a, b = line_seg
    a, b, point = np.array(a), np.array(b), np.array(point)
    v_base = b - a
    v_side = point - a
    area = la.norm(np.cross(v_base, v_side))
    base_length = la.norm(v_base)
    return area / base_length
