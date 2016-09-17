import heapq
import numpy as np
import math

def find_neighbours(points, origin, k):
    n = points.shape[0]
    heap = []
    for i in range(0, n): 
        diff = points[i] - origin
        dist = np.dot(diff, diff)
        if len(heap) == k:
            if -heap[0][0] > dist:
                heapq.heappop(heap)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, i))
    heap = [(np.sqrt(abs(x)), y) for (x,y) in heap]
    heap.sort()
    return heap

def get_real_distances(neighs, points, origin):
    res = []
    for (dist, ind) in neighs:
        diff = points[ind] - origin
        r_dist = np.dot(diff, diff)
        res.append(np.sqrt(r_dist))
    return res


def jackart_distance(red_points, red_origin, points, origin, k):
    real = find_neighbours(points, origin, k)
    red = find_neighbours(red_points, red_origin, k)
    (_, red_i) = zip(*red)
    (_, real_i) = zip(*real)
    red_is = set(red_i)
    real_is = set(real_i)
    inter = red_is.intersection(real_is)
    union = red_is.union(real_is)
    return (1 - (len(inter)/len(union)), len(inter), len(union))

def neighbours_lp_ratio(red_points, red_origin, points, origin, k, p):
    real = find_neighbours(points, origin, k)
    red = find_neighbours(red_points, red_origin, k)
    (distances, _) = zip(*real)
    appr_distances = get_real_distances(red, points, origin);
    if math.isinf(p):
        orig_v = max(distances)
        new_v = max(appr_distances)
    else:
        orig_v = sum([d**p for d in distances])
        new_v = sum([d**p for d in appr_distances])
    return new_v/orig_v
