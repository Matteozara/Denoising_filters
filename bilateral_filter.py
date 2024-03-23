import math
import sys
import open3d as o3d
from scipy.spatial import Delaunay
import numpy as np

def neighborhood_radius(v, tri, points):
    triangle = tri.find_simplex(v)
    neighbors = tri.neighbors[triangle]
    maximum = 1e10
    for n in neighbors:
        if (n != -1):
            for p in tri.simplices[n]:
                dist = np.linalg.norm(v-points[p])
                if ((dist > 0) and (dist < maximum)):
                    maximum = dist
    return maximum

def calc_normal(v, degree, tri, points):
    #find neighbors
    triangle = tri.find_simplex(v)
    neighbors = tri.neighbors[triangle]
    
    for d in range(degree):
        extra = []
        for n in neighbors:
            if (n != -1):
                extra.append(tri.neighbors[triangle])
        neighbors = np.append(neighbors, np.unique(extra))
    
    normal_sum = [0, 0, 0]
    count = 0
    
    for n in neighbors:
        if (n != -1):
            s = tri.simplices[n]
            v1 = points[s[0]] - points[s[1]]
            v2 = points[s[0]] - points[s[2]]
            crossp = np.cross(v1, v2)
            normal_sum[0] += crossp[0]
            normal_sum[1] += crossp[1]
            normal_sum[2] += crossp[2]
            count += 1
    
    normal_average = [n / count for n in normal_sum]
    normal = np.linalg.norm(normal_average)
    normal_normalized = [n / normal if normal != 0 else 0 for n in normal_average]
    
    neighbor_points = []
    for n in neighbors:
        if (n != -1):
            for s in tri.simplices[n]:
                neighbor_points.append(s)
    neighbor_points = np.unique(neighbor_points)
    
    return normal_normalized, neighbor_points


def bilateral_filter(pcd, iterations, n_degree, sigma_c=None, sigma_s=None):
    points = np.asarray(pcd.points)
    points = points.astype(np.float32)
    tri = Delaunay(points[:-1], qhull_options="Qbb Qc Qz Q12 QJ Qt")

    for i in range(iterations):
        print("Iteration: " + str(i))
        new_points = []
        index = 0

        toolbar_width = 50
        sys.stdout.write("[%s]"  % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))
        bar_step = round(points.shape[0] / toolbar_width)
        
        for p in points:
            normal, neighbor_points = calc_normal(p, n_degree, tri, points)
            
            #calculate sigma_c if not passed
            if sigma_c is None:
                sigma_c = neighborhood_radius(p, tri, points)
            
            neighbors = []
            for y in neighbor_points:
                v = points[y]
                dist = np.linalg.norm(p-v)
                if (dist < 2*sigma_c):
                    neighbors.append(v)
            
            #calculate sigma_s if not passed
            if sigma_s is None:
                average_offset = 0
                offsets = []

                for n in neighbors:
                    t = np.linalg.norm([x * np.dot((n - p), normal) for x in normal])
                    t = math.sqrt(t*t)
                    average_offset += t
                    offsets.append(t)
                if (len(neighbors) != 0):
                    average_offset /= len(neighbors)
                #calculate standard deviation
                o_sum = 0
                for o in offsets:
                    o_sum += (o - average_offset) * (o - average_offset)
                if (len(offsets)):
                    o_sum /= len(offsets)
                sigma_s = math.sqrt(o_sum)
                
                minimum = 1.0e-12
                if (sigma_s < minimum):
                    sigma_s = minimum
            
            #filter calculations
            total = 0
            normalizer = 0
            for n in neighbors:
                t = np.linalg.norm(n - p)
                h = np.linalg.norm([x * np.dot((n - p), normal) for x in normal])
                wc = math.exp((-1*t*t)/(2 * sigma_c *sigma_c))
                ws = math.exp((-1*h*h)/(2 * sigma_s *sigma_s))
                total += wc * ws * h
                normalizer += wc * ws
            if normalizer != 0:
                factor = total/normalizer
                modification = [n * factor for n in normal]
                new_point = p + modification
                new_points.append(new_point)
            else:
                new_points.append(p)
            index += 1
            if (index % bar_step == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
        points = np.array(new_points)
        points = points.astype(np.float64)
        sys.stdout.write("\n")

    sys.stdout.write("\n")
        
    #pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.io.write_point_cloud("res_bilateral_filter.pcd", pcd)
    return points


