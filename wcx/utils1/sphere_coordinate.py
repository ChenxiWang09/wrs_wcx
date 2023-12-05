import math

def modify_spherical_coordinates(point: object, center: object, dradius: object = -1, dtheta: object = 0.1, dphi: object = 0.1) -> object:
    # Convert Cartesian coordinates to spherical coordinates
    rel_x = point[0] - center[0]
    rel_y = point[1] - center[1]
    rel_z = point[2] - center[2]
    theta = math.atan2(rel_y, rel_x)
    radius = math.sqrt(rel_x**2+rel_y**2+rel_z**2)
    phi = math.acos(rel_z / radius)
    if dradius != -1:
        radius = dradius
    theta += dtheta
    phi += dphi
    # print('phi:', phi)
    # print('theta:', theta)
    # Convert back to Cartesian coordinates
    x_new = center[0] + radius * math.sin(phi) * math.cos(theta)
    y_new = center[1] + radius * math.sin(phi) * math.sin(theta)
    z_new = center[2] + radius * math.cos(phi)

    return x_new, y_new, z_new
