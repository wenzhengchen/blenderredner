

import os
import bpy
import sys
import math
import random
import numpy as np
from mathutils import Matrix


##############################################################
def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0.0
    q1b = 0.0
    q1c = math.sqrt(2.0) / 2.0
    q1d = math.sqrt(2.0) / 2.0
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2.0 * math.pi - yaw
    pitch = 0.0
    # roll = math.acos(tx * cx + ty * cy)
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)


def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)


def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180.0 * math.pi
    theta = float(azimuth_deg) / 180.0 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


#################################################
# Load rendering light parameters
light_num_lowbound = 0
light_num_highbound = 8
light_dist_lowbound = 15 - 0.05
light_dist_highbound = 15

g_syn_light_azimuth_degree_lowbound = 0.0
g_syn_light_azimuth_degree_highbound = 360.0
g_syn_light_elevation_degree_lowbound = 0.0 - 0.05
g_syn_light_elevation_degree_highbound = 0.0
g_syn_light_energy_mean = 2.0
g_syn_light_energy_std = 0.05

# top light
g_syn_light_elevation_degree_lowbound2 = 85.0 - 0.05
g_syn_light_elevation_degree_highbound2 = 85.0

g_syn_light_environment_energy_lowbound = 1.0 - 0.05
g_syn_light_environment_energy_highbound = 1.0

#######################################################################
# Input parameters
shape_file = sys.argv[-3]
view_num = int(sys.argv[-2])
syn_images_folder = sys.argv[-1]
if not os.path.exists(syn_images_folder):
    os.makedirs(syn_images_folder)

'''
shape_file = '/home/wenzheng/untitled.obj'
shape_view_params_file = '/home/wenzheng/untitled.view'
syn_images_folder = '/home/wenzheng'
'''

# view_params = [[float(x) for x in line.strip().split(' ')] for line in open(shape_view_params_file).readlines()]
view_params = []
for i in range(view_num):
    data2 = []
    '''
    data2.append(random.uniform(0.0, 360.))
    data2.append(random.uniform(25.0, 30.0))
    '''
    # fix it
    
    # our
    data2.append(360.0 * i / view_num)
    data2.append(27.0)
    data2.append(0.0)
    data2.append(random.uniform(0.0, 0.0) + 1.45)
    view_params.append(data2)
    '''
    # nm3r
    data2.append(360.0 * i / view_num)
    data2.append(30.0)
    data2.append(0.0)
    data2.append(random.uniform(0.0, 0.0) + 2.732)
    view_params.append(data2)
    '''

view_params = sorted(view_params, key=lambda s: s[0])

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
# bpy.context.scene.render.use_shadows = False
# bpy.context.scene.render.use_raytrace = False

scene = bpy.context.scene
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.resolution_percentage = 100

# load data
bpy.ops.import_scene.obj(filepath=shape_file) 

# turn off high specular
for m in bpy.data.materials:
    m.specular_intensity = 0.05

#################################################
camObj = bpy.data.objects['Camera']
# camObj.data.lens_unit = 'FOV'
# camObj.data.angle = 30.0 / 180 * np.pi
n = camObj.data.clip_start
f = camObj.data.clip_end
fovy = camObj.data.angle
# view_params.append([n, f, 0, fovy])

# 
viewname = '%s/cameras.txt' % syn_images_folder
fid = open(viewname, 'w')
for i in range(view_num):
    line = '%f %f %f %f\n' % (view_params[i][0], view_params[i][1], view_params[i][2], view_params[i][3])
    fid.write(line)
fid.close()

# set lights
bpy.data.objects['Lamp'].data.energy = 0.0
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True  # remove default light
bpy.ops.object.delete()

# YOUR CODE START HERE
for idx, param in enumerate(view_params):
    azimuth_deg = param[0]
    elevation_deg = param[1]
    theta_deg = param[2]
    rho = param[3]

    # clear default lights
    # this has been deleted
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # set environment lighting
    # bpy.context.space_data.context = 'WORLD'
    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    # set point lights
    lnum = -1
    for lidx, i in enumerate(range(light_num_lowbound, light_num_highbound)):
        light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
        light_elevation_deg = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
        light_azimuth_deg = 45 * lidx - 15
        light_elevation_deg = 0
        light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
        lnum = lnum + 1
        if lnum == 0:
            bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
        else:
            bpy.data.objects['Point.%03d' % lnum].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
    
    # top light
    for lidx, i in enumerate(range(0, 3)):
        light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
        light_elevation_deg = np.random.uniform(g_syn_light_elevation_degree_lowbound2, g_syn_light_elevation_degree_highbound2)
        light_azimuth_deg = 120 * lidx - 60
        light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
        lnum = lnum + 1
        if lnum == 0:
            bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
        else:
            bpy.data.objects['Point.%03d' % lnum].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
    
    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    camObj.location[0] = cx
    camObj.location[1] = cy 
    camObj.location[2] = cz
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    
    # ** multiply tilt by -1 to match pascal3d annotations **
    theta_deg = (-1 * theta_deg) % 360
    
    # we should update it!
    # or the camera setting is the last camera seeting
    bpy.context.scene.update()
    
    # model view
    mat = Matrix([[1., 0., 0., 0.], [0., -0., -1., 0.], [0., 1., -0., 0.], [0., 0., 0., 1.]])
    model_view = (
    camObj.matrix_world.inverted() * 
    mat
    )
    np.save('%s/%d' % (syn_images_folder, idx), model_view)
    
    # syn_image_file = '%d.png' % idx
    # v2
    syn_image_file = '%d.png' %  idx # ((idx +  18) % 24)
    bpy.data.scenes['Scene'].render.filepath = os.path.join(syn_images_folder, syn_image_file)
    bpy.ops.render.render(write_still=True)

