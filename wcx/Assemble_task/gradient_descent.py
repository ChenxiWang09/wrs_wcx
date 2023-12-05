import numpy as np
import cv2
import copy
import math
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from wcx.utils1.sphere_coordinate import modify_spherical_coordinates
# from findBoundingBox.BoundingBox import *
file = ''
filename = []

class MyEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global file
        global filename
        if not event.is_directory:
            if file != event.src_path:
                file = event.src_path
                filename.append(event.src_path)
                # print("read cost:", file)


# Compute the gradient of the cost function
def compute_gradients(initial_point: object, center_point: object, shreshold=[0.3, 0.1], dtheta=1, dphi=1,
                       learning_rate: object = 20) -> object:
    # Compute the gradients of theta and phi using finite differences
    global filename
    filename = []
    my_event = MyEventHandler()
    folder_path = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/watch_npy'
    observer = Observer()
    observer.schedule(my_event, folder_path, recursive=False)
    observer.start()
    print('Waiting for matching result...')
    while True:
        try:
            cost_origin = np.load(filename[0])
            cost_theta = np.load(filename[1])
            cost_phi = np.load(filename[2])

            observer.stop()
            break
        except:
            continue
    # Finite difference approximation
    print('cost_origin:', cost_origin, 'cost_theta:', cost_theta, 'cost_phi:', cost_phi)
    cost = np.array([cost_origin, cost_theta, cost_phi])
    d_theta = (cost_origin - cost_theta ) / dtheta*learning_rate
    d_phi = (cost_origin - cost_phi) / dphi*learning_rate
    print('d-theta:', d_theta, 'd-phi:', d_phi)
    new_point = modify_spherical_coordinates(initial_point, center_point, -1, d_theta, d_phi)
    print('new_point:', new_point)

    state = False

    if math.fabs(d_phi)+math.fabs(d_theta) < shreshold[1] and cost_origin < shreshold[0]:
        state = True

    return new_point, state, cost

def compute_gradients_boundingbox(initial_point: object, center_point: object, radius: object, epsilon: object = 5, learning_rate: object = 20) -> object:
    # Compute the gradients of theta and phi using finite differences
    temp_bounding_box = np.load('data/temp_lhrate.npy')
    images = []
    for i in range(3):
        path = "/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/watch_image/screenshot_" + str(
            i) + ".png"
        images.append(cv2.imread(path, 0))
    cost_origin = c_lh_rate(images[0], background_color=[85, 120]) - temp_bounding_box
    cost_theta = c_lh_rate(images[1], background_color=[85, 120]) - temp_bounding_box
    cost_phi = c_lh_rate(images[2], background_color=[85, 120]) - temp_bounding_box

    print('cost_origin:', cost_origin, 'cost_theta:', cost_theta, 'cost_phi:', cost_phi)
    d_theta = (cost_origin - cost_theta ) / epsilon*learning_rate
    d_phi = (cost_origin - cost_phi) / epsilon*learning_rate
    print('d-theta:',d_theta, 'd-phi:', d_phi)
    new_point = modify_spherical_coordinates(initial_point, center_point, radius, d_theta, d_phi)
    print('new_point:', new_point)

    state = False

    if(math.fabs(d_phi)+math.fabs(d_theta) < 0.15 and cost_origin < 0.30):
        state = True

    return new_point, state


# Example usage
if __name__ == '__main__':
    object_template = cv2.imread('/wcx/Assemble_task/data/simulation_image/screenshot_0.png')
    object_template = cv2.cvtColor(object_template, cv2.COLOR_BGR2GRAY)
    initial_position = np.array([0, 0, 0])
    # final_position = gradient_descent(image, object_template, initial_position)
    # print("Optimal position: ", final_position)
