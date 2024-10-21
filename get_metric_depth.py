import json
import cv2
import numpy as np

# Original image
image = cv2.imread('frame_00110.jpg')

# Get relative depth
relative_depth = cv2.imread('depth_00110.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
# interpolate depth to image size
relative_depth = cv2.resize(relative_depth, (image.shape[1], image.shape[0]))
# save relative depth
cv2.imwrite('new_depth_00110.png', relative_depth / relative_depth.max() * 255)

# Get parameters for specific frame
parameters = open('frame_00110.json')
parameters = parameters.read()
json_parameters = json.loads(parameters)

intrinsic_parameters = np.array(json_parameters["intrinsics"]).reshape(3, 3)

# Get 3D point coordinates using relative depth and intrinsic parameters
three_d_points = np.zeros((relative_depth.shape[0], relative_depth.shape[1], 3))
for i in range(relative_depth.shape[0]):
    for j in range(relative_depth.shape[1]):
        three_d_points[i, j, 0] = (j - intrinsic_parameters[0, 2]) * relative_depth[i, j] / intrinsic_parameters[0, 0]
        three_d_points[i, j, 1] = (i - intrinsic_parameters[1, 2]) * relative_depth[i, j] / intrinsic_parameters[1, 1]
        three_d_points[i, j, 2] = relative_depth[i, j]

# Use camera to world transformation matrix to get world coordinates
camera_to_world_parameters = open('info.json')
camera_to_world_parameters = camera_to_world_parameters.read()
json_camera_to_world_parameters = json.loads(camera_to_world_parameters)
camera_to_world_parameters = json_camera_to_world_parameters["transformToWorldMap"]
camera_to_world_parameters_array = [camera_to_world_parameters["m11"], camera_to_world_parameters["m12"], camera_to_world_parameters["m13"], camera_to_world_parameters["m14"], camera_to_world_parameters["m21"], camera_to_world_parameters["m22"], camera_to_world_parameters["m23"], camera_to_world_parameters["m24"], camera_to_world_parameters["m31"], camera_to_world_parameters["m32"], camera_to_world_parameters["m33"], camera_to_world_parameters["m34"], camera_to_world_parameters["m41"], camera_to_world_parameters["m42"], camera_to_world_parameters["m43"], camera_to_world_parameters["m44"]]
camera_to_world_parameters_array = np.array(camera_to_world_parameters_array).reshape(4, 4)

# Get middle point coordinates
three_d_points_homogenious = np.concatenate([three_d_points, np.ones(three_d_points.shape[:-1] + (1,))], axis=-1)

# middle_point = three_d_points[three_d_points.shape[0] // 2][three_d_points.shape[1] // 2]
# middle_point = middle_point.tolist()
# middle_point.append(1)
# middle_point = np.array(middle_point)

# real_world_coordinates = camera_to_world_parameters_array @ middle_point
# print(real_world_coordinates)

# metric_distance = np.linalg.norm(real_world_coordinates)
# print(metric_distance)
# print(camera_to_world_parameters_array.shape)
# print(three_d_points_homogenious.transpose(2, 0, 1).shape)
# real_world_coordinates = camera_to_world_parameters_array @ three_d_points_homogenious.transpose(2, 0, 1)
# print(real_world_coordinates.shape)
# metric_distance_matrix = np.linalg.norm(real_world_coordinates, axis=-1)
# print(metric_distance_matrix.shape)

real_world_coordinates = np.zeros_like(three_d_points)
for i in range(three_d_points_homogenious.shape[0]):
    for j in range(three_d_points_homogenious.shape[1]):
        sample_point = three_d_points_homogenious[i][j]
        real_world_coordinates[i][j] = (camera_to_world_parameters_array @ sample_point)[:-1]

i1 = three_d_points.shape[0] // 2 - 30
j1 = three_d_points.shape[1] - 290
i2 = three_d_points.shape[0] // 2
j2 = 260

# original image
image = cv2.imread('frame_00110.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (three_d_points.shape[1], three_d_points.shape[0]))
image = cv2.circle(image, (j1, i1), 5, (0, 0, 255), -1)
image = cv2.circle(image, (j2, i2), 5, (0, 0, 255), -1)

# put a line between two points
image = cv2.line(image, (j1, i1), (j2, i2), (0, 0, 255), 2)

point_1 = real_world_coordinates[i1][j1]
point_2 = real_world_coordinates[i2][j2]
metric = np.linalg.norm(point_1 - point_2)
true_metric = metric / 1000

# show true metric above the middle of the line
font = cv2.FONT_HERSHEY_SIMPLEX
image = cv2.putText(image, f'{true_metric:.2f} m', (j1 + 10, i1 - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite('new_image_00110.png', image)


# metric_distance_matrix = np.linalg.norm(real_world_coordinates, axis=-1)
# print(metric_distance_matrix[metric_distance_matrix.shape[0] // 2][metric_distance_matrix.shape[1] // 2])
# print(np.unique(metric_distance_matrix / 1000))
# print(metric_distance_matrix[0, 0] / relative_depth[0, 0])
# print(metric_distance_matrix[1, 1] / relative_depth[1, 1])
# print(metric_distance_matrix[0, 2] / relative_depth[0, 2])
# print(metric_distance_matrix[0, 3] / relative_depth[0, 3])
# print(metric_distance_matrix[0, 4] / relative_depth[0, 4])
# print(metric_distance_matrix[0, 5] / relative_depth[0, 5])
# print(metric_distance_matrix[metric_distance_matrix.shape[0] // 2][metric_distance_matrix.shape[1] // 2])
# cv2.imwrite('metric_distance.png', metric_distance_matrix / metric_distance_matrix.max() * 255)