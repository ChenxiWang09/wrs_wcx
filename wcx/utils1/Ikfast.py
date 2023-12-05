import math
import wcx.utils1.rotate_matrix as rot_m

from ur_ikfast import ur_kinematics
import numpy as np

ur3e_arm = ur_kinematics.URKinematics('ur3e')

joint_angles = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.]  # in radians
print("joint angles", joint_angles)

pose_quat = ur3e_arm.forward(joint_angles)
pose_matrix = ur3e_arm.forward(joint_angles, 'matrix')
trans = np.array([[777.4147183, -214.3917632, 899.41354136]]).T
rot_mat = np.array([[0.83151734,  0.30117316, -0.46676936],
 [0.52442904, -0.14852815, 0.83839941],
 [0.18317501, -0.94193106, -0.28144768]])
joint_angles = [94.29238708*math.pi/180, -144.35684839*math.pi/180,   33.48551148*math.pi/180 ,-122.78410571*math.pi/180 , 144.58603424*math.pi/180,
 -157.83476072*math.pi/180]

mid = np.hstack((rot_mat, trans))
Homo = np.vstack((mid, np.array([[0, 0, 0, 1]])))
rot_ma = np.dot(rot_m.matrix_generate(-90, 'x'), rot_m.matrix_generate(-90, 'z'))
rot_ma = np.hstack((rot_ma, np.array([[365, -344.99999999999994, 1330]]).T))
rot_ma = np.vstack((rot_ma, np.array([[0, 0, 0, 1]])))
Homo = np.dot(Homo, rot_ma)
Homo = Homo[:-1, :]

# print("forward() quaternion \n", pose_quat)
# print("forward() matrix \n", pose_matrix)
# # print("inverse() all", ur3e_arm.inverse(pose_quat, True))

print("inverse() one from matrix", ur3e_arm.inverse(Homo, False, q_guess=joint_angles))