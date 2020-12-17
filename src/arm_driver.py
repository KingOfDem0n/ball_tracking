#!/usr/bin/env python
from __future__ import print_function

import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose2D, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

delta_x = 0
delta_y = 0

def callback(msg):
    global delta_x, delta_y

    delta_x = msg.x
    delta_y = msg.y

if __name__ == "__main__":
    rospy.init_node('arm_driver', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    rospy.Subscriber("shapeCenter", Pose2D, callback)
    move_group = moveit_commander.MoveGroupCommander("arm")
    move_group.set_goal_position_tolerance(0.01)
    move_group.set_goal_orientation_tolerance(0.01)

    pose_goal = move_group.get_current_pose().pose

    while(not rospy.is_shutdown()):
        # pose_goal = move_group.get_current_pose().pose
        q0 = pose_goal.orientation.x
        q1 = pose_goal.orientation.y
        q2 = pose_goal.orientation.z
        q3 = pose_goal.orientation.w
        rpy = list(euler_from_quaternion([q0,q1,q2,q3]))
        rpy[2] += 0.1*delta_x
        q = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        pose_goal.position.z += 0.01*delta_y
        pose_goal.orientation.x = q[0]
        pose_goal.orientation.y = q[1]
        pose_goal.orientation.z = q[2]
        pose_goal.orientation.w = q[3]

        move_group.set_pose_target(pose_goal)
        plan = move_group.go(wait=True)
        # move_group.stop()
        # move_group.clear_pose_targets()
        rate.sleep()
