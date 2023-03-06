# Author: wxk
# Time: 2022/1/6 13:54

import os
from shutil import copyfile
from PIL import Image

# source_path = r'D:\wangxinkuang\data\CarDD\CarDD\CarDD-TR\CarDD-TR-Mask'
# goal_path = r'D:\wangxinkuang\data\CarDD\CarDD\CarDD-TR\CarDD-TR-Edge'
source_path = r'D:\wangxinkuang\data\CarDD\CarDD\CarDD-TR\CarDD-TR-Image'
goal_path = r'D:\wangxinkuang\data\CarDD\CarDD\CarDD-TR\CarDD-TR-Mask'
edge_path = r'D:\wangxinkuang\data\CarDD\CarDD\CarDD-TR\CarDD-TR-Edge'
if not os.path.exists(goal_path):
    os.makedirs(goal_path)
goal_list = os.listdir(goal_path)

for file in os.listdir(source_path):
    # if file.endswith('_edge.png'):
    # copyfile(os.path.join(source_path, file), os.path.join(goal_path, file))
    # os.remove(os.path.join(source_path, file))

    # if file.replace('.png','_edge.png') not in goal_list:
    #     print(file)

    source_file = os.path.join(source_path, file)
    goal_file = os.path.join(goal_path, file.replace('.jpg', '.png'))
    edge_file = os.path.join(edge_path, file.replace('.jpg', '_edge.png'))
    im_source = Image.open(source_file)
    im_goal = Image.open(goal_file)
    im_edge = Image.open(edge_file)
    if im_source.size[0] != im_goal.size[0] or im_source.size[1] != im_goal.size[1]:
        print(file, 'not match!')
        print('source:', source_file, im_source.size[0], 'x', im_source.size[1])
        print('goal:', goal_file, im_goal.size[0], 'x', im_goal.size[1])
        print('edge:', edge_file, im_edge.size[0], 'x', im_edge.size[1])

