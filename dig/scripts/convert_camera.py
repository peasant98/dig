import json
import numpy as np


def polycam2pytorch3d(cameras, path, w, h, fl_x, fl_y, data_transform):
    camera_list = []
    nerfc2py3d = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]])
    mat = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    inv_mat = nerfc2py3d@mat
    
    scale_mat = np.eye(4)
    scale_mat[:3,] = np.array(data_transform["transform"])
    # scale_mat[2,3] = scale_mat[1,3]
    # scale_mat[1,3] = scale_mat[2,3]
    print(scale_mat)
    print(data_transform["scale"])
    print(np.linalg.inv(scale_mat))
    
    
    for idx, camera in enumerate(cameras):
        camera_dict = {}
        camera_dict['id'] = idx
        camera_dict['img_name'] = camera['file_path'].split('/')[-1][:-4]
        camera_dict['width'] = w
        camera_dict['height'] = h
        camera_dict['fx'] = fl_x
        camera_dict['fy'] = fl_y
        camera_dict['cx'] = camera['cx']
        camera_dict['cy'] = camera['cy']
        
        T = np.array(camera['transform_matrix'])
        

        newT = T
        newT = scale_mat @ newT
        newT[:3, 3] = newT[:3,3] * np.array(data_transform["scale"])
        
        newT = newT @ inv_mat
        
        camera_dict['position'] = newT[:3,3].tolist()
        camera_dict['rotation'] = newT[:3,:3].tolist()
        camera_list.append(camera_dict)
    with open(path, 'w') as f:
        json.dump(camera_list, f)
        
            
colmap_opencv_camera_file = "/home/yujustin/xi/data/corl_data/scans/wire_cutters/transforms.json"
colmap_opencv_camera  = json.loads(open(colmap_opencv_camera_file).read())
data_transform = json.loads(open("/home/yujustin/outputs/wire_cutters/dig/2025-02-02_223810/dataparser_transforms.json").read())
# print(colmap_opencv_camera['frames'])
polycam2pytorch3d(colmap_opencv_camera['frames'], "/home/yujustin/xi/dependencies/sugar/output/vanilla_gs/wire_cutter_polycam_exported/cameras.json", colmap_opencv_camera['frames'][0]['w'], colmap_opencv_camera['frames'][0]['h'], colmap_opencv_camera['frames'][0]['fl_x'], colmap_opencv_camera['frames'][0]['fl_y'], data_transform)