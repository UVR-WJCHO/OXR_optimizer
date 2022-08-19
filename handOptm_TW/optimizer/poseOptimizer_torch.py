import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('/root/halo')
sys.path.append('/root/halo/halo')
import torch
import torch.nn as nn
import numpy as np
import json
import os
from os.path import join
from utils.uvd_transform import uvdtouvd
import matplotlib.pyplot as plt
from utils.visualize import visualize

from halo.models.halo_adapter.adapter import HaloAdapter
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight
import trimesh
from scipy.spatial.transform import Rotation as R


baseDir = './test_seq/self/predictedPose'
handposeRoot = join(baseDir, 'onlyHandWorldCoordinate_uvd.json')
root = '/root/OXR_projects/optimizer'
RESULT_PLOT = 0

class Model(nn.Module):
    def __init__(self, idx, ref_cam):
        super().__init__()
        weights = self.initPose(idx, ref_cam) #[21,3]
        self.weights = nn.Parameter(weights)

    def initPose(self, idx, ref):
        with open(handposeRoot, 'r') as fi:
            data = json.load(fi)
        handpose_0 = np.array(data['0_%d'%ref])
        handpose_1 = np.array(data['1_%d'%ref])
        handpose_2 = np.array(data['2_%d'%ref]) #[116, 21, 2]
        init_joint = torch.FloatTensor(np.mean([handpose_0, handpose_1, handpose_2], axis=0))
        print("Initial joint : ", init_joint.shape)

        return init_joint[idx]
    
    def forward(self):
        output_joint = self.weights
        return output_joint

def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh

def create_skeleton_mesh(joints):
    mesh_list = []
    # Sphere for joints
    for idx, j in enumerate(joints):
        joint_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.005)
        joint_sphere.vertices += j.detach().cpu().numpy()
        mesh_list.append(joint_sphere)
    
    parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    # Cylinder for bones
    for idx in range(1, 21):
        ed = joints[idx].detach().cpu().numpy()
        st = joints[parent[idx]].detach().cpu().numpy()
        skel = trimesh.creation.cylinder(0.003, segment=(st, ed))
        mesh_list.append(skel)

    skeleton_mesh = concat_meshes(mesh_list)
    return skeleton_mesh

def render_halo(kps, halo_adapter, renderer, out_img_path=None):
    hand_joints = torch.from_numpy(kps)

    halo_mesh = halo_adapter(hand_joints.unsqueeze(0).cuda() * 100.0, joint_order="mano", original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / 100.0

    # render HALO
    camera_t = np.array([0., 0.05, 2.5])
    rend_img = renderer.render(halo_mesh.vertices * 1.0, halo_mesh.faces, camera_t=camera_t)
    
    skeleton_mesh = create_skeleton_mesh(hand_joints)

    skeleton_img = renderer.render(skeleton_mesh.vertices, skeleton_mesh.faces, camera_t=camera_t)
    final = np.hstack([skeleton_img, rend_img])
    # plt.imshow(final)
    if out_img_path is not None:
        plt.imsave(out_img_path, final)
    # plt.show()
    return 

class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=600, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


    def render_vertex_color(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               vertex_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if vertex_color is None:
            vertex_color = self.colors['light_blue']


        self.renderer.set(v=vertices, f=faces,
                          vc=vertex_color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


def refinement(frame, model, optimizer, device, refpose_set, Ks, ext, n=100, refcam=0):
    smoothl1_criterion = nn.SmoothL1Loss(reduction='mean')
    mse_criterion = nn.MSELoss(reduction='mean')
    losses = []
    for i in range(n):
        print("%d / %d"%(i+1, n))
        losslist = []
        refined_joint = model()

        # add weight for each cam's loss
        for cam in range(3):
            losslist.append(smoothl1_criterion(uvdtouvd(refined_joint, Ks[refcam], Ks[cam], ext['ref%d'%refcam][cam])[:,:-1], refpose_set[cam][:,:-1].to(device)))

        loss = sum(losslist) / len(losslist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu())
        if i == n-1:
            result_joint = model()
        if frame == RESULT_PLOT:
            visualize(refined_joint, frame, refcam, Ks, ext, i)
    return losses, result_joint


if __name__ == "__main__":
    import pickle
    Frames = 5
    REFCAM = 0
    Ks = np.load('./intrinsic.npy') #0,1,2
    with open('./extrinsic.json', 'r') as f:
        ext = json.load(f) #0_ref, 1_ref, 2_ref
    with open(handposeRoot, 'r') as fi:
            data = json.load(fi)
    refpose_0 = np.array(data['0_0'])
    refpose_1 = np.array(data['1_1'])
    refpose_2 = np.array(data['2_2'])
    refpose_set = torch.FloatTensor([refpose_0, refpose_1, refpose_2])
    ###
    use_multigpu = False
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
    device = torch.device("cuda:%d" % 0 if use_cuda else "cpu")
    gpu_brand = torch.cuda.get_device_name(0) if use_cuda else None
    gpu_count = torch.cuda.device_count() if use_multigpu else 1
    if use_cuda:
        print('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
    ###

    loss_dict = {}
    refined_joints = []
    for i in range(Frames):
        print("%dst frame"%(i+1))
        m = Model(i, REFCAM).to(device)
        if use_multigpu:
            m = nn.DataParallel(m)
            print("Training on Multiple GPU's")
        optm = torch.optim.Adam(m.parameters(), lr=0.1, betas=(0.5, 0.99))
        losses, refined_joint = refinement(i, m, optm, device, refpose_set[:, i, :, :], Ks, ext, n=10, refcam=REFCAM)
        loss_dict['CAM%d_%d'%(REFCAM, i)] = losses
        refined_joints.append(refined_joint)
        

    loss_result = loss_dict['CAM0_%d'%RESULT_PLOT]
    # plt.plot(range(len(loss_result)), loss_result)
    # plt.savefig('./result.png')
    # plt.close()

    with open('./test_joint.pickle', 'wb') as fi:
        pickle.dump(refined_joints, fi, pickle.HIGHEST_PROTOCOL)
    '''
    HALO
    '''
    # renderer = Renderer()

    # halo_config_file = "/root/halo/configs/halo_base/yt3d_b16_keypoint_normalized_fix.yaml"
    # halo_adapter = HaloAdapter(halo_config_file, device = device, denoiser_pth=None)

    # output_path = os.path.join(root,'halo_output')
    # print("Rendering HALO...")
    # for idx, joint in enumerate(refined_joints):
    #     out_img_path = os.path.join(output_path, "%03d.png"%(idx))
    #     joint = np.array(joint.detach().cpu()) / 1000
    #     joint -= joint[9]
    #     print(np.min(joint))
    #     print(np.max(joint))
    #     render_halo(joint, halo_adapter, renderer, out_img_path=out_img_path)