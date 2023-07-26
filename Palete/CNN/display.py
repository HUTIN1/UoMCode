import argparse

import math
import os
import pandas as pd
import numpy as np 
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
TexturesVertex, FoVPerspectiveCameras, look_at_rotation,look_at_view_transform
)

from pytorch_lightning.callbacks import ModelCheckpoint
from monai.transforms import Compose


from dataset import TeethDatasetLm, TeethDatasetLmCoss, TeethDatasetPatch
from net import MonaiUNetHRes, MonaiUnetCosine
from ManageClass import RandomPickTeethTransform, PickLandmarkTransform, MyCompose, RandomRotation, UnitSurfTransform, RandomTranslation

from utils import image_grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from azureml.core.run import Run
# run = Run.get_context()
from utils import WriteSurf
from torch import tensor
from PIL import Image
from vtk.util.numpy_support import  numpy_to_vtk
import plotly.graph_objects as go
# import cv2 
import utils
from PIL import Image
import io


def ListToMesh(list,radius=0.05):
    list_verts =[]
    list_faces = []
    for point in list:
        sphere = ico_sphere(2)
        list_verts.append(sphere.verts_packed()*radius+tensor(point).unsqueeze(0).unsqueeze(0))
        list_faces.append(sphere.faces_list()[0].unsqueeze(0))


    list_verts = torch.cat(list_verts,dim=0)
    list_faces = torch.cat(list_faces,dim=0)
    mesh = Meshes(verts=list_verts,faces=list_faces)

    return mesh

def makeline(l,color='red'):
        return go.Scatter3d(x=l[0],y=l[1],z=l[2],mode='lines',
        line=go.scatter3d.Line(color=color,width=10))

def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    mount_point = args.mount_point

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None

    train_transfrom = MyCompose([UnitSurfTransform()])#,RandomTranslation(0.25)])
    # train_transfrom = MyCompose([PickLandmarkTransform(args.landmark,args.property)])
    radius = 1.6
    # model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples,radius=radius)
    model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=4, subdivision_level=2,radius=1.6)
    path_model = '/home/luciacev/Downloads/epoch=1681-val_loss=0.528_unetseg_butterfly_create_patch_manually_+randomtranslation.ckpt'

    model.load_state_dict(torch.load(path_model)['state_dict'])


    # path_scan = utils.search('/home/luciacev/Desktop/Data/IOSReg/Felicia/oriented/T1/Upper','.vtk')['.vtk']
    # print(path_scan)
    # train_ds  = TeethDatasetLmCoss(mount_point = args.mount_point, df = df_train ,surf_property = args.property,transform =train_transfrom,landmark=args.landmark ,test=True,random_rotation=True)
    train_ds  = TeethDatasetPatch(mount_point = args.mount_point, df = df_train ,surf_property = args.property,transform =train_transfrom,landmark=args.landmark ,test=True,random_rotation=False)
    # train_ds  = TeethDatasetPatch(mount_point = args.mount_point, df = df_train ,surf_property = args.property,landmark=args.landmark ,test=True)
    
    dataloader = DataLoader(train_ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')


    model.to(device)
    model.eval()


    

    # iterdata= iter(dataloader)
    # data = next(iterdata)
    data = train_ds[1]
    print(f'name {train_ds.getName(1)}')
    V, F, CN, CL, landmarks = data
    # V, F, CN = data
    # V, F, CN, CL, vector , distance = data
    V = V.unsqueeze(0)
    F = F.unsqueeze(0) 
    CN = CN.unsqueeze(0)
    CL = CL.unsqueeze(0)

    # landmark = vector * distance
    # print(f'landmark pos in display {landmark}')

    print('V.size()',V.size())
    print('F.size()',F.size())
    # print('CL.shape',CL.shape)
    # print(f'color map {torch.unique(CL)}')

    # a = torch.linspace(-0.75,0.75,4)
    # ico_verts = torch.tensor([[x.item(),y.item(),0.75] for x in a for y in a]).to(torch.float32) 
    # ico_verts  = ico_sphere(1).verts_packed()
    ico_verts, ico_faces = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=1, sl=4))
    # # ico_verts[...,2] = ico_verts[...,2]+0.5
    # ico_verts = ico_verts.to(torch.float32)
    # ico_list = []
    # for ico in ico_verts :
    #     if ico[1] < 0  and ico[2] > 0:
    #         ico_list.append(ico.unsqueeze(0))
    # ico_verts = torch.cat(ico_list,dim=0)
    # for idx, v in enumerate(ico_verts):
    #     # if (torch.abs(torch.sum(v)) == radius):
    #         ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))

    # ico_verts = torch.tensor([[0,0,1]],device=device).to(torch.float32)
    # # sphere =  Pointclouds(points=[sphere_verts])
    # ico_verts = torch.tensor([[0,0,-0.9],[0.2,0,-0.9],[-0.2,0,-0.9]]).to(torch.float32)+torch.tensor([[0,-0.2,0]])
    ico_verts = torch.tensor([[0,0,-0.9],[0.2,0,-0.9],[-0.2,0,-0.9],[0,0.2,-0.9],[0,-0.2,-0.9],[-0.2,-0.2,-0.9],[0.2,-0.2,-0.9]]).to(torch.float32)


    # matrix_rotation = torch.tensor(utils.RotationMatrix(np.array([1,0,0]),np.array(3.1415/8))).to(torch.float32)
    # ico_verts = torch.matmul(matrix_rotation,ico_verts.t()).t()
    # for i in range(1,3):
    #      CL2 = CL
    #      CNL = CN
    #      index = torch.argwhere(CL2)
    #      CNL[index] = i*5
    #      CL2 = CL
    texture = TexturesVertex(CN)
    print(f'V { V.shape}, F {F.shape}, CN {CN.shape}')
    mesh = Meshes(verts=V,faces=F,textures=texture)
    V = V.to(device)
    F = F.to(device)
    # CL = CL.to(device)
    CN = CN.to(device)
    # X, PF = model.render(V, F, CN)
    X, _ , _ = model((V,F,CN))
    # R=[]
    # T = []

    # for camera_position in ico_verts:

    #     camera_position = camera_position.unsqueeze(0).to(device)

    #     r = look_at_rotation(camera_position,at=2*camera_position).to(device)  # (1, 3, 3)
    #     t = torch.bmm(r.transpose(1, 2), camera_position[:,:,None])[:, :, 0].to(device)   # (1, 3)
    #     # r = -r 
    #     # t = torch.tensor([[0,0,0.5]])
    #     # _ , t = look_at_view_transform(1,1,0)
    #     # t = t.to(device)
    #     # t = camera_position
    #     # r = - torch.eye(3).unsqueeze(0)
    #     R.append(r)
    #     T.append(t)

    # R = torch.cat(R,dim=0)
    # T = torch.cat(T,dim=0)
    # cam = FoVPerspectiveCameras(R=R, T=T)
    fig = plot_scene({
    "subplot1": {
        "mouth" : mesh,
        # 'sphere':ListToMesh(ico_verts.tolist(),radius=0.01),
        # 'cam' : cam,
        # 'point cam': ListToMesh(t.tolist()),
        # 'landmark' : ListToMesh(landmark.tolist(),radius=0.01)
        # 'landmarks': ListToMesh(landmarks.values(),radius=0.01)
    }
    })



    xline = [[0,10],[0,0],[0,0]]
    yline = [[0,0],[0,10],[0,0]]
    zline = [[0,0],[0,0],[0,10]]
    fig.add_trace(makeline(xline))
    fig.add_trace(makeline(yline,color='cyan'))
    fig.add_trace(makeline(zline,color='green'))
    fig.show()
    print("fait")


    # image_grid(y[...,:3].cpu().numpy(),rows=4,cols=3)
    X = X.permute(1,0,3,4,2)
    # print(X.size())
    # image_grid(X[...,:3].cpu().numpy(),rows=2,cols=5)
    # plt.show()
    # print('unique',torch.unique(X))
    # X = X.permute(1,0,3,4,2)
    # Xmax = torch.max(X)
    # Xmin = torch.min(X)
    # X = (X + Xmin)/Xmax *255
    for i in range(X.size()[0]):
        imname = f'/home/luciacev/Desktop/Project/ALIDDM/figure/prediction{i}.jpeg'
        # plt.figure()
        # plt.imshow(X[i,0,...,:3].cpu().numpy())
        # plt.show()
    #     # image = X[i,0,...,:3]
    #     # mini = torch.min(image)
    #     # maxi = torch.max(image)
    #     # new_image = torch.tensor((image-mini)/(maxi-mini)*255,dtype=torch.int)
    #     # print(torch.unique(new_image))
    #     # cv2.imwrite(imname,new_image.cpu().numpy())
    #     print(f'iamge shape {X.shape}')
    #     # plt.plot(X[0,i,...,:3].cpu().numpy())
    #     # plt.plot(X[0,i,:3,...].cpu().numpy())
    #     # plt.figure()
    #     buf = io.BytesIO()
        # plt.imsave(imname,X[i,0,...,:3].cpu().numpy().astype(np.uint8))
        image = X[i,0,...,0].cpu().detach().numpy().astype(np.uint8)
        image = np.expand_dims(image,axis=-1)
        image = np.concatenate((image,image,image),axis=-1)
        print(f'X shape {image.shape}')
        mpimg.imsave(imname,image)
        # plt.plot(X[i,0,...,:3].cpu().numpy())
        # plt.title(imname)
        # buf = io.BytesIO()
        # plt.savefig(buf,format='jpeg')
        # buf.seek()
        # plt.close()

        # plt.savefig(imname)
        # print(torch.unique(X))
        # img = Image.fromarray(X[i,0,...,:3].cpu().numpy().astype(np.uint8),"RGB")
        # img.save(imname)



    # surf = train_ds.getSurf(1)
    # name = train_ds.getName(1)

    # V_labels_prediction = torch.where(CL[0,:,1] == 255,1,0)
    # print(V_labels_prediction.shape)

    # V_labels_prediction = numpy_to_vtk(V_labels_prediction.numpy())
    # V_labels_prediction.SetName('Palete')
    # surf.GetPointData().AddArray(V_labels_prediction)

    # path_out = '/home/luciacev/Desktop/Data/IOSReg/ARON_GOLD/organize/test/seg/'
    # WriteSurf(surf,os.path.join(path_out,f'{name}_test_seg_reg.vtk'))

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/train_palete_denise.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/val_UL1O.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/test_UL1O.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Test/random_rotation")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/Denise/")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=4)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--property', help='label of segmentation', type=str, default="PredictedID")
    parser.add_argument('--landmark',help='name of landmark to found',default=['L3RM','R3RM'])
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/tensorboard')
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")





    args = parser.parse_args()
    main(args)

