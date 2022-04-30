import os
import argparse
import os
import argparse
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from util import *
from networks import *
from tqdm import tqdm
from ssim import SSIM

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--images_path', default='../../data/vi', help='Path to drive image')
    # parser.add_argument('--images_path', default='/data/Disk_A/fuyu/code/data/ir_vi', help='Path to drive image')
    parser.add_argument('--images_path', default='/data/Disk_B/MSCOCO2014/train2014', help='Path to drive image')

    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--eopch', default=1000, help='eopch')
    parser.add_argument('--learning_rate', default=0.0001, help='learning_rate')
    parser.add_argument('--load_weights', default=True, help='load weights')
    return parser.parse_args()

torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    print(args)
    startepoch = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DatasetFromFolder(args.images_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, drop_last=True)

    generator = Generator().to(device)
    # generator = torch.nn.DataParallel(generator)

    if args.load_weights:
        generator.load_state_dict(torch.load('./weights/G_eopch_last_l1.pth'))

    optimizer_gen = optim.Adam(generator.parameters(), lr=args.learning_rate)


    # optimizer_gen = torch.nn.DataParallel(optimizer_gen)
    MSE = nn.MSELoss()
    l1 = nn.L1Loss()
    ssim = SSIM()
    # vgg_loss = VGGLoss()
    loss_G=0
    datalen = len(dataloader)
    print(datalen)

    for epoch in range(startepoch,args.eopch):
        generator.train()
        data = tqdm(dataloader)
        for en, x in enumerate(data):
            x = x.to(device)
            # print(x.shape)
            # print(x, x.max(),x.mean())
            # img_ir = img_ir.to(device)
            fuse_re = generator(x)
            # print(fuse_re.shape,x.shape)
            optimizer_gen.zero_grad()

            l1loss = MSE(fuse_re, x)*10
            # l1loss = l1(fuse_re, x)
            ssim_loss = (ssim(fuse_re,x))
            ssim_loss = (1-ssim_loss)
            # print(fuse_image.shape)
            # loss_VGG = vgg_loss(fuse_image, img_vi,img_ir)
            # loss_G = l1loss+loss_VGG+ssim_loss
            loss_G = l1loss+ssim_loss

            loss_G.backward()
            optimizer_gen.step()
            # print("It %s:  Loss G:%.5f[%.5f %.5f %.5f] " % (
            # en, loss_G.item(), l1loss.item(), ssim_loss.item(), loss_VGG.item()))
            printstring = "epoch[%d][%d/%d] Loss G:%.5f [%.5f %.5f] " %(epoch,en+1, datalen,loss_G.item(),l1loss.item(),ssim_loss.item())
            data.set_description(printstring)

            if en%200==0:
                img = torchvision.utils.make_grid([x[0].cpu(),fuse_re[0].cpu()],nrow=2)
                save_image(img, fp=(os.path.join('output/img_train' + str(epoch) + '.jpg')))
            if en%10000==0:
                img = torchvision.utils.make_grid([x[0].cpu(),fuse_re[0].cpu()],nrow=2)
                save_image(img, fp=(os.path.join('output/img_train' + str(epoch) + '_en' + str(en) + '.jpg')))

        # with torch.no_grad():
        #     generator.eval()
        #     fake_test = generator(img_test_vi.unsqueeze(0),img_test_ir.unsqueeze(0))
        # save_image(fake_test, fp=(os.path.join('output/fake_img_test' + str(epoch) + '.jpg')))

        # print("Epoch %s:  Loss G:%.5f[%.5f %.5f %.5f] " %(epoch, loss_G.item(), l1loss.item(),ssim_loss.item(),loss_VGG.item()))

        if epoch%5==0:
            torch.save(generator,"weights/G_eopch_"+str(epoch)+".pth")
        if epoch % 1 == 0:
            torch.save(generator.state_dict(), "weights/G_eopch_last.pth")
    torch.save(generator,"weights/G_eopch_final.pth")










