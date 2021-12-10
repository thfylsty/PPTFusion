import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image,ImageOps
import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image
import time 
from networks import Generator

def mkdir(pathname):
    try:
        os.mkdir(pathname)
    except:
        pass


def topatch(img):
    w, h = img.size
    img_pad = ImageOps.expand(img, (0, 0, 256 - w % 256, 256 - h % 256), fill=128)

    nh = (h // 256 + 1)
    nw = (w // 256 + 1)

    cis = []
    for j in range(nh):
        for i in range(nw):
            area = (256 * i, 256 * j, 256 * (i + 1), 256 * (j + 1))
            cropped_img = img_pad.crop(area)
            cis.append(cropped_img)

    return cis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(mode='test').to(device)
pth_path = "./G_eopch_20.pth"
generator.load_state_dict(torch.load(pth_path))
generator.eval()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in generator.parameters())))

dataname = "road" #road TNO
pictype =".jpg" # jpg  bmp
pic_num = 50

unloader = transforms.ToPILImage()

if not os.path.exists("result"):
    os.makedirs("result")

if not os.path.exists("./result/%s"%dataname):
    os.makedirs("./result/%s"%dataname)

with torch.no_grad():
    begin_time = time.time()
    for idx in range(pic_num):
        
        img_test_vi = Image.open('./'+dataname+'/vi/'+str(idx+1)+pictype).convert('L')
        img_test_vi_cis = topatch(img_test_vi)

        img_test_ir = Image.open('./'+dataname+'/ir/'+str(idx+1)+pictype).convert('L')
        img_test_ir_cis = topatch(img_test_ir)


        w, h = img_test_vi.size
        nh = (h // 256 + 1)
        nw = (w // 256 + 1)

        n = 0
        fuse_cis = []
        start_time = time.time()
        for pic_i in range(nh*nw):
            img_test_vi = transforms.ToTensor()(img_test_vi_cis[pic_i]).to(device)
            # print(img_test_ir_cis[pic_i])
            img_test_ir = transforms.ToTensor()(img_test_ir_cis[pic_i]).to(device)

            fuse_test = generator(img_test_vi.unsqueeze(0), img_test_ir.unsqueeze(0))

            fuse_img = fuse_test.cpu().clone()
            fuse_img = fuse_img.squeeze(0)
            fuse_img = unloader(fuse_img)
            fuse_cis.append(fuse_img)

        result_img = Image.new('L', (w, h))
        for j in range(nh):
            for i in range(nw):
                result_img.paste(fuse_cis[n], (256 * i, 256 * j))
                n = n + 1
        end_time = time.time()
        save_path = "./result/%s/%d.png"%(dataname,idx)
        result_img.save(save_path)
        print("idx: %d, time: %f, save %s:"%(idx,end_time-start_time,save_path))
    print("nums: %d, time avg: %f"%(pic_num,(time.time()-begin_time)/pic_num))
    del generator