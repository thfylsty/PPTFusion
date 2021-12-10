import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
import torch.nn.init as init
from util import *
from torchvision.utils import save_image
from torch.nn.modules.utils import _pair
from Transformer import *
from functions import *
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
        print(position_ids.shape)
        position_embeddings = self.pe(position_ids)
        print(position_embeddings.shape)
        return x + position_embeddings

class Attention(nn.Module):
    def __init__(self,hidden_size ):
        super(Attention, self).__init__()
        self.num_attention_heads = 1
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print("query_layer", query_layer.shape)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print("attention_scores", attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, hidden_size,mlp_dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransBlock(nn.Module):
    def __init__(self, hidden_size,mlp_dim,):
        super(TransBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size,mlp_dim)
        self.attn = Attention(hidden_size)

    def forward(self, x):
        h = x
        # print("a1",x.shape)
        x = self.attention_norm(x)
        # print("a2", x.shape)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Block(nn.Module):
    def __init__(self,):
        super(Block, self).__init__()
        # self.soft_split = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.soft_split = nn.Unfold(kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.position_encoding = LearnedPositionalEncoding(16384, 25, 16384)
        self.attention=CrissCrossAttention(49)

    def forward(self, x):
        x = self.soft_split(x).transpose(1, 2)
        x = self.position_encoding(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, 128, 128)
        x = self.attention(x)

        return x

class Patch_transformer(nn.Module):
    def __init__(self,img_dim=256,patch_dim=32,embedding_dim=16):
        super(Patch_transformer, self).__init__()
        self.img_dim = img_dim
        self.num_patches_sqrt = img_dim // patch_dim
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.flatten_dim = patch_dim * patch_dim
        self.patch_dim = patch_dim
        # print("self.num_patches",self.num_patches)
        self.position_parameter = nn.Parameter(torch.zeros(self.num_patches, 1024, 16))
        self.linear_encoding = nn.Linear(1, embedding_dim)
        self.pe_dropout = nn.Dropout(p=0.1)
        self.transformer = nn.Sequential(
            TransBlock(hidden_size=16, mlp_dim=3072),
            TransBlock(hidden_size=16, mlp_dim=3072),
            TransBlock(hidden_size=16, mlp_dim=3072),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def recon(self,x):
        for j in range(0, self.num_patches_sqrt):
            for i in range(0, self.num_patches_sqrt):
                t = x[:, i + j * self.num_patches_sqrt:i + 1 + j * self.num_patches_sqrt, :]
                t = t.view(t.shape[0], 1, 32, 32)
                # print(t.shape)
                if i == 0:
                    lines = t
                else:
                    lines = torch.cat((lines, t), 3)
            # print("lines",lines.shape)
            if j == 0:
                fea = lines
            else:
                fea = torch.cat((fea, lines), 2)
        return fea

    def forward(self, x):
        # print("patch trans",x.shape)
        n, c, h, w = x.shape
        # print(x.shape)
        x = (
            x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
        )
        # print(x.shape)
        x = x.view(n, c, -1, self.patch_dim ** 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape)
        x = x.view(x.size(0), -1, self.flatten_dim)
        # print(x.shape)
        x = x.permute(1, 2, 0).contiguous()
        # print(x.shape)
        x = self.linear_encoding(x)
        # print("p",x.shape)
        x = x + self.position_parameter
        x = self.pe_dropout(x)
        # print(x.shape)
        x = self.transformer(x)
        # print("6",x.shape)
        x = self.norm(x)
        # print("patch trans", x.shape)
        x = x.permute(2, 0, 1).contiguous()
        x = self.recon(x)
        x = x.permute(1, 0, 2, 3).contiguous()
        # print("patch trans", x.shape)

        return x


class Encoder(nn.Module):
    def __init__(self,img_dim=128,patch_dim=1,num_channels=1,embedding_dim=16,dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.dropout_rate = dropout_rate
        self.flatten_dim = patch_dim * patch_dim * num_channels

        # self.position_encoding = LearnedPositionalEncoding(
        #     self.seq_length, self.embedding_dim, self.seq_length
        # )
        # self.position_parameter = nn.Parameter(torch.zeros(64, 1024,16))
        # self.position_parameter_d = nn.Parameter(torch.zeros(1, 1024, 16))
        # self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        # self.linear_encoding = nn.Linear(1, embedding_dim)
        # self.linear_encoding_r = nn.Linear(embedding_dim, 1)
        # self.linear_encoding_d = nn.Linear(1, embedding_dim)
        # self.attn_dropout_rate = 0.1
        # self.transformer = TransBlock(hidden_size=16,mlp_dim=3072)
        # self.transformer_d = TransBlock(hidden_size=16, mlp_dim=3072)
        # self.norm = nn.LayerNorm(embedding_dim)
        # self.norm_d = nn.LayerNorm(embedding_dim)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.upsample_4 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=None)
        self.upsample_8 = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=None)

        self.patch_transformer = Patch_transformer()
        self.patch_transformer_2 = Patch_transformer(img_dim=128)
        self.patch_transformer_4 = Patch_transformer(img_dim=64)
        self.patch_transformer_8 = Patch_transformer(img_dim=32)


    def forward(self, input):
        x = input
        # ===============
        x_2 = self.pool(input)
        x_4 = self.pool(x_2)
        x_8 = self.pool(x_4)
        # print("xxx", x.shape, x_2.shape, x_4.shape, x_8.shape)

        x = self.patch_transformer(x)
        x_2 = self.patch_transformer_2(x_2)
        x_4 = self.patch_transformer_4(x_4)
        x_8 = self.patch_transformer_8(x_8)

        x_2 = self.upsample_2(x_2)
        x_4 = self.upsample_4(x_4)
        x_8 = self.upsample_8(x_8)
        # print("xxx", x.shape, x_2.shape,x_4.shape,x_8.shape)

        x = torch.cat((torch.cat((x, x_2), 1),torch.cat((x_4, x_8), 1)), 1  )
        # print("=========")
        return x


class Decoder(nn.Module):
    def __init__(self,img_dim=128,patch_dim=1,embedding_dim=1024):
        super(Decoder, self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim

        self.linear_1 = nn.Linear(2048, 4096)
        self.linear_2 = nn.Linear(4096, 16384)

        # self.deconv = nn.Sequential(
        #     nn.Conv2d(32, 8, 1, 1, 0),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 1, 1, 1, 0),
        #     nn.ReLU(),
        #     nn.Tanh()
        # )

        self.deconv = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.GELU(),
            nn.Tanh()
        )

        self.decoder_mlp = nn.Linear(64, 1)


    def forward(self, x):
        x = x.view(1, 64, -1)
        x = x.permute(0, 2, 1).contiguous()
        # print("t", t.shape)
        x = self.deconv(x)
        # print("t", t.shape)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(1, 1, 256, 256)

        # print("d", x.shape)
        return x


class Generator(nn.Module):
    def __init__(self,img_dim=256,patch_dim=32,num_channels=1,embedding_dim=16,dropout_rate=0.1,mode="train"):
        super(Generator, self).__init__()
        self.encoder = Encoder(img_dim=img_dim,patch_dim=patch_dim,num_channels=num_channels,
                               embedding_dim=embedding_dim,dropout_rate=dropout_rate)
        self.decoder = Decoder(img_dim=img_dim,patch_dim=patch_dim,embedding_dim=embedding_dim)

        if mode == "train":
            self.forward_function = self.train_forward
        else:
            self.forward_function = self.test_forward


    def train_forward(self,x,tmp=None):
        # if not tmp==None:
        #     print(">????????")
        x = self.encoder(x)
        out = self.decoder(x)
        return out


    def outputfeatures(self,x,name):
        InstanceNorm2d = nn.InstanceNorm2d(3)

        # output features
        unloader = transforms.ToPILImage()
        print(x.shape)
        for i in range(64):
            x_t = x[:, i:i + 1, :, :].cpu().clone()
            # print(x_t.shape)
            x_t = InstanceNorm2d(x_t)
            x_t = x_t.squeeze(0)
            # print(x_t.shape)
            features = unloader(x_t)
            # print(features.shape)
            features.save(os.path.join('features/'+name+'_' + str(i + 1) + '.bmp'))

    def test_forward(self,x,y):
        x = self.encoder(x)
        y = self.encoder(y)

        # self.outputfeatures(x,'x')
        # self.outputfeatures(y, 'y')

        # z = (x+y)/2
        # mean_x = torch.mean(torch.mean(x,2),2)
        # mean_y = torch.mean(torch.mean(y, 2), 2)
        # mean_x_e = torch.exp(mean_x).unsqueeze(2).unsqueeze(3)
        # mean_y_e = torch.exp(mean_y).unsqueeze(2).unsqueeze(3)
        # mean_sum = mean_x_e+mean_y_e
        # x = x * (mean_x_e/mean_sum)
        # y = y * (mean_y_e / mean_sum)

        # z = (x + y) / 2
        # z = x+y
        # print("===add===")
        # z = torch.max(x,y)
        # print("===max===")
        ex = torch.exp(x)
        ey = torch.exp(y)
        esum = ex + ey
        z = x * ex/esum + y * ey/esum
        # print("===softmax===")
        # self.outputfeatures(z, 'z')
        # exit()
        out = self.decoder(z)
        return out

    def forward(self, input,tmp=None,):

        return self.forward_function(input,tmp  )

