import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.cuda.amp import autocast, GradScaler

class Encoder(torch.nn.Module):
    def __init__(self,nfeat,embed_dim,dropout,layer1):
        super(Encoder,self).__init__()
        self.layer1=layer1
        self.drug_conv1 = GCNConv(nfeat, nfeat)#78-78
        self.drug_conv2 = GCNConv(nfeat, nfeat*2)#78-156
        self.drug_conv3 = GCNConv(nfeat*2, nfeat * 4)#156-312
        self.drug_conv4 = GCNConv(nfeat, nfeat * 4)#156-312
        self.drug_fc_g1 = torch.nn.Linear(nfeat*4, nfeat*2)
        self.drug_fc_g2 = torch.nn.Linear(nfeat*2, embed_dim)#128
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,data):
        x, edge_index, batch,ptr =data.x, data.edge_index, data.batch, data.ptr
        if self.layer1==1:
            x = self.drug_conv4(x, edge_index)
            x = self.relu(x)
        elif self.layer1==2:    
            x = self.drug_conv2(x, edge_index)
            x = self.relu(x)
            x = self.drug_conv3(x, edge_index)
            x = self.relu(x)
        elif self.layer1==3:
            x = self.drug_conv1(x, edge_index)
            x = self.relu(x)            
            x = self.drug_conv2(x, edge_index)
            x = self.relu(x)
            x = self.drug_conv3(x, edge_index)
            x = self.relu(x)

        
        x = gmp(x, batch)       # global max pooling
        x = self.relu(self.drug_fc_g1(x))
        x = self.dropout(x)
        x = self.relu(self.drug_fc_g2(x))
        x = self.dropout(x)
        return x
      
class ContrastiveLoss(torch.nn.Module):
    def __init__(self,batch_size,device='cuda',temperature=0.5):
        super(ContrastiveLoss,self).__init__()
        self.batch_size=batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())# 主对角线为0，其余位置全为1的mask矩阵
    def forward(self,emb_i,emb_j):
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
  class Encoder2(torch.nn.Module):
    def __init__(self,nfeat,embed_dim,dropout,layer2):
        super(Encoder2,self).__init__()
        self.layer2=layer2
        self.drugs_conv1 = GCNConv(nfeat, nfeat)#156
        self.drugs_conv2 = GCNConv(nfeat ,nfeat*2 )#312
        self.drugs_conv3 = GCNConv(nfeat * 2, nfeat * 4)#624
        self.drugs_conv4 = GCNConv(nfeat , nfeat * 4)#624
        self.drugs_fc_g1 = torch.nn.Linear(nfeat * 4, embed_dim*2)#312
        # self.drugs_fc_g2 = torch.nn.Linear(num_features_xd1*2, num_features_xd1)#156
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,data):
        x, edge_index, batch,ptr =data.x, data.edge_index, data.batch, data.ptr
        if self.layer2==1:
            x = self.drugs_conv4(x, edge_index)
            x = self.relu(x)
        elif self.layer2==2:    
            x = self.drugs_conv2(x, edge_index)
            x = self.relu(x)
            x = self.drugs_conv3(x, edge_index)
            x = self.relu(x)
        elif self.layer2==3:
            x = self.drugs_conv1(x, edge_index)
            x = self.relu(x)            
            x = self.drugs_conv2(x, edge_index)
            x = self.relu(x)
            x = self.drugs_conv3(x, edge_index)
            x = self.relu(x)
        
        index=torch.tensor([i-1 for i in ptr[1:]])#得要一个batch每一个图的中心节点的位置
        x=x[index]      
        x = self.drugs_fc_g1(x)
        x=self.relu(x)
        x=self.dropout(x)
        return x
class GCNNet(torch.nn.Module):
    def __init__(self,encoder,encoder2,contrastiveLoss,nfeat=128,outdim=2,dropout=0.5):
        super(GCNNet,self).__init__()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout()
        self.encoder=encoder
        self.encoder2=encoder2
        self.contrastiveLoss=contrastiveLoss
        self.fc1 = torch.nn.Linear(nfeat*2, 512)#定义损失1
        self.fc2 = nn.Linear(512*4, 512)
        self.fc3 = torch.nn.Linear(512, 2)#定义损失1

        self.fc_1 = torch.nn.Linear(nfeat*2, 512)#定义损失1
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = torch.nn.Linear(128, 2)#定义损失1

        self.fc_2_2 = nn.Linear(512, 128)
        self.fc_3_3 = torch.nn.Linear(128, 2)#定义损失1
    def forward(self,data1,data2,data3):
        x1=self.encoder(data1)
        x2=self.encoder(data2)
        x3=self.encoder2(data3)

        #cat x1,x2
        xc = torch.cat((x1, x2), 1)
        xc= self.fc1(xc)
        xc=self.relu(xc)
        xc=self.dropout(xc)#512

        x3= self.fc_1(x3)
        x3=self.relu(x3)
        x3=self.dropout(x3)
        loss=self.contrastiveLoss(xc,x3)#512
        add=x3+xc
        prouduct=torch.mul(x3,xc)
        concatenate=torch.cat((x3, xc),dim=1)
        feature=torch.cat((add,prouduct,concatenate),dim=1)
#        out5=feature
        feature= self.fc2(feature)
        feature=self.relu(feature)
        feature=self.dropout(feature)

        out2= self.fc3(feature)
        x3= self.fc_2(x3)
        x3=self.relu(x3)
        x3=self.dropout(x3)
        out4= self.fc_3(x3)

        xc= self.fc_2_2(xc)
        xc=self.relu(xc)
        xc=self.dropout(xc)
        out3= self.fc_3_3(xc)
        return loss,out2,out3,out4
