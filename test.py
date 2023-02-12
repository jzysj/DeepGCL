

def data_to_device(data):
 
      data1 = data[0]
      data2 = data[1]
      # print(batch_idx)
      features1=torch.tensor([])
      features2=torch.tensor([])
      for e in data1.x_index:
          features1=torch.cat((features1,smile_graph[e][1]),0)

      data1.x=features1
      for e in data2.x_index:
          features2=torch.cat((features2,smile_graph[e][1]),0)
      data2.x=features2  
      data1 = data1.to(device)
      data2 = data2.to(device)

      data3 = data[2]
      nodes_index=data3.x.reshape(-1)# 读取所有的节点序号
      features_cat=torch.cat((features,data3.node),dim=0)#拼接特征
      mint=torch.LongTensor((nodes_index==-1).nonzero()).reshape(-1)#取出-1索引的位置
      node_index = torch.arange(sample_len, sample_len+Batch_size)#生成中心节点位置
      nodes_index.scatter_(0,mint,node_index)#更新中心节点的索引
      data3.x=features_cat[nodes_index] 
      data3.edge_index=data3.edge_index.type(torch.long)
      data3 = data3.to(device)
      return data1,data2,data3

encoder=Encoder(nfeat=78,embed_dim=128,dropout=0.2,layer1=layer1).to(device)
encoder2=Encoder2(nfeat=166,embed_dim=128,dropout=0.5,layer2=layer2).to(device)
contrastiveLoss=ContrastiveLoss(batch_size=24,device='cuda',temperature=0.5).to(device)
model=GCNNet(encoder,encoder2,contrastiveLoss,nfeat=128,outdim=2,dropout=0.2).to(device)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
    
a=0.1
b=1
c=1
d=1
Batch_size=24
LR = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
layer1=3
layer2=2
loader_train12 = DataLoader(traindataset_11, batch_size=Batch_size, shuffle=False,drop_last =True)
loader_valid12 = DataLoader(validdataset_11, batch_size=Batch_size, shuffle=False,drop_last =True)
loader_test12 = DataLoader(testdataset_11, batch_size=Batch_size, shuffle=False,drop_last =True)

loader_train1 = DataLoader(traindataset1, batch_size=Batch_size, shuffle=False,drop_last =True)
loader_train2 = DataLoader(traindataset2, batch_size=Batch_size, shuffle=False,drop_last =True)

loader_valid1 = DataLoader(validdataset1, batch_size=Batch_size, shuffle=False,drop_last =True)
loader_valid2 = DataLoader(validdataset2, batch_size=Batch_size, shuffle=False,drop_last =True)

loader_test1 = DataLoader(testdataset1, batch_size=Batch_size, shuffle=False,drop_last =True)
loader_test2 = DataLoader(testdataset2, batch_size=Batch_size, shuffle=False,drop_last =True)
auc_list=[]
auc_list1=[]
auc_list2=[]
best_auc2=0
best_auc3=0
best_auc4=0
acc_list=[]
R=[]
max_auc=0
model_path = './BioSnapData/jieguo/pre_res_model.ckpt'
NUM_EPOCHS=100
animator = Animator(xlabel='epoch', xlim=[1, NUM_EPOCHS], ylim=[0.0, 1.0],
                    legend=[ 'AUC2','AUC3','AUC4'])
for epoch in range(NUM_EPOCHS):
    train_metrics=train_con(model, device, loader_train1,loader_train2,loader_train12,  optimizer, epoch+1)
    test_acc=valid_con(model, device, loader_valid1,loader_valid2,loader_valid12)
    if best_auc2<test_acc[0]:        
        best_auc2=test_acc[0]

    if best_auc3<test_acc[1]:
        best_auc3=test_acc[1]

    if best_auc4<test_acc[2]:
        best_auc4=test_acc[2]
#保存分子图部分结果

    if best_auc2 > max_auc:
        max_auc=best_auc2
        model_max = copy.deepcopy(model)
        torch.save(model_max.state_dict(),model_path)
#           print(epoch + 1,  test_acc)
print(layer1,layer2)
print(train_metrics,test_acc)
print(layer1,layer2)




test_acc=test_con(model_max, device, loader_test1,loader_test2,loader_test12)
print(test_acc)
