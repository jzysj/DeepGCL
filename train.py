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
def train_con(model, device, drug1_loader,drug2_loader,drugs_loader,optimizer, epoch):
    model.train()
    metric = Accumulator(3)
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader, drug2_loader,drugs_loader)):
        train_loss=[]
        train_accs=[]
        data1,data2,data3=data_to_device(data)     
        y = data1.y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()

        out1,out2,out3,out4 = model(data1,data2,data3)
        loss =out1*a+loss_fn(out2, y)*b+loss_fn(out3, y)*c+d*loss_fn(out4, y)
        loss.backward()
        optimizer.step()
        y=y.cpu().numpy()
        ys = F.softmax(out2, 1).to('cpu').data.numpy()                
        predicted_labels =np.array( list(map(lambda x: np.argmax(x), ys)))         
        ACC=predicted_labels==y

        metric.add(
          float(loss) * len(y), ACC.sum(),
          len(ys))       
    return metric[0] / metric[2], metric[1]/ metric[2 ]
 
def valid_con(model, device, drug1_loader_test,drug2_loader_test,drugs_loader_test):
    model.eval()
    total_labels = torch.Tensor()
    total_labels1 = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_preds3 = torch.Tensor()
    total_preds2 = torch.Tensor()
    total_preds4 = torch.Tensor()
    metric = Accumulator(2)
    # print('Make valid prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test,drugs_loader_test):
            valid_loss=[]
            valid_accs=[]
            data1,data2,data3=data_to_device(data) 
            y = data1.y.view(-1, 1).long().to(device)
            y = y.squeeze(1)
            out1,out2,out3 ,out4= model(data1,data2,data3)

            loss =out1*a+loss_fn(out2, y)*b+loss_fn(out3, y)*c+d*loss_fn(out4, y)
            y=y.cpu().numpy()
            ys2 = F.softmax(out2, 1).to('cpu').data.numpy()#按行做归一化
            ys3 = F.softmax(out3, 1).to('cpu').data.numpy()#按行做归一化
            ys4 = F.softmax(out4, 1).to('cpu').data.numpy()#按行做归一化

            predicted_scores2 = list(map(lambda x: x[1], ys2))#预测分数         
            predicted_scores3 = list(map(lambda x: x[1], ys3))#预测分数
            predicted_scores4 = list(map(lambda x: x[1], ys4))#预测分数
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

            total_preds2 = torch.cat((total_preds2, torch.Tensor(predicted_scores2)), 0)
            total_preds3= torch.cat((total_preds3, torch.Tensor(predicted_scores3)), 0)
            total_preds4= torch.cat((total_preds4, torch.Tensor(predicted_scores4)), 0)
            metric.add(float(loss),len(ys2)) 
    AUC2 = roc_auc_score(total_labels.numpy().flatten(),total_preds2.numpy().flatten())#只有一种标签  
    AUC3= roc_auc_score(total_labels.numpy().flatten(),total_preds3.numpy().flatten())#只有一种标签  
    AUC4= roc_auc_score(total_labels.numpy().flatten(),total_preds4.numpy().flatten())#只有一种标签       
    return  AUC2,AUC3, AUC4
  
 def test_con(model, device, drug1_loader_test,drug2_loader_test,drugs_loader_test):
    model.eval()
    total_labels = torch.Tensor()
    total_labels1 = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_preds3 = torch.Tensor()
    total_preds2 = torch.Tensor()
    total_preds4 = torch.Tensor()
    total_out_features=torch.Tensor()
    total_prelabels2 = torch.Tensor()
    total_prelabels3 = torch.Tensor()
    total_prelabels4 = torch.Tensor()
    # print('Make valid prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test,drugs_loader_test):
            valid_loss=[]
            valid_accs=[]
            data1,data2,data3=data_to_device(data) 
            y = data1.y.view(-1, 1).long().to(device)
            out1,out2,out3 ,out4= model(data1,data2,data3)
#            out5=torch.cat([y,out5],dim=1).cpu()
            y = y.squeeze(1)
            

            loss =out1*a+loss_fn(out2, y)*b+loss_fn(out3, y)*c+d*loss_fn(out4, y)
            y=y.cpu().numpy()
            ys2 = F.softmax(out2, 1).to('cpu').data.numpy()#按行做归一化
            ys3 = F.softmax(out3, 1).to('cpu').data.numpy()#按行做归一化
            ys4 = F.softmax(out4, 1).to('cpu').data.numpy()#按行做归一化

            predicted_scores2 = list(map(lambda x: x[1], ys2))#预测分数         
            predicted_scores3 = list(map(lambda x: x[1], ys3))#预测分数
            predicted_scores4 = list(map(lambda x: x[1], ys4))#预测分数

            predicted_labels2 = list(map(lambda x: np.argmax(x), ys2))#预测的标签
            predicted_labels3 = list(map(lambda x: np.argmax(x), ys3))
            predicted_labels4 = list(map(lambda x: np.argmax(x), ys4))

            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

            total_preds2 = torch.cat((total_preds2, torch.Tensor(predicted_scores2)), 0)
            total_preds3= torch.cat((total_preds3, torch.Tensor(predicted_scores3)), 0)
            total_preds4= torch.cat((total_preds4, torch.Tensor(predicted_scores4)), 0)

            total_prelabels2=torch.cat((total_prelabels2, torch.Tensor(predicted_labels2)), 0)
            total_prelabels3=torch.cat((total_prelabels3, torch.Tensor(predicted_labels3)), 0)
            total_prelabels4=torch.cat((total_prelabels4, torch.Tensor(predicted_labels4)), 0)
           # total_out_features=torch.cat((total_out_features, out5), 0)
    AUPR2=average_precision_score(total_labels.numpy().flatten(), total_preds2.numpy().flatten())
    f1_score2=f1_score(total_labels.numpy().flatten(), total_prelabels2.numpy().flatten())
    AUC2 = roc_auc_score(total_labels.numpy().flatten(),total_preds2.numpy().flatten())#只有一种标签  

    AUC3= roc_auc_score(total_labels.numpy().flatten(),total_preds3.numpy().flatten())#只有一种标签  
    AUPR3=average_precision_score(total_labels.numpy().flatten(), total_preds3.numpy().flatten())
    f1_score3=f1_score(total_labels.numpy().flatten(), total_prelabels3.numpy().flatten())

    AUC4= roc_auc_score(total_labels.numpy().flatten(),total_preds4.numpy().flatten())#只有一种标签  
    AUPR4=average_precision_score(total_labels.numpy().flatten(), total_preds4.numpy().flatten())
    f1_score4=f1_score(total_labels.numpy().flatten(), total_prelabels4.numpy().flatten())
    return   AUC2,AUPR2,f1_score2,AUC3,AUPR3,f1_score3,AUC4,AUPR4,f1_score4
