def load_data(G, val_ratio, test_ratio):#验证比例，测试比例
  
    """Read data from path, convert data into loader"""

    fileName = './data/ChCh-Miner_durgbank-chem-chem.tsv'
    train_data = []
    with open(fileName, "r") as f:
        for line in f.readlines():
          
            if line.strip().split("\t")[0] in sample_drug and line.strip().split("\t")[1] in sample_drug:
                if [sample_drug.index(line.strip().split("\t")[0]),sample_drug.index(line.strip().split("\t")[1])] in G.edges():
                    train_data.append([sample_drug.index(line.strip().split("\t")[0]),sample_drug.index(line.strip().split("\t")[1])])
    # sample negative
    negative_all = list(nx.non_edges(G))#构建负样本

    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:len(train_data)])
    print("positve examples: %d, negative examples: %d." % (len(train_data), len(negative)))

    # split data
    val_ratio=0.1
    test_ratio=0.1
    val_size = int(val_ratio * len(train_data))#验证比例
    test_size = int(test_ratio * len(train_data))#测试比例

    train_data = np.concatenate([train_data, np.ones(len(train_data), dtype=np.int64).reshape(len(train_data), 1)], axis=1)#加上标签
    negative = np.concatenate([negative, np.zeros(len(train_data), dtype=np.int64).reshape(len(train_data), 1)], axis=1)
    np.random.shuffle(train_data)
    np.random.shuffle(negative)
    train_data1 = np.vstack((train_data[: -(val_size + test_size)], negative[: -(val_size + test_size)]))#拼接成训练数据
    val_data = np.vstack((train_data[-(val_size + test_size): -test_size], negative[-(val_size + test_size): -test_size]))#拼接成验证数据
    test_data = np.vstack((train_data[-test_size:], negative[-test_size:]))#拼接测试数据

    np.random.shuffle(train_data1)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    print('Loading finished!')
    
    return train_data1, val_data, test_data
sample_drug=[]
with open('./BioSnapData/bio_drug.txt','r') as f:
    for i in f.readlines():      
      sample_drug.append(i.strip())
print(len(sample_drug))
train_data, valid_data, test_data = load_data(G,val_ratio=0.1, test_ratio=0.1)
