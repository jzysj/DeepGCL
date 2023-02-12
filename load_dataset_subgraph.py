from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch
from scipy import sparse
from scipy import stats
import networkx as nx
from tqdm import tqdm
def update_G(G):
    edge_list=[]
    for n in G.nodes:
        edge_list.append((n,-1))
    G.add_edges_from(edge_list)       
    return G
  
  
def ciyuan(id):
    smiles=drug_sm_dic[id]
    m1 = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=166)
    finger=fp1.ToBitString()    
    fingers=list(map(int,finger))
    return fingers  
  
  
class TestbedDataset2(InMemoryDataset):
    def __init__(self, root='/temp',dataset='_drug1',
                 xd=None,G=None,features=None,G1=None,G2=None):
        super(TestbedDataset2, self).__init__(root)
        self.dataset=dataset
        self.xd=xd
        self.features=features
        self.G=torch.tensor(nx.adjacency_matrix(G).todense(),dtype=torch.short)#每个图的邻接矩阵有反应
        self.G1=G1
        self.G2=G2
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd)
            self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    #处理后的文件保存路径
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')     
    @property
    def raw_file_names(self):   
#         pass
        return ['test.csv']
        #raw_file_names 这个函数给出多张graph所存的路径，假设 graph a ,graph b，这里return就是两幅图对应的文件名
        
    def download(self):
        pass
            #下载数据集，本地有就Pass
    @property
    def processed_file_names(self):      
        return [self.dataset+'.pt']        
    def process(self, xd):
        
        data_list = []
        data_len=len(xd)
        for i in tqdm(range(data_len)):         
            dru1 = xd[i][0]#药物对 A-B
            dru2 = xd[i][1]
            labels = xd[i][2]
            c=[1306,dru1,dru2]
            arr1=self.G1.indices[ self.G1.indptr[dru1]:  self.G1.indptr[dru1+1] ]
            arr2=self.G1.indices[ self.G1.indptr[dru2]:  self.G1.indptr[dru2+1] ]
            ad1=np.union1d(arr1, arr2)#取并集
            ad1=np.union1d(ad1,c)


            arr1=self.G2.indices[self.G2.indptr[dru1]: self.G2.indptr[dru1+1] ]
            arr2=self.G2.indices[self.G2.indptr[dru2]: self.G2.indptr[dru2+1] ]
            ad2=np.intersect1d(arr1, arr2)#求交集 

            nodes=torch.LongTensor(np.union1d(ad1, ad2)).reshape(-1)

            matrix=torch.index_select(self.G, 1, nodes) 
            matrix=torch.index_select(matrix, 0, nodes)
          
            # ad=ladies_sampler3(matrix)#采样的邻接矩阵
            ad=torch.nonzero(matrix==1).transpose(1, 0).short()#计算所有邻接矩阵索引
            node_index=nodes[:len(nodes)-1] 
            # node_index=torch.LongTensor([dru1,dru2])
            node=(self.features[node_index].sum(dim=0))/len(node_index)#计算中心节点的特征
            nodes[-1]=-1
            GCNData = DATA.Data(x=nodes,                         
                        node=torch.FloatTensor(node).unsqueeze(0),
                                edge_index=ad,
                        y=torch.LongTensor([labels])
                               )
            data_list.append(GCNData)
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)  
  
G_train = update_G(G)
features=torch.tensor([])#把选择出来的药物构成n*1024的矩阵
for drug in sample_drug:
    feature=torch.tensor(ciyuan(drug))
    features= torch.cat((features, feature), 0)#按行拼接
features=features.reshape(-1,166)
        datafile="test"

testdataset_11=TestbedDataset2(root='./BioSnapData/data', dataset=datafile + 'net_test_int', xd=test_data,G=G_train,features=features,G1=A,G2=A2)
traindataset_11=TestbedDataset2(root='./BioSnapData/data', dataset=datafile + 'net_train_int', xd=train_data, G=G_train,features=features,G1=A,G2=A2)
validdataset_11=TestbedDataset2(root='./BioSnapData/data', dataset=datafile + 'net_valid_int', xd=valid_data, G=G_train,features=features,G1=A,G2=A2)

