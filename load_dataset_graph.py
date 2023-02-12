from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
import torch
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])



  
  
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # mol=Chem.AddHs(mol)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
#         print(feature / sum(feature))
        features.append(feature / sum(feature))
#     print(features)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     print(edges)有方向
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])#无方向
    features=torch.FloatTensor(features)
    return c_size, features, edge_index#得到原子个数，每个原子的特征，边的索引
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
            # ad1=np.union1d(ad1,ad2)
            # arr1=self.G3.indices[self.G3.indptr[dru1]: self.G3.indptr[dru1+1] ]
            # arr2=self.G3.indices[self.G3.indptr[dru2]: self.G3.indptr[dru2+1] ]
            # ad3=np.intersect1d(arr1, arr2)#求交集

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


class TestbedDataset3(InMemoryDataset):
    def __init__(self, root='/temp',dataset='_drug1',
                xd=None, y=None,transform=None,
                pre_transform=None, smile_graph=None):
        super(TestbedDataset3, self).__init__(root, transform, pre_transform)
    #         self.dataset=dataset

      #保存处理好的数据文件，文件存储在processed_path属性方法返回的文件路径
      #processed_paths属性方法在基类中定义，对self.processed_dir文件与
      #processed_file_names属性方法的返回的每一个文件名做一个拼接，然后返回
        self.dataset = dataset
        print(self.processed_paths[0])#data\processed\new_labels_0_10_drug1.pt
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y)
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
        return ['drug_interaction_sample.csv']
      #raw_file_names 这个函数给出多张graph所存的路径，假设 graph a ,graph b，这里return就是两幅图对应的文件名
      
    def download(self):
        pass
           #下载数据集，本地有就Pass
    @property
    def processed_file_names(self):      
        return [self.dataset+'.pt']    
  
    def process(self, xd,  y):
        assert (len(xd) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in tqdm(range(data_len)):
          # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            labels = y[i]
          # convert SMILES to molecular representation using rdkit
            c_size, f, edge_index = smile_graph[smiles]
          # make the graph ready for PyTorch Geometrics GCN algorithms:
            features=torch.zeros(len(f),1)
            GCNData = DATA.Data(x=torch.Tensor(features),
                              x_index=smiles,
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                              y=torch.LongTensor([labels]))
 
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def get_cell_feature(self,cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if row[0] == cellId:
                return row[1: ]
        return False
sample_len=len(sample_drug)
smile_graph={}
for k in range(sample_len):
    drug_id=sample_drug[k]
    g = smile_to_graph(drug_sm_dic[drug_id])
    smile_graph[k]=g
    
train_drug1=train_data[:,0]
train_drug2=train_data[:,1]
valid_drug1=valid_data[:,0]
valid_drug2=valid_data[:,1]
test_drug1=test_data[:,0]
test_drug2=test_data[:,1]
train_label=train_data[:,2]
valid_label=valid_data[:,2]
test_label=test_data[:,2]

datafile="drug_interaction_sample"
traindataset1=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_train1', xd=train_drug1 , y=train_label)
traindataset2=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_train2', xd=train_drug2 , y=train_label)
validdataset1=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_valid1', xd=valid_drug1 , y=valid_label)
validdataset2=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_valid2', xd=valid_drug2 , y=valid_label)
testdataset1=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_test1', xd=test_drug1 , y=test_label)
testdataset2=TestbedDataset3(root='./BioSnapData/data', dataset=datafile + 'desc_test2', xd=test_drug2 , y=test_label)
print('创建数据成功')
