import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

#BatchNormNode 類別對節點特徵（Node Features）進行批次正規化。批次正規化有助於加速訓練並提升模型穩定性
class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """
    #__init__ 方法初始化了一個 BatchNorm1d 層，用來進行一維的批次正規化，維度為 hidden_dim
    def __init__(self, hidden_dim): 
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    
    #forward 方法接受節點特徵 x，其形狀為 (batch_size, num_nodes, hidden_dim)(F*n*H)
    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        #將維度順序調整為 (batch_size, hidden_dim, num_nodes) 以符合 BatchNorm1d 的輸入需求
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans) #正規化
        #進行正規化後，再將張量轉回原始形狀 (batch_size, num_nodes, hidden_dim)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn

#BatchNormEdge 類別對邊特徵（Edge Features）進行批次正規化
class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    #BatchNorm2d 會針對兩個維度的特徵進行正規化，常用於處理圖卷積網路的邊特徵
    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    #forward 方法的邊特徵 e 維度為 (batch_size, num_nodes, num_nodes, hidden_dim)
    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        #經過 transpose 轉換形狀後進行正規化，再轉回原始維度
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn

#NodeFeatures 類別用於計算節點特徵，並根據 sum 或 mean 的聚合方式進行卷積操作
##初始化兩個全連接層 U 和 V 用於節點的特徵轉換
class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
  
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
    
    #將節點特徵 x 經 U 和 V 轉換。Vx 經擴展以便與邊特徵進行門控運算 
    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        
        #根據聚合模式計算新的節點特徵。mean 模式下使用 edge_gate 的和作為平均值分母
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new

#EdgeFeatures 類別用於卷積邊特徵，並根據節點特徵來更新邊特徵
##將邊特徵 e 經 U 映射，將節點特徵經 V 映射，並透過擴展運算以便形狀匹配
class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new
        #最終將映射後的特徵進行加和，以生成更新的邊特徵

#ResidualGatedGCNLayer 類別結合門控和殘差連接來進行圖卷積操作
class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """
    #建立節點和邊特徵層以及批次正規化層
    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
