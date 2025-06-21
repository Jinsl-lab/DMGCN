import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


        
class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Decoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(Decoder, self).__init__()
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU())
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nfeat),
            torch.nn.BatchNorm1d(nfeat),)
        
    def forward(self, h1):
        emb_z = self.decoder1(h1)
        x_rec = self.decoder2(emb_z)
        return emb_z,x_rec
        
class ZINB(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINB, self).__init__()
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, x):
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return pi, disp, mean


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
        
class Discriminator(nn.Module):            
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.FCN = nn.Bilinear(hidden_size, hidden_size, 1) 
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, emb1, emb2):
        c_x = c.expand_as(emb1)  
        sc_1 = self.FCN(emb1, c_x)
        sc_2 = self.FCN(emb2, c_x)

        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        
    def forward(self, emb, mask=None):
        vsum = torch.mm(mask.to_dense().float(), emb)
        row_sum = torch.sum(mask.to_dense(),dim=1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T 
        global_emb = vsum / row_sum 
        return nn.Sigmoid()(F.normalize(global_emb, p=2, dim=1))


        
class DMGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(DMGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CLGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.Decoder = Decoder(nfeat, nhid1, nhid2)

        self.ZINB = ZINB(nfeat, nhid1, nhid2)
        
        self.read = AvgReadout()
        self.Disc = Discriminator(nhid2)
        
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.att_d = Attention(2)
        self.MLP = nn.Linear(nhid2, nhid2)

    def forward(self, x, x_a, sadj, fadj):
        emb_f  = self.FGCN(x, fadj) 
        com_f  = self.CGCN(x, fadj)
        emb_s  = self.SGCN(x, sadj)
        com_s  = self.CGCN(x, sadj)
        emb_sa = self.CLGCN(x_a,sadj)
        emb_fa = self.CLGCN(x_a,fadj)

        h = torch.stack([emb_s,emb_f,(com_s + com_f)/2], dim=1)
        emb,_ = self.att(h)
        emb = self.MLP(emb)
        emb_z, x_rec = self.Decoder(emb)
        
        pi, disp, mean = self.ZINB(emb_z)

        g_s,g_sa = self.read(com_s, sadj), self.read(emb_sa, sadj)
        g_f,g_fa = self.read(com_f, fadj), self.read(emb_fa, fadj)
        
        ret_s,ret_sa = self.Disc(g_s, com_s, emb_sa),self.Disc(g_sa, emb_sa, com_s)
        ret_f,ret_fa = self.Disc(g_f, com_f, emb_fa),self.Disc(g_fa, emb_fa, com_f)

        ret_s,_ = self.att_d(torch.stack([ret_s,ret_sa],dim=1))
        ret_f,_ = self.att_d(torch.stack([ret_f,ret_fa],dim=1))
        
        # return com_s, com_f, emb, pi, disp, mean, x_rec, ret_s, ret_f
        return com_s, com_f, emb, pi, disp, mean, ret_s, ret_f       
