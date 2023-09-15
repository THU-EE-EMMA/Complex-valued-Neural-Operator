import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import NNConv
import torch.nn as nn
import torch.nn.functional as F

class CVNeuralOpKernel(torch.nn.Module):
    def __init__(self, transchannel, ker_width, edgefeasize):
        super(CVNeuralOpKernel, self).__init__()

        self.realaggr = nn.Sequential(nn.Linear(edgefeasize, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, transchannel**2))
        self.realconv = NNConv(transchannel, transchannel, self.realaggr, aggr='mean')
        self.imagaggr = nn.Sequential(nn.Linear(edgefeasize, ker_width), nn.PReLU(),
                                      nn.Linear(ker_width, ker_width), nn.PReLU(),
                                      nn.Linear(ker_width, ker_width), nn.PReLU(),
                                      nn.Linear(ker_width, transchannel ** 2))
        self.imagconv = NNConv(transchannel, transchannel,  self.imagaggr, aggr='mean')

        self.nonlinearr = nn.PReLU()
        self.nonlineari = nn.PReLU()

    def forward(self, xr, xi, edge_index, edge_attr):
        rr = self.realconv(xr, edge_index, edge_attr)
        ri = self.realconv(xi, edge_index, edge_attr)
        ir = self.imagconv(xr, edge_index, edge_attr)
        ii = self.imagconv(xi, edge_index, edge_attr)
        return self.nonlinearr(rr-ii), self.nonlineari(ri+ir)

class CVNeuralOp(torch.nn.Module):
    def __init__(self, inchannel, transchannel, outchannel, ker_width, edgefeasize):
        super(CVNeuralOp, self).__init__()
        self.onestep1 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep2 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep3 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep4 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep5 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep6 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep7 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep8 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep9 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)
        self.onestep10 = CVNeuralOpKernel(transchannel,  ker_width, edgefeasize)

        self.fcup1r = nn.Linear(inchannel, transchannel//2)
        self.fcup1i = nn.Linear(inchannel, transchannel//2)
        self.NLup1r = nn.PReLU()
        self.NLup1i = nn.PReLU()

        self.fcup2r = nn.Linear(transchannel//2, transchannel)
        self.fcup2i = nn.Linear(transchannel//2, transchannel)
        self.NLup2r = nn.PReLU()
        self.NLup2i = nn.PReLU()

        self.fcdown1r = nn.Linear(transchannel, transchannel//2)
        self.fcdown1i = nn.Linear(transchannel, transchannel//2)
        self.NLdown1r = nn.PReLU()
        self.NLdown1i = nn.PReLU()

        self.fcdown2r = nn.Linear(transchannel//2, outchannel)
        self.fcdown2i = nn.Linear(transchannel//2, outchannel)
        self.NLdown2r = nn.PReLU()
        self.NLdown2i = nn.PReLU()


    def forward(self, data):
        xr, xi, edge_index, edge_attr = data.xr, data.xi, data.edge_index, data.edge_attr

        xr = self.NLup2r(self.fcup2r(self.NLup1r(self.fcup1r(xr))))
        xi = self.NLup2i(self.fcup2i(self.NLup1i(self.fcup1i(xi))))

        xr, xi = self.onestep1(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep2(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep3(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep4(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep5(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep6(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep7(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep8(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep9(xr, xi, edge_index, edge_attr)
        xr, xi = self.onestep10(xr, xi, edge_index, edge_attr)

        xr = self.NLdown2r(self.fcdown2r(self.NLdown1r(self.fcdown1r(xr))))
        xi = self.NLdown2i(self.fcdown2i(self.NLdown1i(self.fcdown1i(xi))))

        return xr, xi