# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:39:49 2020

@author: xubinwei
"""
import torch
class AttRestoration:
    def __init__(self, device):
        self.device = device
    def forward(self, data_pred, grid):
        self.grid = grid
        self.data_pred = data_pred
        x_index = grid[0,1,:,0]#400 这里需要cuda
        print(x_index)
        y_index = grid[0,:,1,1]#300
        new_data_size = tuple(data_pred.shape[1:4])  #这里需要cuda吗
        new_data = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],device=self.device)#创建新的图
        new_data_final = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                     device=self.device)#创建新的图
        x_index = (x_index+1)*new_data_size[2]/2
        y_index = (y_index+1)*new_data_size[1]/2
        
        #restoration x direction
        xl = 0 #节点
        grid_l = x_index[0] #有关距离的变量（距离）
        data_l = data_pred[:,:,:,0]#这里的数据格式可能有问题，需要做出改变（关于像素值的变量）
        for num in range(1,len(x_index)):
            grid_r = x_index[num] #分别是grid前后对应的坐标值
            xr = torch.ceil(grid_r)-1 #分别是前后的序号
            xr = xr.int()
            data_r = data_pred[:,:,:,num]
            for h in range(xl+1,xr+1):#插值过程
                if h == grid_r: #grid的坐标和点重合 直接将值赋给新的数据
                    new_data[:,:,h] = data_r 
                else:
                    new_data[:,:,h] = (( h - grid_l)*data_r/(grid_r - grid_l))+ ((grid_r - h)*data_l/(grid_r - grid_l))
            xl = xr #右边的换成左边的
            grid_l = grid_r
            data_l = data_r
        new_data[:,:,0] = new_data[:,:,1]
        try:
            for h in range(xr+1,len(x_index)):
                new_data[:,:,h] = new_data[:,:,xr]
        except:
            print('h',h)
            print('xr',xr)
            
        #restoration y direction
        yl = 0
        grid1_l = y_index[0] #有关距离的变量（距离）
        data1_l = new_data[:,0,:]#这里的数据格式可能有问题，需要做出改变（关于像素值的变量）
        for num in range(1,len(y_index)):
            grid1_r = y_index[num] #分别是grid前后对应的坐标值
            #print('grid1_r',grid1_r)
            yr = torch.ceil(grid1_r)-1 #分别是前后的序号
            #print('yr',yr)
            yr = yr.int()
            data1_r = new_data[:,num,:]
            for h in range(yl+1,yr+1):#插值过程
        #         print('h',h)
                if h == grid1_r: #grid的坐标和点重合 直接将值赋给新的数据
                    new_data_final[:,h,:] = data1_r 
                else:
                    new_data_final[:,h,:] = (( h - grid1_l)*data1_r/(grid1_r - grid1_l))+ ((grid1_r - h)*data1_l/(grid1_r - grid1_l))
            yl = yr #右边的换成左边的
            grid1_l = grid1_r
            data1_l = data1_r     
        new_data_final[:,0,:] = new_data_final[:,1,:]#第一列补0
        try:
            for h in range(yr+1,len(y_index)):
                new_data_final[:,h,:] = new_data_final[:,yr,:]#最终的结果是new_data_final
        except:
            print('h',h)
            print('yr',yr)
        new_data_final = torch.unsqueeze(new_data_final,dim = 1)
        return new_data_final