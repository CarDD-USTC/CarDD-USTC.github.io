import mobula


@mobula.op.register
class AttSamplerGrid:
    def __init__(self, scale=1.0, dense=4, iters=5):
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):#data[1, 1, 224, 224]  attx[1, 224, 1] 
        F = self.F._mobula_hack
        # attx: (N, W, 1)
        # atty: (N, H, 1)
        N, _, in_size, in_sizey = data.shape#N大概是batch_size. in_size是图片的宽度224
        att_size = attx.shape[1]#的到图片的宽度
        att_sizey = atty.shape[1]

        out_size = int(in_size * self.scale)#输出的size和输入的size一样大
        out_sizey = int(in_sizey * self.scale)#输出的size和输入的size一样大
        #print('out_sizey',out_sizey)
        
        #threshold应该是 方大缩小的界限
        threshold = float(self.scale * self.dense * in_size) / att_size
        
        #print('threshold',threshold)
        #attention的尺寸根据输入和输出的尺寸改变
        attx = attx * out_size
        atty = atty * out_sizey
        #print('attx',attx)
        for j in range(self.iters):
            max_attx = F.max(attx, 1, keepdims=True)  # (N, 1, 1)
            #print('max_attx',max_attx)
            max_atty = F.max(atty, 1, keepdims=True)  # (N, 1, 1)
            #print('max_atty',max_atty)
            if j == 0:
                threshold = F.minimum(F.minimum(
                    max_attx, max_atty), threshold)  # (N, 1, 1)
            else:
                threshold = F.minimum(max_attx, max_atty)
            #print(j)
            #print(threshold)
            #print('attx',attx)
            
            F.broadcast_minimum(threshold, attx, out=attx)
            #print('attx',attx)
            F.broadcast_minimum(threshold, atty, out=atty)
            sum_x = F.sum(attx, 1, keepdims=True)  # (N, 1, 1)
            sum_y = F.sum(atty, 1, keepdims=True)  # (N, 1, 1)
            deltax = (out_size - sum_x) / att_size
            deltay = (out_sizey - sum_y) / att_sizey
            # compensate the drop value
            attx += deltax
            atty += deltay
        '''
        it is the same as the original implemenation.
        the first element is 1.
        '''
        attx[:, 0] = 1
        #print(attx)
        atty[:, 0] = 1
        
        #产生逆变换函数的过程
        attxi = F.cumsum(attx, 1)#新的attention坐标300
        #print('attxi',attxi)
        attyi = F.cumsum(atty, 1)

        stepx = attxi[:, -1] / out_size
        stepy = attyi[:, -1] / out_sizey #stepy tensor([[1.0034]])
        ctx = F.get_ctx(stepx)
        
        #创建随机变量的过程
        index_x = F.empty((N, out_sizey, 1), ctx=ctx)#-1,1 应该是
        index_y = F.empty((N, out_size, 1), ctx=ctx)

        #应该是逆变换的过程（离散逆变换的过程，涉及插值的部分）
        mobula.func.map_step(N, attxi, index_y, stepx, att_size, out_size)
        mobula.func.map_step(N, attyi, index_x, stepy, att_sizey, out_sizey)
        #GG = F.tile(F.reshape(index_x, (N, 1, out_sizey)), (1, out_size, 1))
        #MM = F.tile(index_y, (1, 1, out_sizey))
        #print('GG',GG)
        #print('GG',GG.shape)
        #print('MM',MM)
        #print('GG',MM.shape)
        return F.tile(F.reshape(index_x, (N, 1, out_sizey)), (1, out_size, 1)),\
            F.tile(index_y, (N, 1, out_sizey))

    def backward(self, dy_x, dy_y):
        return [0, 0, 0]

    def infer_shape(self, in_shape):
        #in_shape [torch.Size([1, 1, 342, 400]), torch.Size([1, 342, 1]), torch.Size([1, 400, 1])]
        dshape = in_shape[0]
        out_size = int(dshape[2] * self.scale)
        #dshape1 = in_shape[1]
        out_size1 = int(dshape[3] * self.scale)
        #print('out_size1',out_size1.shape)
        oshape = (dshape[0], out_size, out_size1)
        return in_shape, [oshape, oshape]
