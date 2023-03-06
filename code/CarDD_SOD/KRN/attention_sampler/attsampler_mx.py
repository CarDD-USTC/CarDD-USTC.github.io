import mobula
import mxnet as mx

# Hack MobulaOP for the compatible functions
mx.nd._mobula_hack = mx.nd
mx.nd.get_ctx = lambda self: self.context


class AttSampler(mx.gluon.HybridBlock):
    def __init__(self, scale=1.0, dense=4, iters=5):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def hybrid_forward(self, F, data, attx, atty):
        grid = mobula.op.AttSamplerGrid(data, attx, atty,
                                        scale=self.scale,
                                        dense=self.dense,
                                        iters=self.iters)
        grid = F.stack(*grid, axis=1)
        return F.BilinearSampler(data, grid)


def AttSamplerWrapper(data, attx, atty, scale=1.0, dense=4, iters=5):
    return AttSampler(scale=scale, dense=dense, iters=iters)(data, attx, atty)


mx.nd.AttSampler = AttSamplerWrapper
mx.nd.contrib.AttSampler = AttSamplerWrapper
mx.sym.AttSampler = AttSamplerWrapper
mx.sym.contrib.AttSampler = AttSamplerWrapper
