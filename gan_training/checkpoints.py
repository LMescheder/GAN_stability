import os
import torch


class CheckpointIO(object):
    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, it, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        outdict = {'it': it}
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print('=> Loading checkpoint...')
            out_dict = torch.load(filename)
            it = out_dict['it']
            for k, v in self.module_dict.items():
                if k in out_dict:
                    v.load_state_dict(out_dict[k])
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)
        else:
            it = -1

        return it
