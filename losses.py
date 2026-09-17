#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import Tensor, einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth: float = kwargs.get('smooth', 1e-8)
        self.batch_dice: bool = kwargs.get('batch_dice', True)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        sum_str: str = "bkwh->k" if self.batch_dice else "bkwh->bk"
        inter = einsum(f"bkwh,{sum_str}", pred, mask)
        union = einsum(sum_str, pred) + einsum(sum_str, mask)

        dices = (2 * inter + self.smooth) / (union + self.smooth)

        return 1 - dices.mean()
    

class CrossEntropyAndDice():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.alpha: float = kwargs.get('alpha', 1.0)
        self.beta: float = kwargs.get('beta', 1.0)

        self.ce = CrossEntropy(idk=self.idk)
        self.dice = DiceLoss(idk=kwargs.get('dice_idk', self.idk),
                             smooth=kwargs.get('smooth', 1e-8),
                             batch_dice=kwargs.get('batch_dice', True))
        print(f"Initialized {self.__class__.__name__} with alpha={self.alpha} beta={self.beta}")

    def __call__(self, pred_softmax, weak_target):
        return (self.alpha * self.ce(pred_softmax, weak_target)
                + self.beta * self.dice(pred_softmax, weak_target))