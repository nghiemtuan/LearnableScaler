# LearnableScaler
LearnableScaler, a possible replacement for normalization methods such as LayerNorm, BatchNorm, RMS norm etc. in neural networks.
##Introduction:
Normalization techniques like LayerNorm and BatchNorm are employed to mitigate the risk of overflow during the training of neural networks. Nonetheless, these methods may inadvertently lead to a decrease in the model's generalization performance on unseen (testing) data.

LearnableScaler is proposed as a replacement for those normalization methods in neural networks. Compared with the former methods, it is faster and doesn't affect generalization ability of neural networks as much as LayerNorm or BatchNorm. For example, in the experiment with Vision Transformer model having 36 layers, just by replacing LayerNorm with LearnableScaler, the testing accuracy increases 3.4 % (from 76.32% to 79.77%). A detailed explanation of LearnableScaler will be presented in a blog post.

##Code
To use LearnableScaler, just replace LayerNorm in Transformer with this code:
```
class LearnableScaler(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        #ipdb.set_trace()
        self.a = torch.nn.Parameter( torch.nn.init.normal_(torch.ones( num_channels )) )
        self.b = torch.nn.Parameter( torch.zeros( num_channels ) )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a[None,None,:] * x + self.b[None, None,:]
```
Here `x` has the format `b n d`.
To use LearnableScaler, just replace BatchNorm in CNN with this code:
```
class LearnableScaler2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.a = torch.nn.Parameter( torch.nn.init.normal(torch.ones( num_channels )) )
        self.b = torch.nn.Parameter( torch.zeros( num_channels ) )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a[None,:,None,None] * x + self.b[None,:, None, None]
```
Here `x` has the format `b c h w`.
When training networks using LearnableScaler, there are several important points to consider. Firstly, to train those networks, one should use smaller learning rate than the one used to train networks with LayerNorm or BatchNorm. Secondly, the number of training epochs should be long enough so that the networks should have enough time to learn small details of training data. Finally, LearnableScaler would be more suitable for deep networks than the shallow ones.
##Training:
To replicate the training result, run:
```
python train.py /home/anghiem/extra_disk/ILSVRC/Data/CLS-LOC/ --model vit_learnable_scaler_embed_dim192_patch16_head3_depth36 --weight-decay 0.02 --opt lamb --aa rand-m7-mstd0.5 --warmup-epoch 0 --epochs 300 --sched custom --mixup 0.1 --cutmix 1 --batch-size 256 --workers 14  --clip-grad 14 --grad-accum-steps 8 --lr 0.005 --amp --amp-dtype bfloat16 --torchcompile inductor --epoch-factor 0.8 --epoch-threshold 0.5 --train-feature-decay 0.1 --attn-extra-loss 0.03 --attn-extra-loss-layers 6
```
For transformers with large embeding dimension, one can use an additional loss by adding '--attn-extra-loss 0.03 --attn-extra-loss-layers 6' into the training command.

