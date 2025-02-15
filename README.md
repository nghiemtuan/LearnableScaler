# LearnableScaler
LearnableScaler, a possible replacement for normalization methods such as LayerNorm, BatchNorm, RMS norm etc. in neural networks.
##Introduction:
Normalization techniques like LayerNorm and BatchNorm are employed to mitigate the risk of overflow during the training of neural networks. Nonetheless, these methods may inadvertently lead to a decrease in the model's generalization performance on unseen (testing) data.

LearnableScaler is proposed as a replacement for those normalization methods in neural networks. Compared with the former methods, it is faster and doesn't affect generalization ability of neural networks as much as LayerNorm or BatchNorm. For example, in the experiment with Vision Transformer model having 36 layers, just by replacing LayerNorm with LearnableScaler, the testing accuracy increases 3.4 % (from 76.32% to 79.77%). A detailed explanation of LearnableScaler will be presented in a blog post.
##Benchmark
The following table shows the effect of LearnableScaler with deep Transformers:

|__vit(patch, embed dim, depth, norm)__| ImageNet1k acc. | ImageNet-V2 acc. | Pre-train weight |
|--------------------------------------|-----------------|------------------|------------------|
|vit(16, 192, 24, LayerNorm)           | 77.14           | 64.9             |
|vit(16, 192, 24, LearnableScaler)     | 78.77           | 67.02            |
|vit(16, 192, 36, LayerNorm)           | 76.2            | ??               |
|vit(16, 192, 36, LearnableScaler)     | 79.77           | 68.21            |[checkpoint](https://drive.google.com/file/d/1jVEP0IAzJO2MMGRe26rHLixfgB2W_eQv/view?usp=drive_link)|
|vit(16, 288, 24, LayerNorm)           | 78.2            | 65.71            |                   |
|vit(16, 288, 24, LearnableScaler)     | 79.72           | 68.16            |[checkpoint](https://drive.google.com/file/d/1YcKPs9Q3MeebsR2WRvH0oQCBaqRpOfHO/view?usp=drive_link)|
|vit(32, 288, 24, LayerNorm)           | 69.93           | 55.81            |                   |
|vit(32, 288, 24, LearnableScaler)     | 74.53           | 60.82            |                   |
|vit(32, 288, 36, LayerNorm)           | 67.86           | 53.35            |                   |
|vit(32, 288, 36, LearnableScaler)     | 75.71           | 62.76            |                   |



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
The training protocol is the one in [this paper](https://arxiv.org/abs/2110.00476)
To replicate the training result, run:
```
python train.py <path to imagenet dataset> --model <model_name> --weight-decay 0.02 --opt lamb --aa rand-m7-mstd0.5 --warmup-epoch 0 --epochs 300 --sched custom --mixup 0.1 --cutmix 1 --batch-size 256 --workers 14  --clip-grad 14 --grad-accum-steps 8 --lr 0.005 --amp --amp-dtype bfloat16 --torchcompile inductor --epoch-factor 0.8 --epoch-threshold 0.5 --train-feature-decay 0.1 --attn-extra-loss 0.03 --attn-extra-loss-layers 6
```
For transformers with large embeding dimension, one can use an additional loss by adding '--attn-extra-loss 0.03 --attn-extra-loss-layers 6' into the training command.

To validate the checkpoints run:
```
python validate.py <path to the validation dataset> --model <model name> --checkpoint <path to the checkpoint> --amp --amp-dtype bfloat16 --torchcompile inductor
```
