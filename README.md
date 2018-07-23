## AlphaPig
使用AlphaZero算法在五子棋上的实现。五子棋比围棋简单，训练起来也稍微能够简单一点，所以选择了五子棋来作为AlphaZero的复现。

参考:
1. [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
2. [starimpact/AlphaZero_Gomoku](https://github.com/starimpact/AlphaZero_Gomoku)
3. [yonghenglh6/GobangServer](https://github.com/yonghenglh6/GobangServer)

### 上手入门
+ cd AlphaPig/sgf_data/

+ ```
  sh ./download.sh 
  ```

  将会下载并解压SGF棋谱数据，解压后应该实在sgf_data/目录下，目录结构AlphaPig/sgf_data/*.sgf。

  也可自行下载SGF棋谱数据，并自行处理。

+ 直接运行根目录下的start_train.sh开始训练

+ 或者进入 train_mxnet.py 修改网络结构等参数，其中conf下的.yaml为训练定义的一些参数，可修改为适合自己的相关参数。

+ 如果需要和自己的AI对弈，可以进入evaluate目录，运行

  ```
  python ChessServer.py --port 8888
  ```

  既可与自己的AI进行对弈，或者与yixin对弈。详细说明请参阅evaluate目录下的ReadMe。

  对弈例子:

  <img src="http://p324ywv2g.bkt.clouddn.com/test.jpg" height="400px" />

## 致谢

+ 源工程请移步[junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) ，特别感谢大V的很多issue和指导。

+ 特别感谢格灵深瞳提供的很多训练帮助（课程与训练资源上提供了很大支持），没有格灵深瞳的这些帮助，训练起来毫无头绪。

+ 感谢[Uloud](https://www.ucloud.cn/) 提供的P40 AI-train服务，1256小时/实例的训练，验证了不少想法。而且最后还免单了，中间没少打扰技术支持。特别感谢他们。

  <img src="http://p324ywv2g.bkt.clouddn.com/test2.jpg" width="500px">

## AlphaPig

Implementation of the AlphaZero algorithm for playing the simple board game Gomoku . The game Gomoku is much simpler than Go or chess, so that we can focus on the training scheme of AlphaZero and obtain a pretty good AI model on a single PC in a few hours. 

References:  
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. AlphaGo Zero: Mastering the game of Go without human knowledge

### Requirements
To play with the trained AI models, only need:
- Python >= 2.7
- Numpy >= 1.11

To train the AI model from scratch, further need, either:
- Mxnet >0.8

**PS**: if your Theano's version > 0.7, please follow this [issue](https://github.com/aigamedev/scikit-neuralnetwork/issues/235) to install Lasagne,  
otherwise, force pip to downgrade Theano to 0.7 ``pip install --upgrade theano==0.7.0``

If you would like to train the model using other DL frameworks, you only need to rewrite policy_value_net.py.

### Getting Started
To play with provided models, run the following script from the directory:  
```
python human_play.py  
```
You may modify human_play.py to try different provided models or the pure MCTS.

To train the AI model from scratch, with Theano and Lasagne, directly run:   
```
python train.py
```
With PyTorch or TensorFlow, first modify the file [train.py](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py), i.e., comment the line
```
from policy_value_net import PolicyValueNet  # Theano and Lasagne
```
and uncomment the line 
```
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
or
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
```
and then execute: ``python train.py``  (To use GPU in PyTorch, set ``use_gpu=True``)

The models (best_policy.model and current_policy.model) will be saved every a few updates (default 50).  

### Further reading
My article describing some details about the implementation in Chinese: [https://zhuanlan.zhihu.com/p/32089487](https://zhuanlan.zhihu.com/p/32089487) 