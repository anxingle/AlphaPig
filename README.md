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

+  SGF格式详解

  ```
  FF[4] SGF格式的版本号，4是最新
  SZ[15] 棋盘大小，这是15x15 
  PW[Pig]白棋棋手名称
  WR[2a]白棋棋手段位
  PB[stupid]黑棋棋手名称
  BR[2c]黑棋棋手段位
  DT[2018-07-06]棋谱生成日期
  PC[CA]棋局所在位置
  KM[6.5]贴目数量
  RE[B+Resign]B+是黑胜，W+是白胜，Resign是对方GG的
  CA[utf-8]棋局编码
  TM[0]限时情况，0为无限时
  OT[]读秒规则
  ;B[pp];W[dd];B[pc];W[dq] …… 棋谱下棋顺序
  ```

  

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

