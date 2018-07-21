

# 使用指南
```
cd AlphaPig/evaluate
python ChessServer.py  # 将会在本机11111端口启动对弈服务，浏览器打开localhost:11111即可
```

# 与AI对弈

1. 
```
   # AlphaPig/evaluate 目录下
   python ChessClient.py --model XXX.model --cur_role 1/2 --room_name ROOM_NAME --server_url 127.0.0.1:8888
```
--cur_role 1是黑手，2是白手； —room_name是对弈的房间号码，server_url 是对弈服务器的地址（默认是本机喽）

如果是两个AI对弈，则双方出了--cur_role之外，都敲入相同的参数。如果是AI与人，则谁先进去先有优先选择权。

# 人人对弈

直接打开浏览器，约定房间，没什么好说的。