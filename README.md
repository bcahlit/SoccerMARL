# soccer_on_RLlib

在RLlib框架上实现rcssserver的环境。

# FAQ
### 开始训练：
首先需要安装pytorch，ray，hfo，rcssserver
使用python启动文件
example：
python base_IMPALA_multi_agent.py
###　查看训练的录像：

运行中会出现这样的提示：

```shell
(pid=944393) /home/bcahlit/anaconda3/envs/rllib/lib/python3.8/site-packages/hfo_py/bin/HFO
(pid=944393) Starting server with command: /home/bcahlit/anaconda3/envs/rllib/lib/python3.8/site-packages/hfo_py/bin/HFO --headless --frames-per-trial 500 --offense-agents 1 --defense-agents 0 --offense-npcs 0 --defense-npcs 0 --port 41067 --offense-on-ball 0 --seed -1 --ball-x-min 0.000000 --ball-x-max 0.200000 --log-dir log --fullstate --no-logging
```
--port 41067　为monitor的地址。
在命令行，输入
```shell
soccerwindow2 --port 41067
```
会打开soccerwindow2 右键连接就能看到训练过程。