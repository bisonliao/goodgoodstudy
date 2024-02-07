#### 1、如何在一个容器的namespace下执行命令

1. 先用docker ps和ps找到对应的进程的pid
2. 然后用lsns查看指定进程的net namespace
3. 然后用nsenter在指定的netnamespace里执行命令，例如ifconfig或者iptables

```
root@VM-7-84-ubuntu:~# docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED         STATUS         PORTS     NAMES
6c744cd2b310   busybox   "sh"      8 minutes ago   Up 8 minutes             busybox

root@VM-7-84-ubuntu:~# ps auxw|grep sh
root     3071251  0.0  0.0   4400   408 pts/0    Ss+  08:17   0:00 sh
root     3073429  0.0  0.0   6608  2376 pts/0    S+   08:26   0:00 grep --color=auto sh

root@VM-7-84-ubuntu:~# lsns -t net -p 3071251
        NS TYPE NPROCS     PID USER NETNSID NSFS                           COMMAND
4026532614 net       1 3071251 root       2 /run/docker/netns/b700fa1c5d96 sh
root@VM-7-84-ubuntu:~# nsenter --net=/run/docker/netns/b700fa1c5d96 ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.17.0.2  netmask 255.255.0.0  broadcast 172.17.255.255
        ether 02:42:ac:11:00:02  txqueuelen 0  (Ethernet)
        RX packets 5250  bytes 115881923 (115.8 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 5140  bytes 279702 (279.7 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

```

#### 2、如何制造网损

一个场景是与某个服务器ip的通讯要限速，可以用iptables丢包来限速：

```
# 对来源ip为183.2.193.201的ip包按照30%的概率丢包
iptables -A INPUT -i eth0 -s 183.2.193.201 -m statistic --mode random --probability 0.3 -j DROP

#删除上述规则
iptables -D INPUT -i eth0 -d 183.2.193.201 -m statistic --mode random --probability 0.3 -j DROP
```

有意思的是，如果在宿主机上执行上述丢包策略，宿主机上的从该ip的下载会明显降速，但该宿主机上的容器不会受到影响，一定要到容器里面去执行iptables命令。暂时没有想清楚。