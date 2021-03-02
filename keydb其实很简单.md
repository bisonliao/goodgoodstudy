## keydb其实很简单

### 1、keydb是什么？

根据官网，keydb是兼容redis接口协议的nosql数据库，比redis快5倍，支持平行扩展和垂直扩展，支持跨域的多master复制（MMR）

官网：https://keydb.dev/

### 2、构建好用的镜像

为了节约成本、提高效率，我用docker镜像来搭建，并且keydb实例都部署在同一个CVM下。

keydb的官方镜像文件里面很多网络工具没有，不是很方便，为此我基于官方镜像，自己build一个镜像, dockerfile如下：

```
FROM eqalpha/keydb
COPY sources.list /etc/apt/sources.list
RUN  apt-get update
RUN  apt-get install -y net-tools
RUN  apt-get install -y inetutils-ping
```

其中sources.list是腾讯云的apt-get库的镜像地址，不更新进去的化安装工具会失败：

```
deb http://mirrors.cloud.tencent.com/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.cloud.tencent.com/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.cloud.tencent.com/ubuntu/ xenial-updates main restricted universe multiverse
#deb http://mirrors.cloud.tencent.com/ubuntu/ xenial-proposed main restricted universe multiverse
##deb http://mirrors.cloud.tencent.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.cloud.tencent.com/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.cloud.tencent.com/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.cloud.tencent.com/ubuntu/ xenial-updates main restricted universe multiverse
##deb-src http://mirrors.cloud.tencent.com/ubuntu/ xenial-proposed main restricted universe multiverse
##deb-src http://mirrors.cloud.tencent.com/ubuntu/ xenial-backports main restricted universe multiverse
```

具体可以见下面页面中的ubuntu：

https://mirrors.tencent.com/

https://mirrors.tencent.com/help/ubuntu.html

执行命令build自己的镜像：

```
docker build -t keydb:bison .
```



### 3、体验multi master replication(MMR)

#### 3.1  情况A：环形复制关系

创建一张桥接网卡，docker容器都使用该网卡所在的网段：

```
docker network create keydb_network
docker network ls
```

使用ifconfig命令可以看到该网卡的地址为 172.18.0.1，依次创建的容器的地址将是：

```
172.18.0.2
172.18.0.3
172.18.0.4
```

创建node1  node2  node3三个目录，每个目录一个配置文件：

```
#node1/redis.conf
bind 172.18.0.2
port 6379
requirepass mypassword123
masterauth mypassword123

active-replica yes
replicaof  172.18.0.3 6379
```

```
#node2/redis.conf
bind 172.18.0.3
port 6379
requirepass mypassword123
masterauth mypassword123

active-replica yes
replicaof  172.18.0.4 6379
```

```
#node3/redis.conf
bind 172.18.0.4
port 6379
requirepass mypassword123
masterauth mypassword123

active-replica yes
replicaof  172.18.0.2 6379
```

可以看到，这三个实例假设叫A B C，他们的复制关系形成了一个环，如果其中一个环节故障了，会影响同步：

```
A->B->C->A
```

#### 3.1 情况B：三个实例，两两间同步

也可以配置为三个实例两两间同步，因为keydb支持从多个master复制：

三个配置文件分别为：

```
hongkong1#cat /keydb/node1/redis.conf
bind 172.18.0.2
port 6379
requirepass mypassword123
masterauth mypassword123
active-replica yes
multi-master yes
replicaof  172.18.0.3 6379
replicaof  172.18.0.4 6379

hongkong1#cat /keydb/node2/redis.conf
bind 172.18.0.3
port 6379
requirepass mypassword123
masterauth mypassword123
active-replica yes
multi-master yes
replicaof  172.18.0.2 6379
replicaof  172.18.0.4 6379

hongkong1#cat /keydb/node3/redis.conf
bind 172.18.0.4
port 6379
requirepass mypassword123
masterauth mypassword123
active-replica yes
multi-master yes
replicaof  172.18.0.3 6379
replicaof  172.18.0.2 6379
```

#### 3.2 启动和验证


```
docker stop mynode1 mynode2 mynode3
docker rm mynode1 mynode2 mynode3

docker run -v /keydb/node1/redis.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode1 -d keydb:bison
docker run -v /keydb/node2/redis.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode2 -d keydb:bison
docker run -v /keydb/node3/redis.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode3 -d keydb:bison
```

登入进去可以输入命令访问数据：

```
docker exec -it mynode1 /bin/bash
keydb-cli -h 172.18.0.2
```


下面通过在三个实例里分别修改同一个key的不同字段，可以看到数据被merge在一起

```
hongkong1#docker exec -it mynode1 /bin/bash
root@de06f7c6a3b3:/data# keydb-cli -h 172.18.0.2
Message of the day:
  Join the KeyDB community! https://community.keydb.dev/

172.18.0.2:6379> AUTH mypassword123
OK
172.18.0.2:6379> hset bison name liaonb
(integer) 1
172.18.0.2:6379> quit
root@de06f7c6a3b3:/data# keydb-cli -h 172.18.0.3
Message of the day:
  Join the KeyDB community! https://community.keydb.dev/

172.18.0.3:6379> AUTH mypassword123
OK
172.18.0.3:6379> hset bison age 40
(integer) 1
172.18.0.3:6379> hgetall bison
1) "name"
2) "liaonb"
3) "age"
4) "40"
172.18.0.3:6379> quit
root@de06f7c6a3b3:/data# keydb-cli -h 172.18.0.4
Message of the day:
  Join the KeyDB community! https://community.keydb.dev/

172.18.0.4:6379> AUTH mypassword123
OK
172.18.0.4:6379> hset bison gender male
(integer) 1
172.18.0.4:6379> hgetall bison
1) "name"
2) "liaonb"
3) "age"
4) "40"
5) "gender"
6) "male"
172.18.0.4:6379>
```




验证的结论：

1. 相同key的hashtable/set/list会merge field/member；
2. 不同的key数据两边会同步
3. 一个节点宕机后重启，能够拉取到最新的数据
4. key expire也会被同步
5. 可以是多个（超过2个）master，复制关系做成环状即可：A->B->C→A，当然如果有一个节点故障，那其他master之间的同步都会受到影响。如果replicaof配置可以热更新，也许这不是什么问题
6. 支持从多个master复制，也就是配置多次replicaof配置项，如果配置多个，最后一个为准

### 4、体验集群

#### 4.1 准备工作

创建6个目录，每个目录下一个配置文件：

```
hongkong1#cat 7001/keydb.conf
bind 172.18.0.2
port 7001
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7001.log
save 900 1
save 300 10
save 60 10000
```

```
hongkong1#cat 7002/keydb.conf
bind 172.18.0.3
port 7002
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7002.log
save 900 1
save 300 10
save 60 10000
```

```
hongkong1#cat 7003/keydb.conf
bind 172.18.0.4
port 7003
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7003.log
save 900 1
save 300 10
save 60 10000
```

```
hongkong1#cat 7004/keydb.conf
bind 172.18.0.5
port 7004
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7004.log
save 900 1
save 300 10
save 60 10000
```

```
hongkong1#cat 7005/keydb.conf
bind 172.18.0.6
port 7005
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7005.log
save 900 1
save 300 10
save 60 10000
```

```
hongkong1#cat 7006/keydb.conf
bind 172.18.0.7
port 7006
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
dir ./
loglevel notice
logfile 7000.log
save 900 1
save 300 10
save 60 10000
```

把他们启动起来：

```
hongkong1#cat start.sh
docker stop mynode1 mynode2 mynode3 mynode4 mynode5 mynode6
docker rm mynode1 mynode2 mynode3  mynode4 mynode5 mynode6

docker run -v /keydb/cluster/7001/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode1 -d keydb:bison
docker run -v /keydb/cluster/7002/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode2 -d keydb:bison
docker run -v /keydb/cluster/7003/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode3 -d keydb:bison
docker run -v /keydb/cluster/7004/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode4 -d keydb:bison
docker run -v /keydb/cluster/7005/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode5 -d keydb:bison
docker run -v /keydb/cluster/7006/keydb.conf:/etc/keydb/redis.conf   --network keydb_network  --name mynode6 -d keydb:bison
```

然后对这些节点创建一个集群：

```
docker exec -it mynode1 /bin/bash
keydb-cli --cluster create 172.18.0.2:7001  172.18.0.3:7002  172.18.0.4:7003  172.18.0.5:7004  172.18.0.6:7005  172.18.0.7:7006 --cluster-replicas 1
```

工具的输出结果提示：创建集群成功，3个Master，3个Slave。各自负责一段slot（hash槽）：

```
Master[0] -> Slots 0 - 5460
Master[1] -> Slots 5461 - 10922
Master[2] -> Slots 10923 - 16383
Adding replica 172.18.0.6:7005 to 172.18.0.2:7001
Adding replica 172.18.0.7:7006 to 172.18.0.3:7002
Adding replica 172.18.0.5:7004 to 172.18.0.4:7003
```

#### 4.2 验证集群功能

连接任一master，写入数据，如果不是它负责的hash槽，会提示连接其他节点：

```
172.18.0.2:7001> hset bison name liaonb
(error) MOVED 8414 172.18.0.3:7002
```

换到对应的节点能够写入，且hash tag也能生效：

```
172.18.0.3:7002> hset bison name liaonb
(integer) 1
172.18.0.3:7002> hset {bison}liao age 40
(integer) 1
```

使用cluster nodes命令查看有哪些节点，然后连接到刚才的master对应的slave，尝试写入，会拒绝：

```
172.18.0.7:7006> hset bison gender male
(error) MOVED 8414 172.18.0.3:7002
```

关闭刚才那个master，再尝试，可以写入，这时候slave已经升级为master了：

```
172.18.0.7:7006> hset bison name liaonb
(integer) 0
```

keydb的配置文件的详细说明在这里：

https://docs.keydb.dev/docs/config-file/#keydb-configuration