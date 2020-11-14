## cassandra其实挺简单

### 7句话说清楚cassandra

1. cassandra是一个开源的、去中心化的、节点间同等地位的P2P的一个数据库。
2. 支持的数据结构类似HBase，行key、column family、column、timestamp、value等概念也是类似的。适合于巨大的稀疏矩阵型的宽表：十亿行 X 百万的列的场景。提供sql语句的友好方式进行数据访问
3. cassandra使用gossip协议在不同节点（机器）间按row key同步数据，使用一致性hash在不同的节点（机器）间进行sharding。
4. cassandra的部署可以根据机架、IDC的配置信息，进行跨IDC的副本复制
5. 可以指定数据的副本的个数，包括在每个机房的副本个数
6. 每次读写数据，可以指定一致性要求，例如是读到2个副本才算成功读，写一半以上的副本数才算成功写
7. 使用commitlog+内存+磁盘的方式平衡读写性能与可靠性

借用网友的两张图：

![这里有张图](img/cassandra/cassandra.jpg)

![这里有张图](img/cassandra/features.jpg)

根据网上的资料，黑石基金的技术部门、饿了么的技术部门据说都在大规模的使用cassandra。

https://cloud.tencent.com/developer/article/1084113

这里简单的实验，演示cassandra跨IDC的数据库分布。

### 实验演示

我这里用香港和硅谷的两台机器，通过docker跑两个节点，演示跨IDC的数据库同步和访问。

两个机器都部署cassandra，并启动一个容器，从容器里拷贝出配置文件：

```shell
docker pull daocloud.io/cassandra
docker run --name cpy -it daocloud.io/cassandra /bin/bash
groupadd cassandra
useradd -g cassandra cassandra
mkdir /opt/cassandra; cd /opt/cassandra
docker cp cpy:/opt/cassandra/conf/cassandra.yaml ./
chown -R cassandra:cassandra /opt/cassandra; chmod -R 777  /opt/cassandra
docker stop cpy;docker rm cpy
```

编辑/opt/cassandra/cassandra.yaml文件，修改配置如下：

```
cluster_name: 'bison'
authenticator: PasswordAuthenticator
```

创建数据存放目录，并启动容器：

```shell
mkdir /var/lib/cassandra
chown -R cassandra:cassandra /var/lib/cassandra
chmod -R 777 /var/lib/cassandra

#注意修改IP为节点间通信暴露的IP
docker run -d --name cass_node1 -e CASSANDRA_BROADCAST_ADDRESS=119.28.XX.XX  \  
-e  CASSANDRA_DC=DC1 -e CASSANDRA_RACK=RAC1  \
-e  CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch  \
-v /etc/localtime:/etc/localtime:ro \
-v /opt/cassandra/cassandra.yaml:/opt/cassandra/conf/cassandra.yaml:rw \
-v /var/lib/cassandra:/var/lib/cassandra:rw \
-u  cassandra \
-p 7000:7000 -p 9042:9042 -p 7199:7199 daocloud.io/cassandra
```

在另外一台机器上也做上述安装和启动，区别就是机器IP、容器名字和机架/IDC要改一下：

```shell
#注意修改IP为节点间通信暴露的IP
docker run -d --name cass_node2 -e CASSANDRA_BROADCAST_ADDRESS=49.51.XX.XX  \  
-e  CASSANDRA_DC=DC2 -e CASSANDRA_RACK=RAC1  \
-e  CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch  \
-e  CASSANDRA_SEEDS=119.28.XXX.XXX \  #香港的那个机器作为种子，这样集群的成员节点就相互知道了
-v /etc/localtime:/etc/localtime:ro \
-v /opt/cassandra/cassandra.yaml:/opt/cassandra/conf/cassandra.yaml:rw \
-v /var/lib/cassandra:/var/lib/cassandra:rw \
-u  cassandra \
-p 7000:7000 -p 9042:9042 -p 7199:7199 daocloud.io/cassandra
```

如果不出意外，集群就部署起来了，可以使用nodetool查看集群状态：

```shell
docker exec -it cass_node2 /bin/bash
nodetool  status #在容器里面执行
```

显示结果如下：

```shell
cassandra@0ae069411888:/$ nodetool  status
Datacenter: DC1
===============
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address        Load       Tokens       Owns    Host ID                               Rack
UN  119.28.xxx.xxx  6.27 MiB   256          ?       56680a4f-18e9-404c-929e-787ecb6aab63  RAC1
Datacenter: DC2
===============
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address        Load       Tokens       Owns    Host ID                               Rack
UN  49.51.xxx.xxx  564.84 KiB  256          ?       f6fb4fdf-b99c-4ffb-a9bd-c71db8be13e2  RAC1

Note: Non-system keyspaces don't have the same replication settings, effective ownership information is meaningless


```

继续在容器内使用cqlsh工具访问数据：

```sql
cqlsh -ucassandra -pcassandra

> CREATE KEYSPACE testdb WITH replication = {'class': 'NetworkTopologyStrategy', 'DC1': '1', 'DC2': '1'} ;
> CREATE TABLE testdb.user (id int PRIMARY KEY, name text);
> insert into testdb.user(id, name) values(1, 'bison');
> select * from testdb.user;
```

### 验证设想

#### 设想一：一个IDC故障/宕机，另外一个IDC还可以读写

将香港机器上的docker容器停止，或者使用安全规则使得两个机器不能相互访问，继续在美国的机器上读写数据，实验符合预期。

```sql
cqlsh -ucassandra -pcassandra

> consistency LOCAL_ONE
> insert into testdb.user(id, name) values(2, 'liao');
> select * from testdb.user; #不会报错
> consistency TWO
> select * from testdb.user;  #会报错
```

把香港机器上的docker容器启动起来，很短的时间后，在香港的机器上应该也可以拉取到新插入的数据。并且这是用用TWO/ALL/QUORUM等一致性级别来读写数据都是ok的。符合预期

### 编程访问cassandra

以python为例，拉取必要的驱动/sdk：

```
pip3 install cassandra-driver
```

```python
# -*- encoding: utf-8 -*-
# 引入Cluster模块
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement


auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
cluster = Cluster(["119.28.XXX.XXX"],auth_provider= auth_provider)
# 连接并创建一个会话
session = cluster.connect("testdb")
session.execute("CREATE TABLE testdb.user (    id int PRIMARY KEY,    name text);")
rows = session.execute('select * from user limit 10;')
for row in rows:
    print(row)

for i in range(1000000):
    sql="insert into user(id, name)values(%d,'bison %s')"%(i, i)
    simple_statement = SimpleStatement(sql, consistency_level=ConsistencyLevel.LOCAL_ONE)
    session.execute(simple_statement)
    if (i%10001) == 0:
        print("i=",i)
cluster.shutdown()
# 查看是否关闭连接
print(cluster.is_shutdown)
```

在本机运行，避免远程网络访问的话，测得每秒插入1176行记录

### 参考文档

一致性级别的参考文档：

https://docs.datastax.com/en/archived/cassandra/2.0/cassandra/dml/dml_config_consistency_c.html

一个机器宕机后，cqlsh也无法连接另一个机器的问题解答：

https://stackoverflow.com/questions/44883940/cassandra-cannot-achieve-consistency-level-quorum

如何在一个服务器上利用docker部署多个cassandra节点，但我没有搞定，后来又买了一台机器组成三个机房/节点：

https://www.cnblogs.com/stoneFang/p/6715298.html