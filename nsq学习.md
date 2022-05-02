#### 基本概念

官网的一个图很好的表达了基本概念之间的关系：

To summarize, messages are multicast from topic -> channel (every channel receives a copy of all messages for that topic) but evenly distributed from channel -> consumers (each consumer receives a portion of the messages for that channel).

即

1. 围绕具体一个topic生产和消费消息
2. topic的消息，对每个关注该topic的channel，完整的复制一份，即每个channel收到该topic的所有消息
3. 同一个channel的多个消费者，比较均等的瓜分channel里的消息

![](https://f.cloud.github.com/assets/187441/1700696/f1434dc8-6029-11e3-8a66-18ca4ea10aca.gif)



#### 用一个机器模拟出多节点nsq集群

##### 方法一：使用docker

```shell
docker run -d --network=host nsqio/nsq  "/nsqlookupd"
sleep 1
docker run -d --network=host nsqio/nsq /nsqd --lookupd-tcp-address='127.0.0.1:4160' --http-address='0.0.0.0:4151' --tcp-address='0.0.0.0:4150'
docker run -d --network=host nsqio/nsq /nsqd --lookupd-tcp-address='127.0.0.1:4160' --http-address='0.0.0.0:4251' --tcp-address='0.0.0.0:4250'
docker run -d --network=host nsqio/nsq /nsqadmin --lookupd-http-address='127.0.0.1:4161' --http-address='0.0.0.0:4171'
```

这样启动的进程，在宿主机层面是可以通过网络直接访问的。

然后就可以一边生产，一边消费了：

```shell
nsq_tail --lookupd-http-address=127.0.0.1:4161 --topic=test
to_nsq -nsqd-tcp-address 127.0.0.1:4150 --topic=test
```



##### 方法二：使用docker-compose

```yaml
version: '3'
services:
  nsqlookupd:
    image: nsqio/nsq
    command: /nsqlookupd
    ports:
      - "4160:4160"
      - "4161:4161"

  nsqd1:
    image: nsqio/nsq
    command: /nsqd --lookupd-tcp-address=nsqlookupd:4160
    depends_on:
      - nsqlookupd
    ports:
      - "4150:4150"
      - "4151:4151"

  nsqd2:
    image: nsqio/nsq
    command: /nsqd --lookupd-tcp-address=nsqlookupd:4160
    depends_on:
      - nsqlookupd
    ports:
      - "4250:4150"
      - "4251:4151"

  nsqadmin:
    image: nsqio/nsq
    command: /nsqadmin --lookupd-http-address=nsqlookupd:4161
    depends_on:
      - nsqlookupd
    ports:
      - "4171:4171"

```

这个方法创建的网络智能是bridge类型的，而且是单独的bridge网络，虽然nsqd都有做端口映射，但客户端从lookupd发现的nsqd的地址是内网地址，例如是172.17.0.4， 是不能够直接访问的，只能启动一个同网络内的docker容器来作为客户端访问，例如：

```shell
docker network ls #看看有哪些网络,不出意外会发现一个叫nsq_default的网络
docker run -it --network=nsq_default nsqio/nsq 'sh'
nsq_tail ... #在容器里面执行，可以直接用nsqadmin/nsqd2这样的hostname代替ip访问网络内的节点
```

