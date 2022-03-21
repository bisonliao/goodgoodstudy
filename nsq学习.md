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

