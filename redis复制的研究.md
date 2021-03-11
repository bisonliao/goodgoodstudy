### 问题背景：

希望对两个redis集群做复制，即从集群A复制数据到集群B，redis目前不支持。也有网友在问，无人回答：

https://stackoverflow.com/questions/59943816/redis-cluster-to-cluster-another-replication

阿里云的全球分布式缓存产品解决了这个问题，但是我们的服务不希望绑定到一个固定的云商。

那我们自己来研究一下redis集群间复制吧，初步设想：

1. 连接到Cluster的一台redis，执行cluster nodes命令，就会发现cluster里的所有master IP / port，然后伪造成多个slave去连这些master ip/port，接收到的同步命令，都写入另外一个cluster
2. 有个难点就是，我们自己的工具从cluster A复制过来，写入cluster B，不要又同步出去了。避免形成环形的复制

### 研究结论：

**设想成立，redis复制的协议很简单，可以自己实现，或者使用网友的开源组件。**



关于redis复制介绍的官方文档：

https://redis.io/topics/replication

关于redis cs协议和复制协议的说明文档（很重要）：

https://redis.io/topics/protocol

简单的说：

1. slave连上master后，发送SYNC命令或者PSYNC命令（后者支持partial同步，即带参数指定从某个位置开始同步，而不是全量的同步）。
2. master有一个固定的replicaID，每个变更有一个offset，顺序递增。
3. master如果发现需要做全量的大数据同步，就会dump一个RDB文件，把文件传输给slave。否则就是一条一条的变更发送给slave。
4. 协议格式遵循上面官网上链接的说明，例如*2表示数组的元素个数，$4表示字符串的长度，\r\n分隔各个字段

简单的用redis-cli来模拟一下：

```shell
172.19.16.7:7001> PSYNC ? -1
Entering slave output mode...  (press Ctrl-C to quit)
SYNC with master, discarding 198062 bytes of bulk transfer... ###注意这里，实际上背后有大量存量数据的同步，也就是RDB
SYNC done. Logging commands from master.
"PING"
"PING"
"PING"
"PING"
"SELECT","0"
"HSET","bison1","name","liaonianbo"  #在另外一个客户端写入，会在这里被同步
"PING"
"SET","hello","world2"
"PING"
```

用一段简陋的python代码也可以模拟和体验：

```python
# -*- coding: utf-8 -*-
import select
import socket
import queue

# Create a TCP/IP socket
sock = [0,1,2]
sock[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)# type: socket
server_address = ('119.28.xxxx', 7001)
sock[0].connect(server_address)
sock[0].setblocking(False)
sock[0].send(bytes("PSYNC ? -1\n", "utf-8"))

sock[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('119.28.xxxx', 7002)
sock[1].connect(server_address)
sock[1].setblocking(False)
sock[1].send(bytes("PSYNC ? -1\n", "utf-8"))

sock[2] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('119.28.xxxx', 7003)
sock[2].connect(server_address)
sock[2].setblocking(False)
sock[2].send(bytes("PSYNC ? -1\n", "utf-8"))




while True:
    # readable fd
    inputs = sock
    readable, writeable, exceptional = select.select(inputs, [], inputs)
    for r in readable:
        data = r.recv(1024)#type:bytes
        if len(data) > 0:
            print(r.getpeername(), " received:", data)
        #流式的数据，严谨的方式应该要拆解出完整的包，先临时这么写
```

有热心网友实现了复制组件，还挺好用的，例如这个：

https://github.com/leonchen83/redis-replicator

写一段体验的代码：

```java
package com.company;

import com.moilioncircle.redis.replicator.Configuration;
import com.moilioncircle.redis.replicator.RedisReplicator;
import com.moilioncircle.redis.replicator.Replicator;
import com.moilioncircle.redis.replicator.cmd.impl.HSetCommand;
import com.moilioncircle.redis.replicator.cmd.impl.SetCommand;
import com.moilioncircle.redis.replicator.event.Event;
import com.moilioncircle.redis.replicator.event.EventListener;
import com.moilioncircle.redis.replicator.event.PostRdbSyncEvent;
import com.moilioncircle.redis.replicator.event.PreRdbSyncEvent;
import com.moilioncircle.redis.replicator.io.RawByteListener;
import com.moilioncircle.redis.replicator.rdb.datatype.KeyStringValueHash;
import com.moilioncircle.redis.replicator.rdb.datatype.KeyStringValueString;
import com.moilioncircle.redis.replicator.rdb.skip.SkipRdbVisitor;
import com.sun.corba.se.impl.orb.PrefixParserAction;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URISyntaxException;

public class Replica {

    public static void main(String[] args) throws IOException, URISyntaxException {
        com.moilioncircle.redis.replicator.Configuration configuration = Configuration.defaultSetting();
        //这个configure很有用，可以用来设置同步的鉴权密码、可以设置SSL证书等
        
        //希望能够通过offset实现断点续传，加了没有起作用，又加下面的RepID，结果异常了
        //configuration.setReplOffset(215); 
        //configuration.setReplId("a1962b36d6a7bcf6b2fe660127bc3b4efbcd10ca");
        Replicator replicator = new RedisReplicator("119.28.xxxx", 7001, configuration);

        replicator.addEventListener(new EventListener() {
            @Override
            public void onEvent(Replicator rep, Event event) {


                System.out.println(">>>"+event+","+rep.getConfiguration().getReplId());

                if (event instanceof KeyStringValueString) {
                    KeyStringValueString kv = (KeyStringValueString) event;
                    System.out.println(new String(kv.getKey()));
                    System.out.println(new String(kv.getValue()));
                    Long offset1 = kv.getContext().getOffsets().getV1();
                    Long offset2 = kv.getContext().getOffsets().getV2();
                    System.out.println("offset:"+offset1 + ", "+offset2);

                }
                else if (event instanceof KeyStringValueHash){
                    KeyStringValueHash kh = (KeyStringValueHash) event;
                    System.out.println(new String(kh.getKey()));
                    System.out.println(kh.getValue());
                    Long offset1 = kh.getContext().getOffsets().getV1();
                    Long offset2 = kh.getContext().getOffsets().getV2();
                    System.out.println("offset:"+offset1 + ", "+offset2);
                }
                else if (event instanceof HSetCommand){
                    HSetCommand cmd = (HSetCommand)event;
                    System.out.println(new String(cmd.getKey()));
                    System.out.println(cmd.getFields());
                    Long offset1 = cmd.getContext().getOffsets().getV1();
                    Long offset2 = cmd.getContext().getOffsets().getV2();
                    System.out.println("offset:"+offset1 + ", "+offset2);
                }
                else if (event instanceof SetCommand){
                    SetCommand cmd = (SetCommand)event;
                    System.out.println(new String(cmd.getKey()));
                    System.out.println(cmd.getValue());
                    Long offset1 = cmd.getContext().getOffsets().getV1();
                    Long offset2 = cmd.getContext().getOffsets().getV2();
                    System.out.println("offset:"+offset1 + ", "+offset2);
                }
            }
        });

        replicator.open();
    }

}
```