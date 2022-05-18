### tun和tap虚拟网卡



tun设备工作在三层，tap设备工作在二层。



借用网友的一个图：

![img](https://pic2.zhimg.com/80/v2-f74a999198febc8709460b42ef575ad5_720w.jpg)

如图所示，两个应用程序 A 和 B，都处于用户层，其他的设备都处于内核层，数据的流向可以顺着数字方向来查看。

TUN0 是一个虚拟网卡设备，可以看到它与物理网卡相同的地方在于它们的一端都连接内核协议栈，不同的地方在于物理网卡另一端连接的是外面的交换机或者路由器等硬件设备，TUN0 连接的是处于用户层的应用程序，协议栈发送给虚拟网卡的数据都被发送到了应用程序中，通过加密，转换，校验等方式后通过物理网卡发送出去。

用户层的应用程序 A 发送了一个普通的数据包，socket 将数据包发送给内核协议栈，内核协议栈根据目的地址进行路由，查询路由表，发现数据包的下一跳地址应该为 TUN0 网卡，所以内核协议栈将数据包发送给虚拟网卡设备 TUN0，TUN0 接收到数据之后通过某种方式从内核空间将数据发送给运行在用户空间的应用程序 B，B 收到数据包后进行一些处理，然后构造一个新的数据包，通过 socket 发送给内核协议栈，这个新的数据包的目的地址变成了一个外部地址，源地址变成了 eth0 的地址，内核协议栈通过查找路由表之后发现目的地址找不到，就会将数据包通过 eth0 网卡发送给网关，eth0 接收到数据之后将数据包发送到和 eth0 网卡物理相连的外部设备。

用户层应用程序B如果往tun0里写入一个IP包，这个IP包会从tun0网卡收到，然后向上走协议栈层层解析，到达应用程序。

#### 一、tun设备



```shell
#确认内核安装了tun/tap
modinfo tun  #不需要额外modinfo tap，也不需要/dev/net/tap的存在，有/dev/net/tun就可以

ip tuntap add dev tun0 mode tun
ip addr add 192.168.1.10/24 dev tun0
ip link set tun0 up
ip route  #确认一下路由表是否ok
```



```c
//简单的驱动程序，读虚拟网卡的包
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <linux/if_tun.h>
#include <linux/if.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int tun_alloc(char *dev, int flags) {

  struct ifreq ifr;
  int fd, err;
  char *clonedev = "/dev/net/tun";
   if( (fd = open(clonedev, O_RDWR)) < 0 ) { /* 使用读写方式打开 */
     return fd;
   }

   memset(&ifr, 0, sizeof(ifr));

   ifr.ifr_flags = flags;   /* IFF_TUN or IFF_TAP, plus maybe IFF_NO_PI */

   if (*dev) {
     strncpy(ifr.ifr_name, dev, IFNAMSIZ); /* 设置设备名称 */
   }

   if( (err = ioctl(fd, TUNSETIFF, (void *) &ifr)) < 0 ) {
     close(fd);
     return err;
   }

  strcpy(dev, ifr.ifr_name);

  return fd;
}

int analyze_iphdr(const char * buf, char * src, char *dest, uint8_t * proto)
{
    uint8_t c = buf[0];
    uint8_t version = c & 0x0f;
    uint8_t hdrlen = c >> 4;
    *proto = *(uint8_t*)(buf+9);
    uint16_t totalLen = *(uint16_t*)(buf+2);
    totalLen = ntohs(totalLen);

    struct in_addr srcIP, destIP;
    srcIP.s_addr = *(uint32_t*)(buf+12);
    destIP.s_addr = *(uint32_t*)(buf+16);

    strcpy(src, inet_ntoa(srcIP));
    strcpy(dest, inet_ntoa(destIP));

    printf("ipv%d, hdrlen:%d, totalLen:%d, svc:%u, [%s]->[%s]\n", version, hdrlen, totalLen, *proto, src, dest);

    return 0;

}

int main()
{
        #define IFRAME_SZ 5000
        char tun_name[IFRAME_SZ];

        strcpy(tun_name, "tun0");
        int tun_fd = tun_alloc(tun_name, IFF_TUN | IFF_NO_PI);  /* tun interface */

        if(tun_fd < 0){
                perror("Allocating interface");
                exit(1);
        }

        while(1) {
                char buffer[5000];
                int nread = read(tun_fd,buffer,sizeof(buffer));
                if(nread < 0) {
                        perror("Reading from interface");
                        close(tun_fd);
                        exit(1);
                }

                printf("Read %d bytes from device %s\n", nread, tun_name);
                uint8_t proto;
                char src[100];
                char dest[100];
                analyze_iphdr(buffer, src, dest, &proto);
        }
        return 0;
}

```

```shell
gcc -o ttx a.c
./ttx 

#在本级的另外一个终端，可以看到ttx有输出
ping 192.168.1.12

#同时可抓包
tshark -i tun0
```

#### 二、tap设备

tap设备工作在二层，链路层。

```shell
ip tuntap add dev tap0 mode tap
ip addr add 192.168.2.10/24 dev tap0
ip link set tap0 up
ip route  #确认一下路由表是否ok
```

一个简单的驱动：

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <linux/if_tun.h>
#include <linux/if.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int tun_alloc(char *dev, int flags) {

  struct ifreq ifr;
  int fd, err;
  char *clonedev = "/dev/net/tun";

   if( (fd = open(clonedev, O_RDWR)) < 0 ) {
     printf("%d\n", __LINE__);
     return fd;
   }

   memset(&ifr, 0, sizeof(ifr));

   ifr.ifr_flags = flags;   /* IFF_TUN or IFF_TAP, plus maybe IFF_NO_PI */

   if (*dev) {
     strncpy(ifr.ifr_name, dev, IFNAMSIZ);
   }

   if( (err = ioctl(fd, TUNSETIFF, (void *) &ifr)) < 0 ) {
     close(fd);
     printf("%d\n", __LINE__);
     return err;
   }

  strcpy(dev, ifr.ifr_name);

  return fd;
}

int analyze_machdr(const char * buf)
{
    int i;
    printf("dst mac");
    for (i = 0; i < 6; i++)
    {
        printf(":%02x", *(uint8_t*)(buf+i));
    }
    printf("\t");
    printf("src mac");
    for (i = 0; i < 6; i++)
    {
        printf(":%02x", *(uint8_t*)(buf+i+6));
    }
    printf("\t");
    uint16_t type = *(uint16_t*)(buf+12);
    type = ntohs(type);
    printf("type:%04x\n", type);

    return 0;

}

int main()
{
        #define IFRAME_SZ 5000
        char tun_name[IFRAME_SZ];

        strcpy(tun_name, "tap0");
        int tun_fd = tun_alloc(tun_name, IFF_TAP | IFF_NO_PI);  /* tun interface */

        if(tun_fd < 0){
                perror("Allocating interface");
                exit(1);
        }

        while(1) {
                char buffer[5000];
                int nread = read(tun_fd,buffer,sizeof(buffer));
                if(nread < 0) {
                        perror("Reading from interface");
                        close(tun_fd);
                        exit(1);
                }

                printf("Read %d bytes from device %s\n", nread, tun_name);
                analyze_machdr(buffer);
        }
        return 0;
}
```

简单测试：

```shell
gcc -o ttx b.c
./ttx 

#在本级的另外一个终端，可以看到ttx有输出
ping 192.168.2.12

#同时可抓包
tshark -i tap0
```

参考文档：

```
https://backreference.org/2010/03/26/tuntap-interface-tutorial/
https://zhuanlan.zhihu.com/p/260405786
```

