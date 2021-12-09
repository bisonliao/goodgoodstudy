### tun和tap虚拟网卡



借用网友的一个图：

![img](https://pic2.zhimg.com/80/v2-f74a999198febc8709460b42ef575ad5_720w.jpg)

如图所示，两个应用程序 A 和 B，都处于用户层，其他的设备都处于内核层，数据的流向可以顺着数字方向来查看。

TUN0 是一个虚拟网卡设备，可以看到它与物理网卡相同的地方在于它们的一端都连接内核协议栈，不同的地方在于物理网卡另一端连接的是外面的交换机或者路由器等硬件设备，TUN0 连接的是处于用户层的应用程序，协议栈发送给虚拟网卡的数据都被发送到了应用程序中，通过加密，转换，校验等方式后通过物理网卡发送出去。

```shell
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

```
gcc -o ttx a.c
./ttx 

#在本级的另外一个终端
ping 192.168.1.12
```

参考文档：

```
https://backreference.org/2010/03/26/tuntap-interface-tutorial/
https://zhuanlan.zhihu.com/p/260405786
```

