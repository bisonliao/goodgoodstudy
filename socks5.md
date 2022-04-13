### 1、 socks5协议

1. 支持tcp（主动连接出去和监听接受连接）、udp协议，即connect、bind、udp三种模式
2. 支持按域名指定对端地址，而不只是IP。
3. 工作在传输层，所以基于此，可以透明的支持http和https

权威出处：https://www.rfc-editor.org/rfc/rfc1928.txt。 （最短最易读的rfc）

### 2、demo

1. 客户端使用python开发语言，socks5客户端库使用PySocks；另外一个库python_socks不支持udp和bind模式。其实PySocks怎么使用udp我也没有折腾明白，他妈的开源的库都不写像样的文档，靠猜。
2. 代理软件使用3proxy

```python
#!/usr/bin/python3
import socks
import socket
import ssl
import requests

# use requests  with proxy
print('''===================
use requests lib with proxy
===================''')
proxies = {
  "https": "socks5://bisonliao:123456@localhost:1080",
  "http": "socks5://bisonliao:123456@localhost:1080",
}

resp = requests.get("https://www.qq.com", proxies=proxies) #type: requests.Response
print(resp.status_code)
print(resp.content[0:50])


#use raw socket with proxy
print('''===================
use raw socket with proxy
===================''')

s = socks.socksocket(socket.AF_INET, socket.SOCK_STREAM) # Same API as socket.socket in the standard lib


s.set_proxy(socks.SOCKS5, "localhost", username="bisonliao", password="123456", port=1080)


s.connect(("www.baidu.com", 443))
s = ssl.create_default_context().wrap_socket(
    sock=s,
    server_hostname='www.baidu.com'
)

s.send( bytes('''GET / HTTP/1.1
Host:www.baidu.com
User-Agent: Mozilla/5.0

''', "utf-8"))
print(s.recv(100))
```



```shell
git clone https://github.com/z3apa3a/3proxy
cd 3proxy
ln -s Makefile.Linux Makefile
make
make install

#主要的配置文件和目录
#直接执行/etc/3proxy/3proxy.cfg，会启动3proxy，并且chroot到/usr/local/3proxy
#/usr/local/3proxy/conf/3proxy.cfg 是配置文件
#/usr/local/3proxy/conf/add3proxyuser.sh 可以增加账号，并设置密码
#/usr/local/3proxy/logs 可以查看日志

root@VM-16-7-ubuntu:~/3proxy# ps auxw|grep proxy
proxy     255488  0.0  0.1 399616  5080 ?        Sl   14:56   0:03 /bin/3proxy /etc/3proxy/3proxy.cfg

```

### 3、c的demo

还是自己用c写一个来验证udp代理吧。没有考虑性能、并发啥的，就同步执行吧

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

//http://www.faqs.org/rfcs/rfc1928.html



typedef struct
{
    struct sockaddr_in proxy_udp_addr; // udp address on proxy server
    int tcpfd;
    int udpfd;
} proxy_t;

static int proxy_fd_readable(int fd, int ms)
{
    fd_set rset;
    struct timeval tv;
    FD_ZERO(&rset);
    FD_SET(fd, &rset);
    tv.tv_sec = ms/1000;
    tv.tv_usec = (ms%1000)*1000;

    int cnt = select(fd+1, &rset, NULL, NULL, &tv);
    if (cnt > 0)
    {
        if (FD_ISSET(fd, &rset))
        {
            return 1;
        }
    }
    return 0;
}

static int proxy_auth(int sockfd, const char * username, const char * password)
{
    //  https://www.ddhigh.com/2019/08/24/socks5-protocol.html
    unsigned char buf[1024];
    buf[0] = 0x5;
    buf[1] = 0x2;
    buf[2] = 0x0;
    buf[3] = 0x2;

    if (send(sockfd, buf, 4, 0) != 4)
    {
        perror("send");
        return -1;
    }
    int cnt = proxy_fd_readable(sockfd, 1000);
    if (cnt != 1)
    {
        fprintf(stderr, "timeout while auth.\n");
        return -1;
    }
    ssize_t len = recv(sockfd, buf, sizeof(buf), 0);
    if (len < 2 || buf[0] != 0x5)
    {
        fprintf(stderr, "response invalid.\n");
        return -1;
    }
    int method = buf[1];
    if (method == 0) //不需要认证
    {
        return 0;
    }
    else if (method != 0x2) //不是账号密码认证
    {
        fprintf(stderr, "auth method unsupported.\n");
        return -1;
    }
    //账号密码认证
    int offset = 0;
    buf[offset++] = 0x1;//不是0x5
    buf[offset++] = strlen(username);
    memcpy(buf+offset, username, strlen(username));
    offset += strlen(username);
    buf[offset++] = strlen(password);
    memcpy(buf+offset, password, strlen(password));
    offset += strlen(password);
    if (send(sockfd, buf, offset, 0) != offset)
    {
        perror("send username and password:");
        return -1;
    }
    cnt = proxy_fd_readable(sockfd, 1000);
    if (cnt != 1)
    {
        fprintf(stderr, "timeout while read response\n");
        return -1;
    }
    len = recv(sockfd, buf, sizeof(buf), 0);
    if (len < 2 || buf[0] != 0x1)
    {
        fprintf(stderr, "invalid response for auth, len:%ld\n", len);
        return -1;
    }
    if (buf[1] == 0x0)
    {
        return 0;
    }
    fprintf(stderr, "username and password invalid,%d!\n", buf[1]);
    return -1;
}

int proxy_init(proxy_t * proxy, const char * proxy_server_ip, uint16_t proxy_server_port, const char * username, const char * password)
{
    int sockfd;
    struct sockaddr_in servaddr, cli;
    unsigned char buf[1024];
   
    sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd == -1) {
        fprintf(stderr, "socket creation failed...\n");
        return -1;
    }
    
    bzero(&servaddr, sizeof(servaddr));
   
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(proxy_server_ip);
    servaddr.sin_port = htons(proxy_server_port);
   
    
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
        fprintf(stderr, "connection with the server failed...\n");
        return -1;
    }
    if (proxy_auth(sockfd, username, password) != 0)
    {
        close(sockfd);
        return -1;
    }
    memset(proxy, 0, sizeof(proxy_t));
    proxy->tcpfd = sockfd;
    return 0;    
}
int proxy_connect(proxy_t * proxy, const char * tcp_server_ip, uint16_t tcp_server_port)
{
    return 0;
}
int proxy_bind(proxy_t * proxy, const char * bind_ip, uint16_t bind_port)
{
    return 0;
}
int proxy_udp(proxy_t * proxy, const char * udp_server_ip, uint16_t udp_server_port)
{
    unsigned char buf[1024];
    int offset = 0;
    buf[offset++] = 0x5;
    buf[offset++] = 0x3;
    buf[offset++] = 0x0;
    buf[offset++] = 0x1;
    *(in_addr_t*)(buf+offset) = inet_addr(udp_server_ip);
    offset += sizeof(in_addr_t);
    *(uint16_t*)(buf+offset) = htons(udp_server_port);
    offset += sizeof(uint16_t);
    if (send(proxy->tcpfd, buf, offset, 0) != offset)
    {
        perror("send udp associate:");
        return -1;
    }
    int cnt = proxy_fd_readable(proxy->tcpfd, 1000);
    if (cnt != 1)
    {
        fprintf(stderr, "timeout while read udp associate response\n");
        return -1;
    }
    int len = recv(proxy->tcpfd, buf, sizeof(buf), 0);
    if (len < 6 || buf[0]!=0x5)
    {
        fprintf(stderr, "invalid response\n");
        return -1;
    } 
    if (buf[1] != 0x0)
    {
        fprintf(stderr, "udp associate failed!%d\n", buf[1]);
        return -1;
    } 
    if (buf[3] != 0x1)
    {
        fprintf(stderr, "address type unsupported\n");
        return -1;
    }
    bzero(&proxy->proxy_udp_addr, sizeof(proxy->proxy_udp_addr));
   
    proxy->proxy_udp_addr.sin_family = AF_INET;
    proxy->proxy_udp_addr.sin_addr.s_addr = *(in_addr_t*)(buf+4);
    proxy->proxy_udp_addr.sin_port = *(in_port_t*)(buf+8);
    printf("udp address:%s:%d\n", inet_ntoa(proxy->proxy_udp_addr.sin_addr), ntohs(proxy->proxy_udp_addr.sin_port));

    proxy->udpfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (proxy->udpfd == -1) {
        fprintf(stderr, "udp socket creation failed...\n");
        return -1;
    }
    return 0;
}
int proxy_udp_sendto(proxy_t * proxy, const unsigned char * buf, size_t sz, const char *dest_ip, int dest_port)
{
    unsigned char package[8*1024];
    if (sz > 8000)
    {
        fprintf(stderr, "udp package is too long");
        return -1;
    }
    int offset = 0;
    package[offset++] = 0x0;
    package[offset++] = 0x0;
    package[offset++] = 0x0;
    package[offset++] = 0x1;
    *(in_addr_t*)(package+offset) = inet_addr(dest_ip);
    offset += sizeof(in_addr_t);
    *(uint16_t*)(package+offset) = htons(dest_port);
    offset += sizeof(uint16_t);
    memcpy(package+offset, buf, sz);
    offset += sz;


    int iret = sendto(proxy->udpfd, package, offset, 0, (const struct sockaddr*)&proxy->proxy_udp_addr, sizeof(proxy->proxy_udp_addr));
    if (iret != offset)
    {
        perror("sendto");
    }
    return iret;
}

int main(int argc, char**argv)
{
    proxy_t p;
    int iret;
    iret = proxy_init(&p, "127.0.0.1", 1080, "bisonliao", "123456");
    printf("iret = %d\n", iret);
    iret = proxy_udp(&p, "127.0.0.1", 1090);
    printf("iret = %d\n", iret);
    
    while(1)
    {
        iret = proxy_udp_sendto(&p, "hello", 5, "127.0.0.1", 1090);
        printf("iret = %d\n", iret);

        unsigned char buf[1024];
        iret = recv(p.udpfd, buf, sizeof(buf), 0);
        if (iret >=0) buf[iret] = 0;
        printf("iret = %d, %s!\n", iret, buf+10);
        sleep(1);
    }
    return 0;
    
}
```



udp服务器可以用python快速凑一个

```python
#!/usr/bin/python3
import socket

localIP     = "127.0.0.1"
localPort   = 1090
bufferSize  = 1024
msgFromServer       = "Hello UDP Client"
bytesToSend         = str.encode(msgFromServer)

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))

print("UDP server up and listening")

# Listen for incoming datagrams

while(True):
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]

    clientMsg = "Message from Client:{}".format(message)
    clientIP  = "Client IP Address:{}".format(address)
    
    print(clientMsg)
    print(clientIP)

    # Sending a reply to client
    UDPServerSocket.sendto(bytesToSend, address)

```



验证通过：

```shell
ubuntu@VM-16-7-ubuntu:~/practice/socks5$ ./ttx
iret = 0
udp address:127.0.0.1:54332
iret = 0
iret = 15
iret = 26, Hello UDP Client!
iret = 15
iret = 26, Hello UDP Client!
```

