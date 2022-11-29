

### 先说背景和结论：

一个业务，服务器pod会给多个客户端通过tcp每秒下发20个业务包，带宽达到4Mbps。这种情况下K8S和istio性能表现如何呢？

测试方式：

1. server和client都是单线程，client和server建立100个tcp连接，server发 client收。每次应用态的读写buffer大小为100kB。单进程单线程
2. 机型：两台腾讯云S5se.2XLARGE16,  每台配置8核16GB，同vpc内网通信。测试带宽吞吐能力与机型明显有关，所以各种场景要统一机型。

结论：

1. **从业务层面看，带宽没有因为k8s的网络插件和istio的边车导致吞吐能力下降，单进程单线程吞吐能力在3Gbps以上。**
2. **腾讯云的global router和VPC-CNI的两种网络插件，性能相差不大**
3. **K8S和istio的组件会引入额外的计算开销，主要是envoy**



###  情况一：直接使用虚拟机

测试结果：

1. 带宽跑到：
   1. client侧观测到入带宽3078Mbps， 
   2. server侧监控到出带宽3355Mbps，比client大是因为有tcp重试？
2. CPU占用：
   1. client 28%cpu（0.28个核），8核服务器空闲率95%；
   2. server 40%cpu（0.40个核），8核服务器空闲率92%；



### 情况二：使用K8S Global Router插件

注意通过nodeAffinity设置client和server在不同的两台机器：

```shell
kubectl get nodes --show-labels
kubectl label nodes 172.16.255.19 tcprole=client
kubectl get deployment tcpclient -o yaml  >client.yaml

#编辑client.yaml， 在client.yaml里增加这样的设置
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: tcpclient
        qcloud-app: tcpclient
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: tcprole
                operator: In
                values:
                - client
      containers:
      - image: ccr.ccs.tencentyun.com/tsf_2202020202/tcpclient
        imagePullPolicy: Always

kubectl apply -f client.yaml
```

测试结果：

1. 带宽跑到：
   1. client侧观测到3072Mbps ，从client侧看没有明显下降
   2. server侧观测到3122Mbps，有点下降
2. CPU占用：
   1. client 40%cpu（0.40个核），8核node显示空闲率93%，说明K8S本身开销不大；
   2. server 45%cpu（0.45个核）， 8核node显示空闲率90%，说明K8S本身开销不大；

**结论：带宽没有明显下降，但client收包的CPU占用多了一些。**

### 情况三：使用K8S Global Router插件，且tcp流量经过istio sidecar

测试结果：

1. 带宽跑到：
   1. client侧观测到3026Mbps ，
   2. server侧观测到3205Mbps，
2. CPU占用：
   1. client 30%cpu（0.3个核），envoy 80%（0.8个核） 8核node显示空闲率83%；
   2. server 15%cpu（0.15个核），envoy 48%（0.48个核）， 8核node显示空闲率85%

**结论：带宽没有明显性能下降，但envoy引入了更多的计算开销很明显，但业务进程因此而节约了计算开销。**

### 情况四：使用K8S VPC-CNI 共享网卡多IP方式

1. 带宽跑到：
   1. client侧观测到3141Mbps 
   2. server侧观测到3378Mbps
2. CPU占用：
   1. client 35%cpu（0.35个核），8核node显示空闲率93%，说明K8S本身开销不大；
   2. server 41%cpu（0.41个核）， 8核node显示空闲率91%，说明K8S本身开销不大；

**结论：带宽没有明显性能下降，和global router相当。**

有个小tips：由于VPC-CNI方式下，pod使用了同VPC下其他ip,  从腾讯云的node监控上看不到带宽，只能自己在pod里安装sar工具查看：

```
apt update
apt install -y sysstat
sar -n DEV 2 10
```

### 情况五：使用K8S VPC-CNI 共享网卡多IP方式，且tcp流量经过istio envoy

**结论：与情况三相当**



### 情况六：使用K8S global router ，用node port暴露tcpserver，client从虚拟机上向NodePort发起请求，跨Node访问tcpserver

这时候需要三台虚拟机，假设叫A，B，C，都在K8S集群里。tcpserver的pod部署在C，对应的service开了NodePort端口是32000，从虚拟机A编译一个client，把请求发送给node B，端口为32000。B会把流量转发到C上面的pod。

1. 带宽跑到：
   1. client侧观测到3075Mbps 
   2. server侧观测到3281Mbps
2. CPU占用：
   1. client 35%cpu（0.35个核），8核node显示空闲率93%，说明K8S本身开销不大；
   2. server 34%cpu（0.34个核）， 8核node显示空闲率94%，说明K8S本身开销不大；
   3. 转发NodePort流量的Node B，cpu空闲率97%

**结论：node port引入的转发，开销也很小**

### 附录测试代码：

client.c

```c
#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/epoll.h>
#include <time.h>

static int peers[65536];
#define  CONN_NUM (100)
static char buffer[102400];

int main(int argc, char *argv[]) {
   int sockfd, portno, n;
   struct sockaddr_in serv_addr;
   struct hostent *server;
   
   
   
   if (argc < 3) {
      fprintf(stderr,"usage %s hostname port\n", argv[0]);
      exit(0);
   }
	
   portno = atoi(argv[2]);
   
   
	
   server = gethostbyname(argv[1]);
   
   if (server == NULL) {
      fprintf(stderr,"ERROR, no such host\n");
      exit(0);
   }
   
   bzero((char *) &serv_addr, sizeof(serv_addr));
   serv_addr.sin_family = AF_INET;
   bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
   serv_addr.sin_port = htons(portno);

   int epfd = epoll_create(CONN_NUM);
   if (epfd < 0)
   {
       perror("epoll_create:");
       exit(1);
   }

   for (int i = 0; i < CONN_NUM; ++i)
   {
       /* Create a socket point */
       sockfd = socket(AF_INET, SOCK_STREAM, 0);

       if (sockfd < 0)
       {
           perror("ERROR opening socket");
           exit(1);
       }
       /* Now connect to the server */
       if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
       {
           perror("ERROR connecting");
           exit(1);
       }
       printf("connected, %d\n", sockfd);
       struct epoll_event event;
       memset(&event, 0, sizeof(event));
       event.data.fd = sockfd;
       event.events |= EPOLLIN;
       if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, sockfd, &event))
       {
           perror("epoll_ctl");
           exit(1);
       }
   }
   uint64_t totalLen = 0;
   time_t startTime = time(NULL);
   for (;;)
   {
        static struct epoll_event event2[CONN_NUM];
        memset(&event2, 0, sizeof(event2));
        int iret = epoll_wait(epfd, event2, CONN_NUM, 100);
        if (iret < 0)
        {
            perror("epoll_wait");
            printf("fuck\n");
            break;
        }
        //printf("%d event occured\n", iret);
        if (iret > 0 )
        {
            for (int i = 0; i < iret; ++i)
            {
                int readLen = read(event2[i].data.fd, buffer, sizeof(buffer));
                if (readLen > 0)
                {
                    totalLen += readLen;
                }
                else
                {
                    int clientfd = event2[i].data.fd;
                    
                    struct epoll_event event;
                    memset(&event, 0, sizeof(event));
                    event.data.fd = clientfd;
                    event.events |= EPOLLIN;
                    if (0 != epoll_ctl(epfd, EPOLL_CTL_DEL, clientfd, &event))
                    {
                        perror("epoll_ctl");
                        exit(1);
                    }
                    close(clientfd);
                }
            }
        }
        time_t current = time(NULL);
        if (current > startTime && (current - startTime) > 30)
        {
            printf("avg bindwidth:%.2fMbps\n", ((double)totalLen) / (current-startTime)/1000000*8);
            startTime = current;
            totalLen = 0;
        } 
   }

   return 0;
}
```

server.c

```c
#include <netdb.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>

#define MAX_CONN_NUM (10000)

static char buffer[102400] = {0};

/** Returns true on success, or false if there was an error */
int SetSocketBlockingEnabled(int fd, int blocking)
{
   if (fd < 0) return -1;


   int flags = fcntl(fd, F_GETFL, 0);
   if (flags == -1) return -1;
   flags = blocking > 0 ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
   return (fcntl(fd, F_SETFL, flags) == 0) ? 0 : -1;

}

int main(int argc, char *argv[])
{
    int sockfd, newsockfd, portno, clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    int epfd;

    /* First call to socket() function */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
    {
        perror("ERROR opening socket");
        exit(1);
    }

    /* Initialize socket structure */
    bzero((char *)&serv_addr, sizeof(serv_addr));
    portno = 50012;

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    /* Now bind the host address using bind() call.*/
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        perror("ERROR on binding");
        exit(1);
    }

    /* Now start listening for the clients, here process will
     * go in sleep mode and will wait for the incoming connection
     */

    listen(sockfd, 5);
    clilen = sizeof(cli_addr);

    epfd = epoll_create(100);
    if (epfd < 0)
    {
        perror("epoll_create:");
        exit(1);
    }
    struct epoll_event event;
    memset(&event, 0, sizeof(event));
    event.data.fd = sockfd;
    event.events |= EPOLLIN;
    if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, sockfd, &event))
    {
        perror("epoll_ctl");
        exit(1);
    }
    uint64_t totalLen = 0;
    time_t startTime = time(NULL);
    for (;;)
    {
        static struct epoll_event event2[MAX_CONN_NUM];
        memset(&event2, 0, sizeof(event2));
        int iret = epoll_wait(epfd, event2, sizeof(event2)/sizeof(event2[0]), 100);
        if (iret < 0)
        {
            perror("epoll_wait");
            printf("fuck\n");
            break;
        }
        for (int i = 0; i < iret; ++i)
        {
            if (sockfd == event2[i].data.fd)
            {
                newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
                if (newsockfd < 0)
                {
                    perror("ERROR on accept");
                    printf("you\n");
                    break;
                }
                SetSocketBlockingEnabled(newsockfd, 0);
                printf("new socket:%d\n", newsockfd);

                struct epoll_event event;
                memset(&event, 0, sizeof(event));
                event.data.fd = newsockfd;
                event.events |= EPOLLOUT;
                if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, newsockfd, &event))
                {
                    perror("epoll_ctl add newsockfd");
                    exit(1);
                }
            }
            else
            {
                int wrLen = write(event2[i].data.fd, buffer, sizeof(buffer));
                if (wrLen < 0)
                {
                    if (errno != EAGAIN && errno != EWOULDBLOCK)
                    {
                        
                        struct epoll_event event;
                        memset(&event, 0, sizeof(event));
                        event.data.fd = event2[i].data.fd;
                        event.events |= EPOLLOUT;
                        if (0 != epoll_ctl(epfd, EPOLL_CTL_DEL, event2[i].data.fd, &event))
                        {
                            perror("epoll_ctl");
                            exit(1);
                        }
                        close(event2[i].data.fd);
                        printf("close %d\n", event2[i].data.fd);
                    }
                }
                else
                {
                    totalLen += wrLen;
                } 
            }
            
        }
       
        time_t current = time(NULL);
        if (current > startTime && (current - startTime) > 30)
        {
            printf("avg bindwidth:%.2fMbps\n", ((double)totalLen) / (current-startTime)/1000000*8);
            startTime = current;
            totalLen = 0;
        } 
    }
    printf("closing...\n");
    
    return 0;
}
```

