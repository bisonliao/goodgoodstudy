#include <sys/types.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/un.h>

#define PORT 8888
#define MAXSIZE 1024

static int listenfd = -1;  // fd to listen
static int clientfd = -1; //clientfd fd

// unix domain socket , for passing fd 
const char* udsfile = "/tmp/echo.sock";
static udsfd4recv = -1;
static int openForSend()
{
    int fd = socket(AF_UNIX,SOCK_DGRAM,0);

    if (fd < 0)
    {
        perror("socket");
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr,0,sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, udsfile);
    int len = sizeof(addr);

    if (connect(fd, (const struct sockaddr *)&addr, (socklen_t)len) < 0)
    {
        perror("connect:");
        return -1;
    }

    return fd;
}
static int openForRecv()
{
    int fd = socket(AF_UNIX,SOCK_DGRAM,0);

    if (fd < 0)
    {
        perror("socket");
        return -1;
    }
    unlink(udsfile);
    fcntl(fd, F_SETFL, O_NONBLOCK);

    struct sockaddr_un addr;
    memset(&addr,0,sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, udsfile);
    int len = sizeof(addr);

    if(bind(fd,(struct sockaddr*)&addr,sizeof(addr)) < 0)
    {
        perror("bind");
        return -1;
    }

    return fd;
}
// recv fd from udsfd, udsfd is a unix domain socket fd
static int recvfd(int udsfd)
{
    struct msghdr msg;
    struct iovec iov[1];
    char buf[100];
    char *testmsg = "test msg.\n";
    
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;
    struct cmsghdr *pcmsg;
    int recvfd;

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    iov[0].iov_base = buf;
    iov[0].iov_len = 100;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    int ret = recvmsg(udsfd, &msg, 0);
    if (ret < 0) 
    {
        perror("recvmsg:");
        return ret;
    }
    if (ret == 0)
    {
        fprintf(stderr, " failed to recvmsg\n");
        return -1 ;
    }
    printf("recv fd name:%s\n", buf);

    if ((pcmsg = CMSG_FIRSTHDR(&msg)) != NULL && (pcmsg->cmsg_len == CMSG_LEN(sizeof(int)))) 
    {
            if (pcmsg->cmsg_level != SOL_SOCKET) {
                    printf("cmsg_leval is not SOL_SOCKET\n");
                    return -1;
            }

            if (pcmsg->cmsg_type != SCM_RIGHTS) {
                    printf("cmsg_type is not SCM_RIGHTS");
                    return -1;
            }

            recvfd = *((int *) CMSG_DATA(pcmsg));
            printf("recv fd = %d\n", recvfd);
            return recvfd;
    }
    return -1;
}
// send fd by udsfd, udsfd is a unix domain socket fd
static int sendfd(int udsfd, int fd, const char *fdname)
{
    struct msghdr msg;
    struct iovec iov[1];
    char buf[100];
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;
    struct cmsghdr *pcmsg;

    strcpy(buf, fdname);

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    iov[0].iov_base = buf;
    iov[0].iov_len = strlen(buf);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    
    pcmsg = CMSG_FIRSTHDR(&msg);
    pcmsg->cmsg_len = CMSG_LEN(sizeof(int));
    pcmsg->cmsg_level = SOL_SOCKET;
    pcmsg->cmsg_type = SCM_RIGHTS;
    *((int *)CMSG_DATA(pcmsg)) = fd;
    
    int ret = sendmsg(udsfd, &msg, 0);
    return ret;
}

// upgrade as soon as recv sigusr1
static void upgrade(int sig)
{
    if (sig != SIGUSR1)
    {
        return;    
    }
    printf("begin upgrade...\n");


    int pid = fork();
    if (pid  < 0)
    {
        return;
    }
    if (pid > 0) // parent
    {
        int i;
        for ( i = 0; i < 3; i++)
        {
            // wait for unix domain socket established
            if (access(udsfile, F_OK) == 0)
            {
                break;
            }
            sleep(1);
        }
        int udsfd = openForSend();
        if (udsfd < 0)
        {
            _exit(0);
        }
        sendfd(udsfd, listenfd, "listen fd");
        sendfd(udsfd, clientfd, "client fd");

        close(udsfd);

        _exit(0);
    }
    execl("./newecho", "newecho", "hahaha", NULL);
    printf("upgrade failed!\n");
    _exit(0);
}
// set sigusr1 handler, make sure it is not blocked
static void installUSR1()
{
    sigset_t  sigset;
    sigemptyset (&sigset);
    sigaddset(&sigset, SIGUSR1);
    sigprocmask (SIG_UNBLOCK, &sigset, NULL);

    signal(SIGUSR1, upgrade);

}
int main(int argc, char const *argv[])
{
    if (argc > 1)
    {
        printf("%s\n", argv[1]);
    }
    // create unix domain socket ,wait for fd  for 2 seconds
    int udsfd = openForRecv();
    if (udsfd < 0)
    {
        printf("failed to create unix domain fd\n");
        return -1;
    }
    sleep(2);
    listenfd = recvfd(udsfd);
    clientfd = recvfd(udsfd);
    printf("I receiv two fd:%d, %d\n", listenfd, clientfd);
    close(udsfd);
    unlink(udsfile);

    if (listenfd < 0) // prev version process does NOT pass valid fd to me
    {
	    // create / bind /listen 
	    listenfd = socket(AF_INET, SOCK_STREAM, 0);
	    if (listenfd == -1) {
	        printf("create socket error!\n");
	        return -1;
	    }

	    struct sockaddr_in saddr;
	    memset(&saddr, 0, sizeof(saddr));
	    saddr.sin_family = AF_INET;
	    saddr.sin_port = htons(PORT);
	    saddr.sin_addr.s_addr = INADDR_ANY;

	    if (bind(listenfd, (struct sockaddr *)&saddr, sizeof(struct sockaddr)) == -1) {
	        printf("bind socket error!, errno=%s\n", strerror(errno));
	        return -1;
	    }
	
	    if (listen(listenfd, 8) == -1) {
	        printf("listen error!\n");
	        return -1;
	    }
    }
    // address for peer/clientfd
    struct sockaddr_in peer_saddr;
    socklen_t peer_len = sizeof(struct sockaddr);
    memset(&peer_saddr, 0, sizeof(peer_saddr));

    printf("set signal handle...\n");
    installUSR1();

    while (1)
    {
        if (clientfd < 0) // previous version process does NOT pass valid fd to me
        {
            clientfd = accept(listenfd, (struct sockaddr *)&peer_saddr, &peer_len);
        }
        if (clientfd == -1) {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            printf("accept error!,errno=%s\n", strerror(errno));
            return -1;
        }
        printf("accept client: %s:%d\n", inet_ntoa(peer_saddr.sin_addr), peer_saddr.sin_port);

        for (;;) {
            char buffer[MAXSIZE] = {0};
            int len = read(clientfd, buffer, MAXSIZE);
            if (len <= 0)
            {
                shutdown(clientfd, SHUT_WR);
                close(clientfd);
                clientfd = -1;
                break;
            }
            printf("read:%s\n", buffer);
            write(clientfd, buffer, strlen(buffer));
            if (strcmp(buffer, "q") == 0) {
                printf("close server!\n");
                if (shutdown(clientfd, SHUT_WR))  { perror("shutdown");}
                if (close(clientfd))  { perror("close");}
                clientfd = -1;
                break;
            }
        }
    }
    close(listenfd);
    listenfd = -1;
    return 0;
}
