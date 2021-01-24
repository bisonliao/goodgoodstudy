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

#define PORT 8888
#define MAXSIZE 1024


//set flag to keep fd open after execl
static void fdCrossExec(int fd)
{
    int flag = fcntl(fd, F_GETFD);
    flag = flag &(~FD_CLOEXEC);
    fcntl(fd, F_SETFD, flag);
}

static int sfd = -1;  // fd to listen
static int client = -1; //client fd
const char ENV_KEY[] = "PASS_FD_BY_ENV";  //pass fd through enviroment variable, the  key

// get fd from enviroment variable
static void getfdFrmEnv()
{
    char * val = getenv(ENV_KEY);
    if (val == NULL)
    {
        return;
    }
    printf("%s=%s\n", ENV_KEY, val);
    sscanf(val, "%d,%d", &sfd, &client);
    printf("I get fd from env:%d, %d\n", sfd, client);
    if (sfd >= 0)
    {
        fdCrossExec(sfd);
    }
    if (client >= 0)
    {
        fdCrossExec(client);
    }
}
// set fd to enviroment variable
static void setfd2Env()
{
    char val[100];
    snprintf(val, sizeof(val)-1, "%d,%d", sfd, client);
    setenv(ENV_KEY, val, 1);
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
    if (pid > 0)
    {
        printf("old version process is exiting...\n");
        _exit(0);
    }
    setfd2Env();
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
    
    fdCrossExec(0);
    fdCrossExec(1);
    fdCrossExec(2);

    getfdFrmEnv();

    if (sfd < 0) // prev version process does NOT pass valid fd to me
    {
	    // create / bind /listen 
	    sfd = socket(AF_INET, SOCK_STREAM, 0);
	    if (sfd == -1) {
	        printf("create socket error!\n");
	        return -1;
	    }
	    fdCrossExec(sfd);

	    struct sockaddr_in saddr;
	    memset(&saddr, 0, sizeof(saddr));
	    saddr.sin_family = AF_INET;
	    saddr.sin_port = htons(PORT);
	    saddr.sin_addr.s_addr = INADDR_ANY;

	    if (bind(sfd, (struct sockaddr *)&saddr, sizeof(struct sockaddr)) == -1) {
	        printf("bind socket error!, errno=%s\n", strerror(errno));
	        return -1;
	    }
	
	    if (listen(sfd, 8) == -1) {
	        printf("listen error!\n");
	        return -1;
	    }
    }
    // address for peer/client
    struct sockaddr_in peer_saddr;
    socklen_t peer_len = sizeof(struct sockaddr);
    memset(&peer_saddr, 0, sizeof(peer_saddr));

    printf("set signal handle...\n");
    installUSR1();

    while (1)
    {
        if (client < 0) // previous version process does NOT pass valid fd to me
        {
            client = accept(sfd, (struct sockaddr *)&peer_saddr, &peer_len);
        }
        if (client == -1) {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            printf("accept error!,errno=%s\n", strerror(errno));
            return -1;
        }
        printf("accept client: %s:%d\n", inet_ntoa(peer_saddr.sin_addr), peer_saddr.sin_port);
        fdCrossExec(client);

        for (;;) {
            char buffer[MAXSIZE] = {0};
            int len = read(client, buffer, MAXSIZE);
            if (len <= 0)
            {
                close(client);
                client = -1;
                break;
            }
            printf("read:%s\n", buffer);
            write(client, buffer, strlen(buffer));
            if (strcmp(buffer, "q") == 0) {
                printf("close server!\n");
                close(client);
                client = -1;
                break;
            }
        }
    }
    close(sfd);
    sfd = -1;
    return 0;
}
