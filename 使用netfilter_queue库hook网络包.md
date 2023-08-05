话不多说，直接上代码：

```shell
 apt install libnetfilter-queue-dev
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include <linux/netfilter.h>

#define WHITELIST_FILE "whitelist.txt"

int check_whitelist(const char* dest_ip) {
    return 1;
}
static int packet_handler(struct nfq_q_handle* qh, struct nfgenmsg* nfmsg,
                         struct nfq_data* nfa, void* data) {
    struct nfqnl_msg_packet_hdr* ph;
    struct ip* ip_header;
    unsigned char* packet_data;
    char src_ip[INET_ADDRSTRLEN], dest_ip[INET_ADDRSTRLEN];

    ph = nfq_get_msg_packet_hdr(nfa);
    if (ph) {
        int packet_len = nfq_get_payload(nfa, &packet_data);
        if (packet_len >= sizeof(struct ip))
        {
            ip_header = (struct ip*)packet_data;
        }
        else
        {
            return nfq_set_verdict(qh, ntohl(ph->packet_id), NF_ACCEPT, 0, NULL);
        }
        inet_ntop(AF_INET, &ip_header->ip_src, src_ip, INET_ADDRSTRLEN);
        inet_ntop(AF_INET, &ip_header->ip_dst, dest_ip, INET_ADDRSTRLEN);

        printf("Packet: src_ip=%s, dest_ip=%s\n", src_ip, dest_ip);

        // Check if the destination IP is in the whitelist
        if (check_whitelist(dest_ip)) {
            //printf("Packet allowed.\n");
            return nfq_set_verdict(qh, ntohl(ph->packet_id), NF_ACCEPT, 0, NULL);
        } else {
            //printf("Packet blocked.\n");
            return nfq_set_verdict(qh, ntohl(ph->packet_id), NF_DROP, 0, NULL);
        }
    }

    return 0;
}

int main() {
    struct nfq_handle* nfq_handle;
    struct nfq_q_handle* nfq_queue;
    int fd, rv;

    // Create a new netfilter queue handle
    nfq_handle = nfq_open();
    if (!nfq_handle) {
        fprintf(stderr, "Error opening nfq library handle\n");
        exit(EXIT_FAILURE);
    }

    // Unbind the existing nf_queue handler (if any) from AF_INET, we don't want it to mess with our packets
    if (nfq_unbind_pf(nfq_handle, AF_INET) < 0) {
        fprintf(stderr, "Error during nfq_unbind_pf()\n");
        exit(EXIT_FAILURE);
    }

    // Bind the nfnetlink_queue as nf_queue handler of AF_INET
    if (nfq_bind_pf(nfq_handle, AF_INET) < 0) {
        fprintf(stderr, "Error during nfq_bind_pf()\n");
        exit(EXIT_FAILURE);
    }

    // Install a callback on queue 0
    //nfq_queue = nfq_create_queue(nfq_handle, NF_INET_LOCAL_OUT, &packet_handler, NULL);
    nfq_queue = nfq_create_queue(nfq_handle, 0, &packet_handler, NULL);
    if (!nfq_queue) {
        fprintf(stderr, "Error creating nfq queue\n");
        exit(EXIT_FAILURE);
    }

    // Set the amount of packet data to copy to userspace
    if (nfq_set_mode(nfq_queue, NFQNL_COPY_PACKET, 0xFFFF) < 0) {
        fprintf(stderr, "Could not set packet copy mode\n");
        exit(EXIT_FAILURE);
    }

    fd = nfq_fd(nfq_handle);

    char buf[4096];
    while ((rv = recv(fd, buf, sizeof(buf), 0)) && rv >= 0) {
        printf("recv %d\n", rv);
        nfq_handle_packet(nfq_handle, buf, rv);
    }
    // Cleanup
    nfq_destroy_queue(nfq_queue);
    nfq_close(nfq_handle);

    return 0;
}

```

```shell
gcc -o ttx filter.c -lnetfilter_queue
```

**要先把ttx运行起来后**，修改iptables，把包发到netfilter队列，如果ttx没有提前运行起来而修改了iptables会让服务器网络不可用只能重启服务器：

```shell
 sudo iptables -A OUTPUT -j NFQUEUE --queue-num 0
```

就会看到ttx有输出了：

```shell
recv 196
Packet: src_ip=172.19.16.7, dest_ip=223.73.211.43
recv 196
Packet: src_ip=172.19.16.7, dest_ip=223.73.211.43
```

要停止，先删除对应的iptables规则：

```shell
iptables -L
iptables -D OUTPUT 1  #数字1取决于上面命令的输出中看到的OUTPUT链里的第几条规则，第一条就是1
```

然后可以安全的停止ttx进程。

