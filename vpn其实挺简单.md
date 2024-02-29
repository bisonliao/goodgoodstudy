# VPN其实挺简单

## 一、实验目的

有两个局域网，每个局域网都有一台可以访问外网的机器，这两台机器相互间通过外网可以通讯。

希望使用vpn软件wireguard，把这两个局域网打通。

![vpn.png](img/namespace/vpn.png)

## 二、步骤

在左边这个绿底的机器上执行：

```shell
#安装必要的软件
apt update
apt install wireguard
apt install openresolv

# 配置为转发
echo "1" > /proc/sys/net/ipv4/ip_forward

#生成公私钥，最后会贴到配置文件里
cd /etc
umask 777
wg genkey >wgpriv.key
wg pubkey <wgpriv.key >wgpub.key

cd /etc/wireguard/
#edit wg0.conf as follow:
# wg0.conf start ============================
[Interface]
#我方的私钥
PrivateKey = OIs5R69frJcqN+AAF8rYKh/qOwPAsPIbW******  
ListenPort = 10086
# VPN的虚拟IP，不是本机的eth0，是wg0这个虚拟网卡的ip
Address = 10.0.0.1
DNS = 114.114.114.114,8.8.8.8
#启动后执行iptables命令，让本机变成一个双向的nat服务器
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE;iptables -A FORWARD -i eth0 -j ACCEPT; iptables -t nat -A POSTROUTING -o wg0 -j MASQUERADE
# 停止前清理iptables规则
PreDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE;iptables -D FORWARD -i eth0 -j ACCEPT; iptables -t nat -D POSTROUTING -o wg0 -j MASQUERADE

[Peer]
#对端的公钥
PublicKey = 4zoLpBxMlcD4frS5II/HWo07lN1k6PIMqpv5eEV1LUM=
#对端的IP地址和端口，用于真正的udp通信
Endpoint = 119.28.214.71:10086
# 允许对端请求过来的网段，可以配置为0.0.0.0/0不限制（小心，会影响默认路由）
AllowedIPs = 10.0.0.2,172.19.16.0/20
PersistentKeepalive = 30
# wg0.conf end ============================

systemctl restart wg-quick@wg0
#用下面的命令可以看到端口起来了。wireguard实际上工作在内核态
netstat -anpl|grep 10086


```

特别说明：AllowedIPs = 10.0.0.2,172.19.16.0/20这项配置，wireguard会据此修改路由表，把目的地为这些IP段的请求都经wg0这个虚拟网卡发送出去。

同样的在右边这个黄底的机器上执行：

```shell
#安装必要的软件
apt update
apt install wireguard
apt install openresolv

# 配置为转发
echo "1" > /proc/sys/net/ipv4/ip_forward

#生成公私钥，最后会贴到配置文件里
cd /etc
umask 777
wg genkey >wgpriv.key
wg pubkey <wgpriv.key >wgpub.key

cd /etc/wireguard/
#edit wg0.conf as follow:
# wg0.conf start ============================
[Interface]
PrivateKey = mFm8HwmZ8F3LWOLxHN7gfL689HszT2+c*******
ListenPort = 10086
Address = 10.0.0.2
DNS = 114.114.114.114,8.8.8.8
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE;iptables -A FORWARD -i eth0 -j ACCEPT; iptables -t nat -A POSTROUTING -o wg0 -j MASQUERADE
PreDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE;iptables -D FORWARD -i eth0 -j ACCEPT; iptables -t nat -D POSTROUTING -o wg0 -j MASQUERADE

[Peer]
PublicKey = wuq5qGADajzJfhfkcovdQ/mpnPafPxy1q8Y2S+m8TmU=
Endpoint = 119.91.159.198:10086
AllowedIPs = 10.0.0.1,10.11.7.0/24
PersistentKeepalive = 30
# wg0.conf end ============================

systemctl restart wg-quick@wg0
#用下面的命令可以看到端口起来了。wireguard实际上工作在内核态
netstat -anpl|grep 10086

```

在绿底的机器上可以ping通黄底的eth0：

```shell
# ping 172.19.16.7
PING 172.19.16.7 (172.19.16.7) 56(84) bytes of data.
64 bytes from 172.19.16.7: icmp_seq=1 ttl=64 time=9.91 ms
64 bytes from 172.19.16.7: icmp_seq=2 ttl=64 time=9.89 ms
```

在黄底机器上可以ping通左边的某个主机：

```shell
# ping 10.11.7.199
PING 10.11.7.199 (10.11.7.199) 56(84) bytes of data.
64 bytes from 10.11.7.199: icmp_seq=1 ttl=63 time=10.1 ms

```

要让两边的主机互相能通，需要修改主机的route，让两台vpn机器作为网关中转目标网段的报文。

例如在右边的每个主机上设置一个路由：

```shell
ip route add 10.11.7.0/24 via 172.19.16.7 dev eth0
```

在云主机的vpc里上面这样做行不通，因为腾讯云的vpc是用Gre实现的虚拟网络，不是用vxlan实现的。每个主机不知道同vpc下其他主机的mac地址，做不到二层转发。只能做三层转发，必须通过云控制台设置路由。



其实还有个vpn软件叫strongswan也不错，但是我折腾了2个小时没有建立起连接，就放弃了。