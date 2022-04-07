### 1、 socks5协议

1. 支持tcp（主动连接出去和监听接受连接）、udp协议，即connect、bind、udp三种模式
2. 支持按域名指定对端地址，而不只是IP。
3. 工作在传输层，所以基于此，可以透明的支持http和https

权威出处：https://www.rfc-editor.org/rfc/rfc1928.txt

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



