# kubernetes其实挺简单

写于2019年12月

之前比较早就耳闻k8s，简单了解功能后也挺兴奋：这就是我们团队过去10年想要的技术运营框架！

无奈操作起来挺麻烦，k8s官网安装文档跟一坨屎一样的，经历了从入门到放弃的过程。

最近稍有进展，一位仁兄发扬共产主义精神，他提供了比较方便的安装包和指导文件：

```
https://blog.csdn.net/qq_33236487/article/details/83896051
```

另外龚正等几位大牛写的《Kubernetes权威指南--从Docker到Kubernetes实践全接触》一书也深入浅出。

在他们的帮助下，成功的跑起来一个实验性质的k8s集群，一台master两台node，成功的验证了node、rc、pod、container、静态pod、volume等等资源的创建和使用。

youtube上也有一些不错的教学视频。

## 1、主要的几个概念/资源：

集群由master（管理集群的机器）和node（工作的实际负载机器）组成。

master上启动的进程有kube-apiserver  kube-scheduler  kube-controller-manager

node上启动的进程有kubelet  kube-proxy



k8s集群内自动扩缩容的单位是pod。一个pod包含一个或者多个用户定义的docker container，且还包含一个k8s系统管理用的docker container，叫做pause。

最典型的pod是无状态的，纯计算资源，k8s根据负载的情况动态在node中调度，生成新的pod或者终止异常的pod。由此很容易引入动态扩缩容需要的几个概念：

1. 复制控制（rc）：定义一类pod应该有多少副本存在，如果个数不够，用来创建pod的容器模板是怎样的
2. 部署（deployment）：类似rc，是rc的升级版本。利用deployment可以方便的实现灰度发布和升级。
3. 平行自动扩展（horizontal pod autoscale）HPA：自动的扩缩容控制，可以配置一类pod副本个数的区间、触发副本个数增加/减少的资源条件（例如cpu利用率大于多少就扩容副本个数）

每个pod的副本会获得一个内部的IP，集群内可以通过该IP访问容器中的服务。由于每个pod副本的IP是变化的，不方便访问，所以在pod的基础上封装出service的概念：一组pod的集合，有固定的clusterIP。通过访问service的clusterIP实现在一组pod副本间自动负载均衡。

可以看到：我们不断的提到一组特定的pod、一类pod等等。如何方便的指定一组/一类pod呢？通过标签（label）和selector字段来指定。标签可以认为是一些key：value形式的数据库表字段，而selector就是对应的sql where子句。



上面这些应对无状态的pod够用，但如果是数据库这样的有状态的服务怎么解决呢？我理解有两个大类方案：

1. 静态的pod：在固定的node上启动的pod，不会被k8s动态调度，所以称为静态。pod中的container把物理服务器的本地磁盘通过volume的方式mount起来，数据写入该mount区。pod产生和终止都不会导致数据丢失。
2. 动态的pod加持久volume（PV）：PV是网络文件系统这类远程的持久化方案。pod被随机调度到不同的node，每次创建副本都会mount远程的PV，用来保存和访问数据。



通过clusterIP、pod IP访问服务的方式，只是在k8s集群内部可行。外部是无法通过这种方式来访问的。解决的办法有：

1. NodePort的方式：在node机器的外部可见的IP上暴露一个端口，通过该端口可以映射转发访问到k8s集群内部的服务。
2. ingress方式？
3. 公有云的loadbalance方式？

## 2、试验示例

### 2.1 把新的node加入集群：

```
#在master上创建和查看token信息，这些token只有24小时有效。每次新的node都需要使用有效的token才能加入
kubeadm token create
kubeadm token list
openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'

#在新的node上运行join命令
kubeadm join --token 7ad372.fcbd8e415b78673f 172.19.16.11:6443 --discovery-token-ca-cert-hash sha256:b3a23d1f30e1c209531e63b272c9fe0c2d25bd11704959f68b0d1e9e54201af6

#在master上查看node资源
kubectl get nodes -o wide   #查看简略信息
kubectl describe nodes [nodeid]  #查看详细信息
```

### 2.2 通过rc创建pod资源：

先编写一个yaml定义文件

```
###########################################################
#centos:bison是我自己定义的一个docker image，里面有个网络服务叫echo.py
apiVersion: v1
kind: ReplicationController
metadata:
  name: centos7
spec:
  replicas: 3
  selector:
    app: centos7
  template:
    metadata:
      labels:
        app: centos7
    spec:
      containers:
      - name: centos7
        image: centos:bison
        command: [ "/echo.py" ]
        ports:
        - containerPort: 12345
#############################################
# mysql是官方的一个镜像，mysql服务怎么启动的，入口在
# 哪里我没有搞清楚！
apiVersion: v1
kind: ReplicationController
metadata:
  name: mysql
spec:
  replicas: 1
  selector:
    app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: docker.io/mysql
        ports:
        - containerPort: 3306
        env:
        - name: MYSQL_ROOT_PASSWORD
          value: "123456"
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
```

敲黑板划重点：command字段非常关键，这里是踩坑的密集区域。

1. docker 容器被创建后就执行一个command，该command退到后台、或者执行结束，都会导致该容器/pod副本结束、重新创建。
2. k8s官方示例的mysql/nginx镜像都没有指定command，我看k8s运行了docker-entrypoint...什么鬼。我确实没有搞清楚

然后执行：

```
kubectl create -f mysql_rc.yaml

#查看创建的结果：
kubectl get rc -o wide 
kubectl get pod -o wide
#当然也可以用 kubectl describe命令查看详细信息
kubectl describe pod 
```

删除和修改后重新加载：

```
kubectl delete -f mysql_rc.yaml
kubectl delete rc centos7 
kubectl replace -f centos_7.yaml 
```

### 2.3 三种方式访问服务

查看pod副本的pod IP

```
[root@master ~]# kubectl get pod -o wide    
NAME                      READY     STATUS    RESTARTS   AGE       IP                NODE
centos7-jvkk9             1/1       Running   0          3h        192.168.166.175   node1

centos7-qk2rd             1/1       Running   0          3h        192.168.166.170   node1

centos7-qpqfd             1/1       Running   0          3h        192.168.166.174   node1

static-mysql-static-pod   1/1       Running   1          1h        192.168.254.235   static-pod
```

使用telnet访问pod中的echo.py服务：

```
[root@master ~]# telnet 192.168.166.170 12345
Trying 192.168.166.170...
Connected to 192.168.166.170.
Escape character is '^]'.
111
111
Connection closed by foreign host.
```

创建service，方便通过固定的clusterIP访问。但是我创建的service端口映射有问题，不知道怎么回事。通过expose命令倒是可以：

```
[root@master ~]# kubectl expose rc centos7
service "centos7" exposed

[root@master ~]# kubectl get services                          
NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)     AGE
centos7      ClusterIP   10.99.211.145   <none>        12345/TCP   6s
kubernetes   ClusterIP   10.96.0.1       <none>        443/TCP     2d

[root@master ~]# telnet 10.99.211.145  12345
Trying 10.99.211.145...
Connected to 10.99.211.145.
Escape character is '^]'.
abcd
abcd
Connection closed by foreign host.
```

创建NodePort，使得外部模块可以访问kubernetes集群内的服务：

```
apiVersion: v1
kind: Service
metadata:
   name: svc-centos7
   labels:
     name: svc-centos7
spec:
   type: NodePort
   ports:
   - protocol: TCP
     port: 12345
     targetPort: 12345
     nodePort: 31245
   # selector的条件需要重点注意。前面我的centos7这个pod的label是写
   # 的app: centos7，所以我也写对应的selector，注意不要和name: centos7
   # 混淆了。我一开始就是写错了，导致这个问题3天没有进展
   selector:
     app: centos7  
```

外部模块使用物理机器的IP加暴露的NodePort访问服务：

```
[root@master ~]# telnet 172.19.0.2 31245
Trying 172.19.0.2...
Connected to 172.19.0.2.
Escape character is '^]'.
request string 
request string 
Connection closed by foreign host.   
```

### 2.4 使用静态pod与volume

静态pod不是通过kubectl create的方式创建的，而是在固定的node机器中，在kubelet进程的--pod-manifest-path参数指定的目录下创建一些pod定义.yaml文件实现的。

例如我的一个mysql静态pod：

```
apiVersion: v1
kind: Pod   #大小写敏感，一开始我写成pod，报错！！
metadata:
  name: static-mysql
  labels:
    role: myrole
spec:
  #定义一个volume
  volumes:
    - name: datadir
      hostPath:
        path: "/data/mysqldata"
  containers:
    - name: mysql
      image: docker.io/mysql  #官方的mysql
      imagePullPolicy: Never
      # 我自己定义执行的command：在前台运行mysqld服务
      command: ["/bin/sh"]
      args: ["-c", "chown -R mysql:mysql /var/lib/mysql; mysqld_safe"]
      ports:
        - name: mysql
          containerPort: 3306
          protocol: TCP
      env:
       - name: MYSQL_ROOT_PASSWORD
         value: "123456"
      #挂载volume
      volumeMounts:
      - mountPath: /var/lib/mysql  #挂载点
        name: datadir
```

重新启动该node上的kubelet进程，就可以让上面的静态pod生效：

```
systemctl stop kubelet
systemctl start kubelet
```

停止对应的docker容器，kubelet会重新拉起，拉起后发现mysql库中的数据没有丢失。如果不指定volume，数据会丢失的。

### 2.5 定位问题常用的几个命令

主要就是看kubelet的日志和状态：

```shell
systemctl status kubelet
journalctl  -u kubelet  >/tmp/bison
journalctl  -f -u kubelet
```

还有就是exec到docker的容器里去看看启动的服务情况，例如我可以进到mysql这个容器里去增删改查数据：

```shell
docker exec -it 50b71b1b4408 /bin/bash
```

另外阅读官方文档和网友的博客也是很重要的：

```
https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/
https://blog.csdn.net/lanwp5302/article/details/87348132
https://www.jianshu.com/p/916f9b111b23
```

## 3、REST API和各种编程语言接口

kubernetes的API Server提供了RESTful 接口协议和java等语言的API库，可以编程的方式来管理kubernetes集群。

详细文档见：

```
https://kubernetes.io/docs/reference/
https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.13/#-strong-api-overview-strong-
```

直接使用curl命令访问API Server的端口会返回没有权限的错误信息。

解决方案是使用kubectl启动一个代理进程，该进程会访问本地.kube/config等配置文件获取用户信息，并接受curl等外部工具的请求：

```
kubectl proxy --accept-hosts='^*$' --address="0.0.0.0"&
```

上述命令可以通过accept-host  accept-paths等等参数实现复杂的白名单控制。

curl等外部工具通过访问上述代理进程的8001端口，协议格式还是k8s的RESTful 协议格式，例如获取service的信息：

```
[root@static-pod ~]# curl -X GET http://master:8001/api/v1/namespaces/default/services/centos7
#下面是k8s集群的返回：
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "centos7",
    "namespace": "default",
    "selfLink": "/api/v1/namespaces/default/services/centos7",
    "uid": "11770cef-14df-11ea-aadc-525400c3782d",
    "resourceVersion": "248829",
    "creationTimestamp": "2019-12-02T08:38:09Z",
    "labels": {
      "app": "centos7"
    }
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": 12345,
        "targetPort": 12345
      }
    ],
    "selector": {
      "app": "centos7"
    },
    "clusterIP": "10.99.211.145",
    "type": "ClusterIP",
    "sessionAffinity": "None"
  },
  "status": {
    "loadBalancer": {}
  }
}
```

有例如，通过POST方式创建一个pod：
```
curl -X POST -H 'content-type: application/yaml' http://master:8001/api/v1/namespaces/default/pods -d '
apiVersion: v1 
kind: Pod
metadata:
  name: pod-example
spec:
  containers:
  - name: ubuntu
    image: ubuntu:trusty
    command: ["echo"]
    args: ["Hello World"]
'
```

成功返回该pod的信息：

```
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "pod-example",
    "namespace": "default",
    "selfLink": "/api/v1/namespaces/default/pods/pod-example",
    "uid": "376ecddc-1594-11ea-aadc-525400c3782d",
    "resourceVersion": "358770",
    "creationTimestamp": "2019-12-03T06:14:51Z"
  },
  "spec": {
    "volumes": [
   #只展示上面部分，截断...
```

python API 列出所有pod的代码示例：
```python
from kubernetes import client, config

#Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()
v1 = client.CoreV1Api()
print("Listing pods with their IPs:")
ret = v1.list_pod_for_all_namespaces(watch=False)
for i in ret.items:
    print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
```

## 4、安装过程

不甘心使用前辈的现成的安装包，想自己一步一步安装一下k8s集群。

不得不说k8s官网的安装文档跟屎一样，各种踩坑折腾，最后成功安装起来了，记录一下。

前面4.1-4.3， master和node都需要执行。

### 4.1 修改基础环境：

准备两台机器，一台叫master，一台叫node1，分别修改他们的hostname，并修改/etc/hosts文件：

```shell
hostnamectl set-hostname master
hostnamectl set-hostname node1
/etc/hosts
172.19.0.11 master
172.19.0.15 node1
```

修改网络参数：

```shell
cat <<EOF >  /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
sysctl --system
echo 1 > /proc/sys/net/ipv4/ip_forward
```

### 4.2 安装docker

```shell
yum install -y docker
systemctl enable docker
systemctl start docker 
docker ps
```

没有问题的话，docker ps命令会输出一些字段标题

### 4.3 安装kubeadm/kubectl/kubelet

centos yum库里的kubernetes版本太低，需要修改yum库的配置。

kubernetes官网给出的yum库配置不可用！

最后找到一个可用的阿里云的yum库配置。

```shell
cat <<EOF >/etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=http://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=0
repo_gpgcheck=0
gpgkey=http://mirrors.aliyun.com/kubernetes/yum/doc/yum-key.gpg
       http://mirrors.aliyun.com/kubernetes/yum/doc/rpm-package-key.gpg
EOF

setenforce 0
sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config

yum install -y kubelet kubeadm kubectl --disableexcludes=kubernetes

systemctl enable --now kubelet
```

### 4.4 继续安装master

```shell
 kubeadm init 
```

上面命令没有问题的话，会输出很多信息，包括node加入集群的命令。

然后准备kubectl的配置文件：

```shell
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

安装网络插件：

```shell
kubectl apply -f https://docs.projectcalico.org/v3.8/manifests/calico.yaml
```

至此，kubectl命令可以使用了，可以用kubectl get nodes试一试。

用docker ps也可以看到启动了很多容器，有kube-apiserver kube-cotrollermanager kube-scheduler等等。

### 4.5 继续安装node

在node上执行：

```shell
kubeadm join 172.19.0.11:6443 --token r5ct19.rted6y03glhqoiyz  --discovery-token-ca-cert-hash sha256:5b02d4c6662bd3a5f9a85767c116ed6dd5984023d826e7bdd022fc6d69090bf3 
```

这条命令来自前面kubeadm init的输出。

没有问题的话，可以看到kubelet、kube-proxy进程在node节点启动了。

过一会（可能10分钟后），在master上执行kubectl get node就可以看到新安装的node是ready状态了。

### 4.6 测试启动一个pod

还是前面的mysql_rc.yaml：

```
apiVersion: v1
kind: ReplicationController
metadata:
  name: mysql
spec:
  replicas: 1
  selector:
    app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: docker.io/mysql
        ports:
        - containerPort: 3306
        env:
        - name: MYSQL_ROOT_PASSWORD
          value: "123456"
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
```

```
kubectl create -f mysql_rc.yaml
```

```shell
[root@VM_0_11_centos ~]# kubectl get pods -o wide
NAME          READY   STATUS    RESTARTS   AGE   IP                NODE    NOMINATED NODE   READINESS GATES
mysql-cnqpl   1/1     Running   0          28m   192.168.166.129   node1   <none>           <none>
```



### 4.7 参考文档

参考了屎一样的官方文档：

``` shell
https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/
https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/

https://support.huaweicloud.com/basics-cce/kubernetes.html #这个还不错
#Kubernates In Action这本书也不错
```

