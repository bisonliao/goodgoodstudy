

### 首先：几个容易混淆的概念

有三个容易混淆的概念

1. kubeproxy
2. 网络插件，例如flanneld
3. istio sidecar，例如envoy

#### 先说kubeproxy

kubeproxy运行在每个节点上，负责实现kubernetes的Service概念，也就是将集群内外的请求转发到正确的后端Pod。

kubeproxy支持多种代理模式，主要有iptables和ipvs两种模式，kubeproxy主要工作在网络的四层（传输层）。

举个例子：Node甲上的Pod A访问service echo的Pod B， Pod B运行在Node 乙上，这个过程中，kubeproxy在以下环节发生作用：

1. 首先，Pod A需要通过service echo的域名或者Cluster IP来访问Pod B，这个域名或者Cluster IP是由K8S的内部DNS服务解析的。
2. 其次，Node甲上的kubeproxy会根据service echo的Cluster IP和端口，查找对应的iptables或者ipvs规则，将请求重定向到一个随机的Pod B的IP和端口23。
   最后，Node 乙上的kubeproxy会根据Pod B的IP和端口，将请求转发给Pod B，完成服务的访问
3. 简单来说，kubeproxy的作用是实现service的负载均衡和流量转发，它在每个节点上都有运行，维护着网络规则和代理服务

#### 再说网络插件，例如flanneld

网络代理负责为pod提供IP地址和互通性，类似一个tap/tun设备。详细可以搜索tap/tun设备的技术文章。它用ipip技术，把发往pod的原始IP包，封装成宿主网络的IP包，发到Node乙后，脱去外层的IP，得到原始IP包，发送给Pod B。

#### sidecar，例如envoy

sidecar是运行在pod里面，提供pod间的流量管理、安全、观察性等能力，主要工作在网络的七层。可以对http、gRPC、WebSocket等协议进行更细粒度的控制。和nginx很类似。



### 一、安装和注入sidecar

```shell
istioctl kube-inject -f deployment.yaml -o deployment-injected.yaml

kubectl label namespace default istio-injection=enabled
```



### 二、流量控制

#### 简介：

```
As the data-plane service proxy, Envoy intercepts all incoming and outgoing requests at runtime (as traffic 
flows through the service mesh). This interception is done trans‐ parently via iptables rules or a Berkeley 
Packet Filter (BPF) program that routes all network traffic, in and out through Envoy. Envoy inspects the 
request and uses the request’s hostname, SNI, or service virtual IP address to determine the request’s 
target (the service to which the client is intending to send a request). Envoy applies that tar‐ 
get’s routing rules to determine the request’s destination (the service to which the ser‐ vice proxy is 
actually going to send the request). Having determined the destination, Envoy applies the destination’s 
rules. 
```



#### 1、ServiceEntry

ServiceEntrys are how you manually add/remove service listings in Istio’s service registry. Entries in the service registry can receive traffic by name and be targeted by other Istio configurations. All service registries with which Istio integrates (Kubernetes, Consul, Eureka, etc.) work by transforming their data into a ServiceEntry. 

1.  you can use them to give a name to an IP address
2. you can forward requests to foo.bar.com to baz.com to use DNS to resolve to endpoints
3.  you can   create virtual IP addresses (VIPs), mapping an IP address to a name

 Istio does not populate DNS entries based on ServiceEntrys. This means that Example 8-1, which gives the address 2.2.2.2 the name some.domain.com, will not allow an application to resolve some.domain.com to 2.2.2.2 via DNS. There is a core DNS plug-in for Istio that generates DNS records from Istio ServiceEntrys, which  you can use to populate DNS for Istio services in environments outside of Kubernetes

一个典型的用途就是，网格内的流量访问集群外部的服务时，如果希望用白名单的方式，那就把目标服务封装成ServiceEntry，通过ServiceEntry来访问这个外部服务。创建指向该外部服务的ServiceEntry，就相当于向istio注册了一个可访问的白名单。

注意：istio默认是放行访问集群外部的请求，如果需要改为白名单这样更安全的方式，需要用istioctl命令把outboundTrafficPolicy 改为 REGISTRY_ONLY。

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: ServiceEntry
metadata:
  name: svc-entry
spec:
  hosts:
    - www.baidu.com
 ports:
    - number: 8080
    name: http
    protocol: HTTP
 resolution: STATIC
 endpoints:
    - address: 10.108.159.183 #可以发现，确实把访问www.baidu.com的请求劫持到访问这个clusterIP上的K8S服务了
```

```yaml
 # 把访问k8s service的流量定向到静态IP
apiVersion: networking.istio.io/v1alpha3
kind: ServiceEntry
metadata:
 name: my-service-in-istio
spec:
  hosts:
  # The names an application can use in k8s to target this service
  - my-service
  - my-service.default
  - my-service.default.svc.cluster.local
  ports:
  - number: 80
  name: http
  protocol: HTTP
  resolution: STATIC
  endpoints:
  - address: 1.2.3.4
```

具体的字段可以参考：

```
https://istio.io/v1.1/docs/reference/config/networking/v1alpha3/service-entry/
```



### 2、DestinationRules：

DestinationRules allow a service operator to describe how a client in the mesh should call their service, including the following：

1. Subsets of the service (e.g., v1 and v2) 
2. The load-balancing strategy the client should use 
3. The conditions to use to mark endpoints of the service as unhealthy
4.  L4 and L7 connection pool settings
5. TLS settings for the server

简单的说包括：

1. 连接池并发的设置
2. TLS的设置
3. subset
4. 比K8S更丰富的负载均衡
5. 熔断

```yaml
#  A DestinationRule configuring low-level connection pool settings and loadbalance
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: foo-default
spec:
  host: foo.default.svc.cluster.local
  trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 4
        http:
          http2MaxRequests: 1000
      loadBalancer:
        consistentHash:
          useSourceIp: True
      outlierDetection: #熔断剔除
        consecutiveErrors: 7
        interval: 5m
        baseEjectionTime: 15m

```

具体的字段和属性可以参考：

```
https://istio.io/v1.1/docs/reference/config/networking/v1alpha3/destination-rule/
```

subset可能是DR用的比较多的一种，例子见下面的VS的部分。

在 Istio 的 `DestinationRule` 中，`spec.host` 字段指定了该规则适用的目标服务。这个字段的含义通常更直接和具体，相比于 `VirtualService` 的 `spec.hosts` 字段。以下是 `DestinationRule` 的 `spec.host` 可能包含的情形：

1. **Kubernetes Services**：最常见的用法是指定 Kubernetes 集群内的服务（Service）。这意味着 `DestinationRule` 将应用于所有发送到这个特定 Kubernetes 服务的流量。它通常用于定义特定服务的负载均衡策略、连接池大小、TLS 设置等。
2. **服务入口**：在一些场景下，`spec.host` 可以被设置为服务网格中定义的服务入口（ServiceEntry）。这主要用于外部服务的流量管理。当您在服务网格中定义了外部服务的访问策略时，`DestinationRule` 可以用来为这些外部服务指定特定的流量策略。
3. **域名**：对于外部服务，`spec.host` 也可以是一个具体的域名。这在定义外部服务的流量策略时特别有用，比如对外部 API 或第三方服务的调用。

这个字段**不**用于匹配 HTTP 请求头部中的 `Host` 字段。这一点与 `VirtualService` 的行为有所不同。`DestinationRule` 主要用于配置针对特定服务的策略，如负载均衡、连接池、TLS 设置等，而不是基于 HTTP 请求的属性（如头部信息）来路由流量。

#### 3、Virtual Service

A virtual service lets you configure how requests are routed to a service within an Istio service mesh, building on the basic connectivity and discovery provided by Istio and your platform. Both for http and tcp.

You can use VirtualServices to target very specific segments of traffic and direct them to different destinations. For example, a VirtualService can match requests by header values, the port a caller is attempting to connect to, or the labels on the client’s workload (e.g., labels on the client’s pod in Kubernetes) and send matching traffic to a different destination (e.g., a new version of a service) . 

除了路由管理，VS还提供：

1. 超时控制
2. 故障注入：按一定比例的请求返回指定错误码或者按一定比例的请求延迟指定的时间
3. 重定向
4. 重试
5. 重写uri或者头部字段
6. 流量镜像
7. 跨源访问策略控制（CORS）
8. 头部字段设置

```yaml
# 创建一个没有对应的EndPoint的K8S service, 使得访问的时候能够识别出来
apiVersion: v1
kind: Service
metadata:
  name: foo
spec:
     ports:
       - port: 8080
         targetPort: 8080

# 创建一个vs，把上述service的流量转到有实际Endpoint的service
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: foo-identity
spec:
 hosts:
 - foo
 http:
 - route:
   - destination:
       host: calcclient

# 发出下面的请求，会转到访问calcclient:8000/calc
curl http://foo:8080/calc 
```

灰度的流量百分比控制：

```yaml
# 两个dp，对应不同的版本
apiVersion: apps/v1
kind: Deployment
metadata:
 name: calcserver1
 labels:
     app: calcserver1
spec:
 replicas: 1
 selector:
   matchLabels:
     app: calcserver1
 template:
    metadata:
      labels:
        app: calcserver1
        service: calcserver
    spec:
      containers:
      - image: bisonliao/calc_server
        name: calcserver
        ports:
        - containerPort: 50012
---
apiVersion: apps/v1
kind: Deployment
metadata:
 name: calcserver2
 labels:
     app: calcserver2
spec:
 replicas: 1
 selector:
   matchLabels:
     app: calcserver2
 template:
    metadata:
      labels:
        app: calcserver2
        service: calcserver
    spec:
      containers:
      - image: bisonliao/calc_server
        name: calcserver
        ports:
        - containerPort: 50012
# 两个DP的pod都属于一个service
---
apiVersion: v1
kind: Service
metadata:
  name: calcserver
spec:
     ports:
       - port: 50012
         targetPort: 50012
     selector:
         service: calcserver
# 用DR把该service劈成两个set
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
 name: dr-calcserver
spec:
 host: calcserver
 subsets:
 - name: v1
   labels:
      app: calcserver1
 - name: v2
   labels:
      app: calcserver2
#用vs对不同的set分发不同比例的流量
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: vs-calcserver
spec:
 hosts:
 - calcserver
 http:
 - route:
   - destination:
       host: calcserver #内部有效的K8S服务
       subset: v1
     weight: 90
   - destination:
       host: calcserver #内部有效的K8S服务
       subset: v2
     weight: 10
```

在 Istio 的 `VirtualService` 配置中，`spec.hosts` 字段指定了该路由规则将应用于哪些服务的访问。这个字段的含义取决于 `VirtualService` 的上下文和位置。`spec.hosts` 字段可以包含以下情形：

1. **Kubernetes Services**：在大多数情况下，`spec.hosts` 中的条目指的是 Kubernetes 中的服务（Service）。这意味着，当您在 `VirtualService` 中指定一个服务名时，它会影响到所有发送到这个 Kubernetes 服务的流量。
2. **目的域名**：`spec.hosts` 也可以包含外部服务的域名。这在您想要控制流量到外部服务（不在 Kubernetes 服务网格中）时非常有用。通过在 `VirtualService` 中定义外部服务的域名，您可以对这些外部服务的流量应用路由规则。
3. **请求中的 Host 头部**：在处理 HTTP 请求时，`spec.hosts` 字段也可以与请求头部中的 `Host` 或 `:authority` 字段匹配。这意味着，当客户端发起一个 HTTP 请求，并且请求头部中的 `Host` 字段与 `spec.hosts` 中定义的条目匹配时，相应的路由规则就会被应用。

在 Istio 的 `VirtualService` 配置中，`spec.hosts` 字段不能是另一个 `VirtualService` 的名称。`VirtualService` 之间不会直接通过名称引用对方。

把istio ingress上进入的外部流量引导到内部某个服务：

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: gw-client
spec:
# selector:
#    istio: ingressgateway # use Istio default gateway implementation
  servers:
    - hosts:
      - calcclient.com
      port:
        number: 80
        name: http
        protocol: HTTP

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vs-calcclient
spec:
  hosts:
    - calcclient.com
  gateways:
    - gw-client
  http:
    - route:
      - destination:
           host: calcclient #内部有效的K8S服务

```

流量镜像：

从镜像目的地回来的应答不会影响到业务，istio sidecar会丢弃它。

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: catalog
spec:
  hosts:
  - catalog
  gateways:
  - mesh
  http:
  - route:
    - destination:
       host: catalog
       subset: version-v1
       weight: 100
    mirror:
      host: catalog
      subset: version-v2
```

注入故障：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: foo
spec:
     ports:
       - port: 8080
         targetPort: 8080

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: foo-identity
spec:
 hosts:
 - foo
 http:
 - route:
   - destination:
       host: calcclient
   fault:
     abort:
        percentage:
          value: 30
        httpStatus: 400
     delay:
        percentage:
          value: 30
        fixedDelay: 1s
```

#### 4、gateway

You use a [gateway](https://istio.io/latest/docs/reference/config/networking/gateway/#Gateway) to manage inbound and outbound traffic for your mesh, letting you specify which traffic you want to enter or leave the mesh. Gateway configurations are applied to standalone Envoy proxies that are running at the edge of the mesh, rather than sidecar Envoy proxies running alongside your service workloads.

我的疑问是：下面的例子我创建成功后还是有问题，好像缺少VIP：

1. 在k8s集群外面访问calcclient.com还是提示dns解析失败
2. 如果我添加DNS A记录，IP是多少呢？有istio-ingressgateway-cd445fcdb-2lf4b这个pod存在

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
 name: gw-client
spec:
  selector:
    istio: ingressgateway # use Istio default gateway implementation
  servers:
   - hosts:
     - calcclient.com
     port:
       number: 8080  
       name: http
       protocol: HTTP

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vs-calcclient
spec:
  hosts:
    - calcclient.com
  gateways:
    - gw-client
  http:
    - route:
      - destination:
           host: calcclient
```

用这个命令查看gw的配置

```shell
istioctl proxy-config route deploy/istio-ingressgateway -o json --name http.8080 -n istio-system
kubectl -n istio-system get svc istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
kubectl -n istio-system get svc istio-ingressgateway -o jsonpath='{.status}'
```

向chatGPT 4.0学习了解到：

在 Istio 中，`Gateway` 资源和 `Ingress Gateway` 控制器之间的关系是非常重要的概念。chatGPT解释如下：

1. **默认 Ingress Gateway**：在 Istio 中，默认的 `Ingress Gateway` 是一个预配置的、部署在 Kubernetes 集群中的特殊负载均衡器，它作为服务网格的入口点，处理从外部网络到网格内服务的流量。它是通过一个 Kubernetes Deployment 和相应的 Service 来实现的，并且通常在安装 Istio 时一起部署。通常使用envoy。
2. **Gateway 资源与 Ingress Gateway 的关系**：
   - `Gateway` 资源是一个配置实体，它定义了如何处理进入网格的流量。它本身不处理流量，而是告诉 Ingress Gateway 如何处理流量。
   - 在 `Gateway` 资源中的 `spec.selector` 字段指定了哪个 Ingress Gateway 控制器应该应用这个 `Gateway` 的配置。通过这个选择器，Istio 知道将这个 `Gateway` 配置应用到哪个实际运行的 Gateway 实例上。

您可以在 Istio 中配置和运行多个 Ingress Gateways。这对于处理不同的流量类型或满足不同的安全需求非常有用。方法就是创建一个新的 Deployment 和 Service。例如下面的配置文件创建了自定义的ingress gateway:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-custom-ingressgateway
spec:
  selector:
    matchLabels:
      istio: custom-ingressgateway
  template:
    metadata:
      labels:
        istio: custom-ingressgateway
    spec:
      containers:
      - name: istio-proxy
        image: gcr.io/istio-release/proxyv2:1.8.0 # 使用与您的Istio版本相匹配的镜像
        ports:
          - containerPort: 80
          - containerPort: 443
---
apiVersion: v1
kind: Service
metadata:
  name: my-custom-ingressgateway
spec:
  type: LoadBalancer
  selector:
    istio: custom-ingressgateway
  ports:
  - port: 80
  - port: 443
```

另外，说到gateway，就不得不说公有云上的LoadBalance产品。对同一个虚拟IP的请求负载，LB怎么做到无上限的横向扩展容量的呢？ 毕竟如果虚拟IP绑定到一台物理服务器上，其网卡和cpu的能力一定是有上限的，而一个IP又不能同时绑定到多台物理服务器。

通过使用虚拟IP、负载均衡技术、网络地址转换等方法，多个物理或虚拟服务器可以共同响应发送到单个虚拟IP的请求，而无需直接将该虚拟IP绑定到每台服务器上。这种方法允许LB灵活地管理流量，并将请求分发到后端服务器池中的任何服务器，从而实现高效的流量管理和高可用性。

与K8S中的cluster IP/Pods的关系很类似，LB的虚拟IP并没有绑定到物理服务器，只是一个地址标识。机房里的路由器被下发了一组规则：如果目的IP是指定的虚拟IP，就把请求包按一定的路由规则（RR、random、sticky等等）下发给一系列的物理服务器。

想象一下机房里的路由器都接受SDN控制面的指令，在iptables里写入了路由规则：目的地为x.x.x.x这个虚拟IP的请求包，就RR的方式发给一组服务器中的某一台。转发的时候，使用的是实际的服务器的物理IP地址作为目的地址。

总之就是此路是我开，只要经过了路由器，那就有上帝视角来灵活控制流量，而虚拟IP就是流量的tag。

### 四、用官方例子bookinfo做实验

#### 1、在腾讯云部署k8s标准集群和托管的网格集群

比较简单，不展开

#### 2、使用helm安装bookinfo

```shell
# install helm
wget https://get.helm.sh/helm-v3.11.1-linux-amd64.tar.gz 
tar zxf helm-v3.11.1-linux-amd64.tar.gz 
mv linux-amd64/helm /usr/local/bin

# install bookinfo
helm repo add evry-ace https://evry-ace.github.io/helm-charts/
helm search repo evry-ace |grep bookinfo
helm install  istio-bookinfo evry-ace/istio-bookinfo
```

#### 3、配置hosts中的域名指向LB

##### 3.1 linux下用curl请求

bookinfo里用到的入口的host本来是要配置为example.com，但如果这个域名没有备案，就算是修改hosts的方式来测试，浏览器也会提示域名没有备案。应该是腾讯云搞的鬼。但用curl是走得通的：

```shell
vi /etc/hosts #加上这么一行，具体的外网地址，要查看网格集群里的gateway的公网IP
159.75.192.183 example.com

curl -v http://example.com/productpage #会返回一个带有login字样的html页面
```

但上面的步骤在windows下用浏览器访问就行不通，会提示域名没有备案，即使修改user-agent伪装为curl也不行。

##### 3.2 windows下用浏览器请求

我也是被逼的没有办法了。就像修改bookinfo项目里virtual service 和gateway的hosts为www.baidu.com算了，因为这个域名是明显备案了，而且我也不会用到百度。

幸亏bookinfo项目里要修改的vs和gw都只有一个，都叫bookinfo，直接编辑yaml修改里面的example.com字样为www.baidu.com。

然后修改C:\Windows\System32\drivers\etc 目录里的hosts文件，添加：

```shell
159.75.192.183 www.baidu.com #具体的外网地址，要查看网格集群里的gateway的公网IP
```

修改需要一点小技巧，先把该文件在当前目录拷贝，就会重命名一个文件，把这个重命名后的文件拿到e盘修改好。拷贝回来，删除掉hosts文件，再修改这个重命名的文件为hosts就生效了。

注意执行一下DNS刷新并检查是否有效：

```shell
ipconfig /flushdns
ping www.baidu.com #看ip生效没有
```

然后用浏览器请求就可以生效了，如果还不行，大概率是因为梯子或者代理导致的:

```
http://www.baidu.com/productpage
```

#### 4、练习一：对reviews的三个版本做流量分配

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
 name: dr-reviews
spec:
 host: reviews
 subsets:
 - name: v1
   labels:
      version: v1
 - name: v2
   labels:
      version: v2
 - name: v3
   labels:
      version: v3
#用vs对不同的set分发不同比例的流量
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: vs-reviews
spec:
 hosts:
 - reviews
 http:
 - route:
   - destination:
       host: reviews   #内部有效的K8S服务
       subset: v1
     weight: 1
   - destination:
       host: reviews #内部有效的K8S服务
       subset: v2
     weight: 1
   - destination:
       host: reviews #内部有效的K8S服务
       subset: v3
     weight: 98
```

```shell
kubectl create -f vs.yaml
```

不断的刷新浏览器，会发现与页面的行为改变了：之前是reviews的三个版本等比例出现，有时候是红猩猩有时候是黑猩猩有时候不出星星，改动后现在是大概率是没有星星，偶尔出现红猩猩或者黑猩猩。

#### 5、练习二、基于bookinfo创建自己的charts

##### 5.1 下载并修改bookinfo，可以指定replicas和cpu的requests

```yaml
helm pull evry-ace/istio-bookinfo
tar zxf istio-bookinfo-1.2.2.tgz
mv istio-bookinfo bookinfo

#在values.yaml里追加：
productpage:
  replicas: 3
  requests:
    cpu: 300m
    memory: 512Mi

details:
  replicas: 3
  requests:
    cpu: 300m
    memory: 512Mi

ratings:
  replicas: 3
  requests:
    cpu: 300m
    memory: 512Mi

reviews:
  replicas: 3
  requests:
    cpu: 300m
    memory: 512Mi
    
# 修改template里的 ratings.yaml等文件，在deployment的定义里引用上述的values：
spec:
  replicas: {{ .Values.ratings.replicas }}
  selector:
    matchLabels:
      app: ratings
      version: v1
  template:
    metadata:
      labels:
        app: ratings
        version: v1
    spec:
      serviceAccountName: bookinfo-ratings
      containers:
      - name: ratings
        image: docker.io/istio/examples-bookinfo-ratings-v1:1.15.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 9080
        resources:
          requests:
            cpu: {{ .Values.ratings.requests.cpu }}
            memory: {{ .Values.ratings.requests.memory }}
#调试，看看values替换是否符合预期            
 helm template --debug bookinfo
 helm lint
```

##### 5.2 打包，并提交到一个http服务器

如何创建一个自己的helm repository，可以参考文档：

```
https://helm.sh/docs/topics/chart_repository/#github-pages-example
```

如何用github快速部署一个http服务器，并拥有自己的主页，可以参考：

```
https://docs.github.com/en/pages/quickstart
```

很重要的一句话：

```
A chart repository is an HTTP server that houses an index.yaml file and optionally some packaged charts. 
When you're ready to share your charts, the preferred way to do so is by uploading them to a chart repository.
```

重要的步骤摘录如下：

```shell
helm package bookinfo
cd bookinfo;  helm repo index --url https://bisonliao.github.io/bookinfo/ ./
```

##### 5.3 使用自己的helm repo

把获得的tgz文件和index.yaml文件push到github项目里，就可以作为一个helm repository来用了：

```shell
helm repo add bisonliao https://bisonliao.github.io/bookinfo
helm repo list
helm search repo bookinfo
helm install  bookinfo bisonliao/istio-bookinfo
helm uninstall bookinfo
```

##### 5.4 广而告之

如果你希望把自己做的charts分享给其他人，可以通过artifacthub等网站把刚才的http服务和charts公示出去。但这不是charts正常运行的必须的步骤。

```
https://artifacthub.io/
```

#### 6、练习三：负载均衡策略、熔断、限流等控制

##### 6.1 负载均衡策略

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: dr-reviews
spec:
  host: reviews.default.svc.cluster.local
  trafficPolicy:
      loadBalancer:
        consistentHash:
          useSourceIp: True
```

再用浏览器刷新的时候，会发现productpage只会访问reviews1 2 3中间的一个，reviews图标不会变化了，因为这时候的路由是使用按调用方IP的一致性hash的。

##### 6.2 故障注入 

下面这个如果是改为productpage，就不会生效，还没有搞清楚。其他微服务是有效的。

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: vs-foo
spec:
 hosts:
 - reviews.default.svc.cluster.local
 http:
 - route:
   - destination:
       host: reviews.default.svc.cluster.local
   fault:
     abort:
        percentage:
          value: 50
        httpStatus: 503
     delay:
        percentage:
          value: 50
        fixedDelay: 2s
```



##### 6.3 超时控制

reviews时延比较大，访问路径一路设置超时时间。CLB的超时时间设置为18s。预期的应该是10s后能成功拉到包含reviews信息的页面。

但不符合预期，6s超时，reviews没有拉到，我怀疑是ingress的默认超时时间是6s，所以用curl尝试直接访问productpage的pod来绕过ingress，也还是拉不到reviews。

```shell
 time curl -v -H 'Host:www.baidu.com' http://172.31.0.73:9080/productpage|grep Sorry
```

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vs-foo
spec:
  hosts:
  - www.baidu.com
  - productpage
  http:
  - route:
    - destination:
        host: productpage
    timeout: 15s

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vs-foo1
spec:
  hosts:
  - reviews
  http:
  - route:
    - destination:
        host: reviews
    timeout: 13s

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: vs-foo2
spec:
 hosts:
 - reviews.default.svc.cluster.local
 http:
 - route:
   - destination:
       host: reviews.default.svc.cluster.local
   fault:
     delay:
        percentage:
          value: 100
        fixedDelay: 10s

```



##### 6.4 熔断

修改productpage为2个pod，查找到其中一个pod ip，在node上通过iptables使得不能访问：

```
sudo iptables -A OUTPUT -p all -d 172.31.0.94 -j DROP
sudo iptables -A INPUT -p all -d 172.31.0.94 -j DROP
```

同样的，也把reviews-v2也通过iptables设置为不能访问。

过一会，这两个pod状态显示为不健康。不断的刷新浏览器，可以看到黑色星星的评价不在出现了。但页面本身不会返回5xx。

##### 6.5 重试

下面这样实验从浏览器的感觉来看似乎是有效的，但在productpage所在的node上抓包没有抓到重复发出的包，下面172.31.0.8是productpage的pod IP：

```shell
sudo tcpdump -i cbr0 -A 'host 172.31.0.8 and port 9080 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420'
```

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vs-foo1
spec:
  hosts:
  - reviews
  http:
  - route:
    - destination:
        host: reviews
    retries:
      attempts: 3
      perTryTimeout: 2s
    timeout: 6s
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
 name: vs-foo2
spec:
 hosts:
 - reviews.default.svc.cluster.local
 http:
 - route:
   - destination:
       host: reviews.default.svc.cluster.local
   fault:
     delay:
        percentage:
          value: 50
        fixedDelay: 10s
```

参考资料：

```
https://istio.io/latest/zh/docs/concepts/traffic-management/#network-resilience-and-testing
```

##### 6.6 限流

听xx说DestineRule的限流功能太弱了，需要用到envoyfilter。还没有来得及学习。

##### 6.7 弹性与k8s的对比

istio主要的三个功能：

1. **弹性（resiliency）**，其实不是我们理解的容量扩缩，自动剔除故障endpointer、对暂时性的网络波动进行重试和超时、剔除异常节点等，对应我们常说的服务治理。我觉得这里翻译为韧性比较好，容量扩缩的弹性，用Elastics这个词更好。
2. **流量控制**，例如路由5%的流量到新版本
3. **可观察性**

对于第一点，弹性。即使在没有 Istio 的情况下，Kubernetes 本身就有机制来检测并剔除有故障的 Pod，这主要通过存活探针和就绪探针来实现。这些探针帮助 Kubernetes 维护服务的健康状态，确保流量只被发送到健康的 Pod。

在 Kubernetes 环境中，即使没有使用 Istio 这样的服务网格，Kubernetes 本身也提供了一种机制来自动检测并剔除有故障的 Pod。这主要是通过 Kubernetes 的健康检查机制，包括存活探针（Liveness Probes）和就绪探针（Readiness Probes）来实现的。

**Kubernetes 健康检查机制**

1. **存活探针（Liveness Probes）**：这些探针用于确定 Pod 是否仍然活着。如果存活探针检查失败（比如超过一定次数），Kubernetes 将会重启该 Pod。
2. **就绪探针（Readiness Probes）**：这些探针用于确定 Pod 是否准备好接受流量。如果一个 Pod 的就绪探针失败，Kubernetes 会从与该 Pod 关联的所有服务的负载均衡器中移除该 Pod。只有当就绪探针成功时，Pod 才会被重新加入负载均衡池。

**原理**

- **探针类型**：存活和就绪探针可以是 HTTP GET 请求、TCP 套接字连接尝试，或者是执行 Pod 内的命令。
- **探针结果**：
  - 如果存活探针失败，意味着程序处于非活动状态，Kubernetes 会重启该 Pod。
  - 如果就绪探针失败，意味着 Pod 无法正常处理流量，Kubernetes 则会暂时从服务端点中移除该 Pod，直到就绪探针再次成功。
- **负载均衡和服务发现**：在 Kubernetes 中，服务（Service）资源用于负载均衡和服务发现。当服务的后端 Pod 发生变化（如 Pod 崩溃、重启、变得不可用或者新 Pod 变得可用）时，Kubernetes 会自动更新服务的端点（Endpoints），确保流量只会被路由到健康的 Pod。

**Kubernetes 与 Istio 的区别**

- **Kubernetes 本身的能力**：即使没有 Istio，Kubernetes 本身就可以处理 Pod 的健康检查和流量的基本路由。
- **Istio 的额外功能**：使用 Istio 可以在 Kubernetes 的基础上提供更复杂的流量管理和故障处理策略，如高级路由规则、流量复制、故障注入等。

k8s的探针的举例：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp-container
        image: myapp:1.0
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 3
          periodSeconds: 3
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 3
```

应用程序应该在 `/healthz` 路径上实现一个端点，以响应健康检查的请求。这通常是一个简单的 HTTP 端点，返回 200 OK 状态表示健康。

### 五、 vs/dr和envoyfilter还有xDS的关系

envoy本质上和nginx没有什么不同。路由和流量分配信息，例如多少比例的流量发送到后端哪个ip地址，这些都是envoy的配置决定的。而后端的pod是在变化的，那就是由istiod根据pod的变化情况实时生成新的配置信息发送给envoy，他们之间使用xDS（各种discover service）协议通信。

envoy的最终的配置，是类似naginx那种偏静态的代理配置，我们暂且叫它xDS配置吧。

我们通过vs/dr，可以方便的筛选pod进行流量分配，istiod根据我们的筛选指示和当时的pod的实际情况，生成带有pod地址作为endpoint的偏静态的xDS配置信息，发送给envoy。当pod发生变化的时候，例如弹性伸缩后pod的集合与地址都可能变化，istiod会重新生成xDS配置信息，刷新给envoy。

参考：[Envoy 基础入门教程-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/2352018)

下面是vs/dr转换为 envoyfilter、并进一步转换为xDS配置的示意：

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: weighted-route
spec:
  configPatches:
  - applyTo: HTTP_ROUTE
    match:
      context: SIDECAR_OUTBOUND
      routeConfiguration:
        vhost:
          name: my-service
    patch:
      operation: MERGE
      value:
        route:
          weighted_clusters:
            clusters:
            - name: my-service-v1
              weight: 80
            - name: my-service-v2
              weight: 20

apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: cluster-definition
spec:
  configPatches:
  - applyTo: CLUSTER
    patch:
      operation: ADD
      value:
        name: "my-service-v1"
        connect_timeout: 0.25s
        type: STATIC
        lb_policy: ROUND_ROBIN
        load_assignment:
          cluster_name: "my-service-v1"
          endpoints:
          - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 127.0.0.1
                    port_value: 8081
  - applyTo: CLUSTER
    patch:
      operation: ADD
      value:
        name: "my-service-v2"
        connect_timeout: 0.25s
        type: STATIC
        lb_policy: ROUND_ROBIN
        load_assignment:
          cluster_name: "my-service-v2"
          endpoints:
          - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 127.0.0.1
                    port_value: 8082

```

上面的envoyfilter可以表示为xDS：

```json
{
  "version_info": "2023-12-09T15:09:19Z",
  "resources": [
    {
      "@type": "type.googleapis.com/envoy.config.route.v3.RouteConfiguration",
      "name": "my-service",
      "virtual_hosts": [
        {
          "name": "my-service",
          "domains": [
            "*"
          ],
          "routes": [
            {
              "match": {
                "prefix": "/"
              },
              "route": {
                "weighted_clusters": {
                  "clusters": [
                    {
                      "name": "my-service-v1",
                      "weight": 80
                    },
                    {
                      "name": "my-service-v2",
                      "weight": 20
                    }
                  ]
                }
              }
            }
          ]
        }
      ]
    }
  ]
}

{
  "version_info": "2023-12-09T15:09:19Z",
  "resources": [
    {
      "@type": "type.googleapis.com/envoy.config.cluster.v3.Cluster",
      "name": "my-service-v1",
      "connect_timeout": "0.25s",
      "type": "STATIC",
      "lb_policy": "ROUND_ROBIN",
      "load_assignment": {
        "cluster_name": "my-service-v1",
        "endpoints": [
          {
            "lb_endpoints": [
              {
                "endpoint": {
                  "address": {
                    "socket_address": {
                      "address": "127.0.0.1",
                      "port_value": 8081
                    }
                  }
                }
              }
            ]
          }
        ]
      }
    },
    {
      "@type": "type.googleapis.com/envoy.config.cluster.v3.Cluster",
      "name": "my-service-v2",
      "connect_timeout": "0.25s",
      "type": "STATIC",
      "lb_policy": "ROUND_ROBIN",
      "load_assignment": {
        "cluster_name": "my-service-v2",
        "endpoints": [
          {
            "lb_endpoints": [
              {
                "endpoint": {
                  "address": {
                    "socket_address": {
                      "address": "127.0.0.1",
                      "port_value": 8082
                    }
                  }
                }
              }
            ]
          }
        ]
      }
    }
  ]
}

```

