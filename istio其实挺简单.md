

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

