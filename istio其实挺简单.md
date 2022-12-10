

### 一、流量控制

#### 1、ServiceEntry

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

#### 2、Virtual Service

```yaml
# 创建一个没有对应的EndPoint的K8S service
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

