



 本小文主要是演示如何在给pod中的应用程序（容器）访问api server的权限，例如在operator中经常需要。

一开始踩坑里了，service account + role + binding的方式总提示没有权限，我role binding的时候犯错了：

1. 我把参数写成了--user=jenkins，应该是--serviceaccount=jenkins。用户不存在k8s也不报错，反正给你建立这个绑定映射关系

2. 绑定clusterrole的时候，应该创建clusterrolebinding，我创建了rolebinding，以为只要指定--clusterrole=xxx即可，k8s也不报错。

3. 没有给到CRD所在的apiGroup的权限

4. 我只注意了kopf的权限，一开始没有注意到operator里kubernetes client lib的权限，我靠我搞了半天

   



### 第一步：创建账号和账号的token，并记录命令显示的token信息，在kopf的代码里要用到

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: jenkins-secret
  annotations:
    kubernetes.io/service-account.name: jenkins
type: kubernetes.io/service-account-token

kubectl create serviceaccount jenkins
kubectl create -f secret.yaml
kubectl describe sa jenkins
kubectl describe secret jenkins-secret #这里会显示token，用于kopf的代码
```

多说几句，如果不确定yaml文件的格式该怎么写，可以查k8s的官方api文档，但似乎也说得不清不楚：

```
https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.19/#secret-v1-core
```



### 第二步：创建role，并bind到jenkins账号

```yaml
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: evc-operator
rules:
- apiGroups: [""] # "" 标明 core API 组
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["kopf.dev"] # 我的crd的group
  resources: ["*"]
  verbs: ["*"]
  
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  namespace: default
  name: evc-operator
rules:
- apiGroups: [""] # "" 标明 core API 组
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["kopf.dev"] # 我的crd的group
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apiextensions.k8s.io"] #CRD所在的apiGroup的权限
  resources: ["*"]
  verbs: ["*"]


kubectl  create -f role.yaml
kubectl create rolebinding jenkins-binding --role=evc-operator --serviceaccount=default:jenkins
kubectl create clusterrolebinding jenkins-binding2 --clusterrole=evc-operator --serviceaccount=default:jenkins

#用curl验证权限,我通常path也写不对，需要查文档
export TOKEN=...
curl --cacert /root/.minikube/ca.crt -H "Authorization: Bearer $TOKEN" -s 'https://192.168.49.2:8443/apis/kopf.dev/v1/ephemeralvolumeclaims/'

curl --cacert /root/.minikube/ca.crt -H "Authorization: Bearer $TOKEN" -s 'https://192.168.49.2:8443/apis/kopf.dev/v1/namespaces/default/ephemeralvolumeclaims/my-claim'
```

多说几句：

curl命令访问k8s的restful api接口，往往不知道path怎么写，这里有个好文档：

```
https://github.com/kubernetes-client/python/blob/master/kubernetes/README.md#documentation-for-api-endpoints
```

这部分表格，除了告诉我们某种资源怎么写path，还告诉我们python的kubernetes client api怎么用，以创建CRD和创建CR为例：

```python
api = kubernetes.client.ApiextensionsV1Api()
api.create_custom_resource_definition(body)

api = kubernetes.client.CustomObjectsApi()
api.create_namespaced_custom_object(group, version, namespace, plural, body)
```



### 第三步：创建CRD和CR，详细见kopf的官方文档

CRD:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: ephemeralvolumeclaims.kopf.dev
spec:
  scope: Namespaced
  group: kopf.dev
  names:
    kind: EphemeralVolumeClaim
    plural: ephemeralvolumeclaims
    singular: ephemeralvolumeclaim
    shortNames:
      - evcs
      - evc
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              x-kubernetes-preserve-unknown-fields: true
            status:
              type: object
              x-kubernetes-preserve-unknown-fields: true
```

CR:

```yaml
kind: EphemeralVolumeClaim
metadata:
  name: my-claim3
spec:
  size: 2G

---
apiVersion: kopf.dev/v1
kind: EphemeralVolumeClaim
metadata:
  name: my-claim2
spec:
  size: 2G

---
apiVersion: kopf.dev/v1
kind: EphemeralVolumeClaim
metadata:
  name: my-claim
spec:
  size: 2G

```

### 第四步：创建operator和镜像

operator evc.py:

```python
import kopf
import kubernetes
import yaml
import logging
import datetime

token = "eyJhbGciOiJSUzI1NiIsImtpZC..."
cadata = 'LS0tLS1CRUdJTiBDRVJUSU...'
clientcadata='LS0tLS1CRUdJTiBDRV...'
clientkeydata='LS0tLS1CRUdJTiBS...'
serverUrl = 'https://169.254.128.18:60002'

template = '''
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: "{name}"
  annotations:
    volume.beta.kubernetes.io/storage-class: standard
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: "{size}"
'''

@kopf.on.login()
def login_fn(logger, **kwargs):
    #不知道为什么写成这个鸟样这一行也能工作, login_via_client源代码也没有看太懂
    return kopf.login_via_client(logger=logger)

    # 这两行是可以的，一个是用的sa的token，一个是用的kubectl的用户证书，从~/.kube/config里可以拿到
    return kopf.ConnectionInfo(server=serverUrl, token=token, scheme='Bearer',  ca_data=cadata, expiration=datetime.datetime(2099, 12, 31, 23, 59, 59))
    return kopf.ConnectionInfo(server=serverUrl,  ca_data=cadata, certificate_data=clientcadata, private_key_data=clientkeydata, expiration=datetime.datetime(2099, 12, 31, 23, 59, 59))

	
    #下面这行在K8S集群外调试的时候怎么都不行，后来看login_with_service_account的代码发现，它只适合在容器内工作，因为它hardcode到如下目录去读鉴权信息：
    #token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    #ns_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    #ca_path = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    return kopf.login_with_service_account(logger=logger, kwargs=kwargs)


@kopf.on.create('ephemeralvolumeclaims')
def create_fn(spec, name, namespace, body, logger, **kwargs):
    logger.info(body)
    size = spec.get('size')
    if not size:
        raise kopf.PermanentError(f"Size must be set. Got {size!r}.")

    tmpl = template
    text = tmpl.format(name=name, size=size)
    data = yaml.safe_load(text)

    configuration = kubernetes.client.Configuration()
    # Configure API key authorization: BearerToken
    configuration.api_key['authorization'] = token
    # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
    configuration.api_key_prefix['authorization'] = 'Bearer'
    configuration.verify_ssl=False

    # Defining host is optional and default to http://localhost
    configuration.host = serverUrl
    api = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
    obj = api.create_namespaced_persistent_volume_claim(
        namespace=namespace,
        body=data,
    )


    logger.info(f"PVC child is created: {obj}")
    return { "size": size}

@kopf.on.delete('ephemeralvolumeclaims')
def delete_fn(spec, name, namespace, logger, **kwargs):
    configuration = kubernetes.client.Configuration()
    # Configure API key authorization: BearerToken
    configuration.api_key['authorization'] = token
    # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
    configuration.api_key_prefix['authorization'] = 'Bearer'
    configuration.verify_ssl=False

    # Defining host is optional and default to http://localhost
    configuration.host = serverUrl
    api = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
    obj = api.delete_namespaced_persistent_volume_claim(name, namespace)

    logger.info(f"PVC child is deleted: {obj}")
@kopf.on.update('ephemeralvolumeclaims')
def update_fn(spec,name, status, namespace, logger, **kwargs):
    size = spec.get('size', None)
    if not size:
        raise kopf.PermanentError(f"Size must be set. Got {size!r}.")

    '''pvc_patch = {'spec': {'resources': {'requests': {'storage': size}}}}
    api = kubernetes.client.CoreV1Api()
    obj = api.patch_namespaced_persistent_volume_claim(name, namespace, body=pvc_patch)

    logger.info(f"PVC child is patched: {obj}")'''

@kopf.on.field('ephemeralvolumeclaims', field='spec.size')
def size_changed(old, new, logger,**kwargs):
    logger.info(f"{old}->{new}")

```

Dockerfile:

```dockerfile
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python && \
    apt-get install -y net-tools && \
    apt-get install -y curl && \
    apt-get install -y python3-pip

RUN pip3 install  kopf && \
    pip3 install  kubernetes


COPY ./evc.py /

ENTRYPOINT kopf run /evc.py --verbose
```

```shell
docker build .  -t bisonliao/evc
docker push bisonliao/evc
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: evc-operator
 labels:
     app: evc-operator
spec:
 replicas: 1
 selector:
   matchLabels:
     app: evc-operator
 template:
    metadata:
      labels:
        app: evc-operator
    spec:
      #serviceAccountName: jenkins  #如果operator用kopf.login_with_service_account方式login k8s，就可以在这里配置用到的serviceaccount
      containers:
      - image: bisonliao/evc
        name: evc-operator

```

### 第五步：见证成功：

```shell
root@VM-16-7-ubuntu:~/kopf# kubectl create -f obj.yml
ephemeralvolumeclaim.kopf.dev/my-claim3 created
ephemeralvolumeclaim.kopf.dev/my-claim2 created
ephemeralvolumeclaim.kopf.dev/my-claim created
root@VM-16-7-ubuntu:~/kopf# kubectl create -f obj.yml ^C
root@VM-16-7-ubuntu:~/kopf# kubectl get pvc
NAME        STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
my-claim    Bound    pvc-a365f41e-a92a-4280-8cdc-58655bf15725   2G         RWO            standard       12s
my-claim2   Bound    pvc-5dab0b5e-cc36-408f-8d4c-3ae82661745e   2G         RWO            standard       12s
my-claim3   Bound    pvc-05be5e65-53f8-4ce6-9782-ae22b019cb7b   2G         RWO            standard       12s
root@VM-16-7-ubuntu:~/kopf# kubectl delete -f obj.yml
ephemeralvolumeclaim.kopf.dev "my-claim3" deleted
ephemeralvolumeclaim.kopf.dev "my-claim2" deleted
ephemeralvolumeclaim.kopf.dev "my-claim" deleted
root@VM-16-7-ubuntu:~/kopf# kubectl get pvc
No resources found in default namespace.

root@VM-16-7-ubuntu:~/kopf# kubectl logs evc-operator-84df664cd-tgn47 #可以看到operator的详细日志
```

### 多说几句身份与权限

#### 验证方式：

可以看到，K8S至少支持两种验证方式：

1. serviceaccount + bearer token的方式。token里面本身带有serviceaccount，无需额外提供serviceaccount。可以使用secret为sa创建一个token
2. 用户证书的方式，认证用户。证书本身带有用户id，无需额外提供userid，往往用于user的验证，没有测试是否可以用于sa。相比sa，不存在user这样一个实体的k8s资源

同时，如果要校验 API server的身份，需要提供ca证书；否则可以不提供，某些实现需要特别指明，例如上面的kubernetes client api，就要调用一下:

```
configuration.verify_ssl=False
```

两种验证方式我们都已经在operator代码的login事件里有演示。 通过minikube的kubectl配置文件，可以更直观感受第二种验证方式：

```yaml
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    client-certificate: /root/.minikube/profiles/minikube/client.crt
    client-key: /root/.minikube/profiles/minikube/client.key
```

这个user叫 minikubeCA，我们可以从证书里的看到用户身份：

```shell
root@VM-16-7-ubuntu:~/kopf# openssl x509 -in  /root/.minikube/profiles/minikube/client.crt -noout -text
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 2 (0x2)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: CN = minikubeCA
        Validity
            Not Before: Aug 20 12:39:22 2022 GMT
            Not After : Aug 20 12:39:22 2025 GMT
        Subject: O = system:masters, CN = minikube-user
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                RSA Public-Key: (2048 bit)
```

如何添加用户和用户证书，可以查K8S官方资料Normal User这一段：

```
https://kubernetes.io/docs/reference/access-authn-authz/certificate-signing-requests/
```

#### 权限管理

主要是两种组合：

1. role + rolebinding
2. clusterrole + clusterrolebinding

他们都可以binding user，也可以binding serviceaccount。 上面已经演示了。权限只能白名单叠加（或的关系），不能做黑名单。