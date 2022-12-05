



 本小文主要是演示如何在给pod中的应用程序（容器）访问api server的权限，例如在operator中经常需要。

一开始踩坑里了，service account + role + binding的方式总提示没有权限，我role binding的时候犯错了：

1. 我把参数写成了--user=jenkins，应该是--serviceaccount=jenkins。用户不存在k8s也不报错，反正给你建立这个绑定映射关系

2. 绑定clusterrole的时候，应该创建clusterrolebinding，我创建了rolebinding，以为只要指定--clusterrole=xxx即可，k8s也不报错。

3. 没有给到CRD所在的apiGroup的权限

   



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

#用curl验证权限,我通常path也写不对，搞不清楚
export TOKEN=...
curl --cacert /root/.minikube/ca.crt -H "Authorization: Bearer $TOKEN" -s 'https://192.168.49.2:8443/kopf.dev/ephemeralvolumeclaims'
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

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6Im45ZlFPc0JIaG5IUXprMXY1dXUxYVpjMVFyS0pDa2s3X2FXd05qMGtLeXcifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImplbmtpbnMtc2VjcmV0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6ImplbmtpbnMiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIyZTgyZDIzOS03NzlkLTRjNTktOTgxZi0xMGJmMWExMGEwN2YiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpqZW5raW5zIn0.tWqBTuKXs-bCNlyrj9DEV-tnTBGLVThzzKVBh23kSqmYVNnsUPI7JbfZYpNvxnfVg0N2ZaYQQzPOS4FDdoKxFy25oYNkvEPgMYDcD0QVU-v3tkzJrMCFVgPCj2dGIuDNUCbYBvpJEFSa3Cf2KIvCx9uBgDWNtw__vUe8k3wfQ6KjcfZL2jUQYTV-Yr-QzZZZbO2KmCO9IvGg4Hw_bGnHShDA-e4ARxZGC8GerJKSVNSA0zVGyUjNMMi_lhzD2DC2DIDDn5xuaSRjCTHyUecLlx7gzly7Dkm_9oeYR30Yw_ReNPOAQq8cNw-F9eQsZYPLZsvh8RFjyK4R2K9E4mp3BA"


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
    #不知道为什么写成这个鸟样这一行也能工作
    return kopf.login_via_client(logger=logger)

    # 下面这行不知道怎么都不对，service account凭借token不就能登录吗
	#return kopf.login_with_service_account(server='https://192.168.49.2:8443',  token=token, logger=logger, kwargs=kwargs)
    # 下面这行不知道怎么都不对，有token和token里隐含的service account，就应该能登录
    #return kopf.ConnectionInfo(server='https://192.168.49.2:8443', token=token)
    #下面这行组织到怎么都不对，有证书和私钥 就应该可以登录。来自 .kube/config
    #return kopf.ConnectionInfo(server='https://192.168.49.2:8443',  ca_path="/root/.minikube/ca.crt", certificate_path='/root/.minikube/profiles/minikube/client.crt',private_key_path='/root/.minikube/profiles/minikube/client.key')


@kopf.on.create('ephemeralvolumeclaims')
def create_fn(spec, name, namespace, body, logger, **kwargs):

    
    logger.info(body)
 

    size = spec.get('size')
    if not size:
        raise kopf.PermanentError(f"Size must be set. Got {size!r}.")

    tmpl = template
    text = tmpl.format(name=name, size=size)
    data = yaml.safe_load(text)

    api = kubernetes.client.CoreV1Api()
    obj = api.create_namespaced_persistent_volume_claim(
        namespace=namespace,
        body=data,
    )
    

    logger.info(f"PVC child is created: {obj}")
    return { "size": size}

@kopf.on.delete('ephemeralvolumeclaims')
def delete_fn(spec, name, namespace, logger, **kwargs):
    api = kubernetes.client.CoreV1Api()
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

