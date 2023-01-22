



#### 使用k8s client java 开发包从外部访问K8S api server 并添加deployment的例子

```java
/*
 一个简单的http服务器，接收到请求后，访问K8S集群创建一个deployment，下面这个命令可以给http服务器发请求
  curl -X POST -d '{"region":"shanghai","image":"vrs"}' http://localhost:8000/createresource 
  http服务器使用的是~/.kube/config里的鉴权配置
 */
package bison.edgecontainer;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.awt.*;
import java.io.*;
import java.net.InetSocketAddress;
import java.util.*;
import java.util.List;

import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.Config;
import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1Deployment;
import io.kubernetes.client.openapi.models.V1DeploymentSpec;
import io.kubernetes.client.openapi.models.V1LabelSelector;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/createresource", new CreateResource());
        server.setExecutor(null); // creates a default executor
        server.start();
    }

    static class CreateResource implements HttpHandler {



        private int generatePortno()
        {
            return 50000 + new Random(879423).nextInt() % 10000;
        }

        private void respond(HttpExchange t, int code, String response)
        {
            try {
                t.sendResponseHeaders(code, response.getBytes().length);
                OutputStream os = t.getResponseBody();
                os.write(response.getBytes());
                os.close();
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }

        @Override
        public void handle(HttpExchange t) throws IOException
        {

            int code = 200;
            String response = "{'error':0, 'msg':'success'}";
            byte[] bytes = new byte[10240];

            InputStream in = t.getRequestBody();

            int offset = 0;
            int len = 0;

            while ( offset < bytes.length && in.available() > 0)
            {
                try
                {
                    len = in.read(bytes, offset, bytes.length-offset);

                }
                catch (Exception e)
                {
                    e.printStackTrace();
                    code = 502;
                    response = "{'error':-1, 'msg':'invalid input'}";
                    respond(t, code, response);
                    return;
                }

                offset += len;
            }

            System.out.println(new String(bytes, 0, offset));
            JSONObject json = (JSONObject)JSONValue.parse(new String(bytes, 0, offset));
            if (json == null)
            {
                code = 502;
                response = "{'error':-1, 'msg':'invalid input'}";
                respond(t, code, response);
                return;
            }
            String region = json.get("region").toString();
            String image = json.get("image").toString();
            String portno = "" + generatePortno();

            String kubeConfigPath = System.getenv("HOME") + "/.kube/config";
            FileReader reader = null;
            ApiClient client = null;

            try
            {
                reader = new FileReader(kubeConfigPath);
                client = ClientBuilder.kubeconfig(KubeConfig.loadKubeConfig(reader)).build();
            }
            catch (Exception e)
            {
                e.printStackTrace();
                return;
            }

            // loading the out-of-cluster config, a kubeconfig from file-system


            // set the global default api-client to the in-cluster one from above
            Configuration.setDefaultApiClient(client);

            List<String> cmd = new ArrayList<String>();
            cmd.add("vrs");
            List<String> args = new ArrayList<>();
            args.add(portno);
            Map<String, String> labels = new HashMap<>();
            labels.put("app", "vrs");
            Map<String, String> selector = new HashMap<>();
            selector.put("region", region);

            V1Container container = new V1Container()
                    .image(image)
                    .name("vrs")
                    .command(cmd)
                    .args(args);
            List<V1Container> containererList = new ArrayList<V1Container>();
            containererList.add(container);

            AppsV1Api api = new AppsV1Api(client);
            //这一段写起来有点麻烦，也可以直接用Yaml这个包来load一个文件，建立起V1Deployment
            //详细可以见官方example代码：
            //https://github.com/kubernetes-client/java/blob/master/examples/examples-release-15/src/main/java/io/kubernetes/client/examples/YamlExample.java
            V1Deployment dp = new V1Deployment()
                    .apiVersion("apps/v1")
                    .kind("Deployment")
                    .metadata(new V1ObjectMeta().name("vrs-"+region+"-"+portno))
                    .spec(new V1DeploymentSpec()
                            .replicas(1)
                            .selector(new V1LabelSelector().matchLabels(labels))
                            .template(new V1PodTemplateSpec()
                                    .metadata(new V1ObjectMeta().labels(labels))
                                    .spec(new V1PodSpec().containers(containererList)
                                    .nodeSelector(selector)
                                    .hostNetwork(Boolean.TRUE))));
            System.out.println(dp.toString());
            try {
                api.createNamespacedDeployment("default", dp, "true", null, null, null);
            } catch (ApiException e) {
                e.printStackTrace();
                System.out.println(e.getResponseBody());
                code = 502;
                response = String.format("{'error':-1, 'msg':'%s'}", e.toString());
            }

            respond(t, code, response);



        }
    }
    
}

```

其对应的maven配置文件：

```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>bison.edgecontainer</groupId>
  <artifactId>edgecontainer</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>edgecontainer</name>
  <!-- FIXME change it to the project's website -->
  <url>http://www.example.com</url>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>1.7</maven.compiler.source>
    <maven.compiler.target>1.7</maven.compiler.target>
  </properties>

  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.11</version>
      <scope>test</scope>
    </dependency>
    <dependency>
    <groupId>com.googlecode.json-simple</groupId>  
    <artifactId>json-simple</artifactId>  
    <version>1.1</version>  
    </dependency>
    <dependency>
      <groupId>io.kubernetes</groupId>
      <artifactId>client-java</artifactId>
      <version>17.0.0</version>
    </dependency>
  </dependencies>

  <build>
      <plugins>
        <plugin>
          <artifactId>maven-clean-plugin</artifactId>
          <version>3.1.0</version>
        </plugin>

        <plugin>
          <artifactId>maven-resources-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.8.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-surefire-plugin</artifactId>
          <version>2.22.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-jar-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-install-plugin</artifactId>
          <version>2.5.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-deploy-plugin</artifactId>
          <version>2.8.2</version>
        </plugin>
        <!-- site lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#site_Lifecycle -->
        <plugin>
          <artifactId>maven-site-plugin</artifactId>
          <version>3.7.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-project-info-reports-plugin</artifactId>
          <version>3.0.0</version>
        </plugin>
        <plugin>
          <groupId>org.apache.maven.plugins</groupId>
          <artifactId>maven-assembly-plugin</artifactId>
          <version>2.4.1</version>
          <configuration>
            <descriptorRefs>
              <descriptorRef>jar-with-dependencies</descriptorRef>
            </descriptorRefs>
            <archive>
              <manifest>
                <mainClass>bison.edgecontainer.App</mainClass>
              </manifest>
            </archive>
          </configuration>
          <executions>
            <execution>
              <id>make-assembly</id>
              <phase>package</phase>
              <goals>
                <goal>single</goal>
              </goals>
            </execution>
          </executions>
        </plugin>
      </plugins>


  </build>
</project>

```

