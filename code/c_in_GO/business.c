#include <stdlib.h>
#include <string.h>
#include <stdio.h>

	    int MSG_BUS_SendRequest(const char * svc_name,  int svc_name_len,
	    				const char * intf_name, int intf_name_len, 
	    				const unsigned char * request,
	    				int req_len);
	    
	    int MSG_BUS_RecvResponse(unsigned char * response, int resp_max_len);



int do_business(const unsigned char* req,  int reqlen,
		unsigned char * resp,  int max_resp_len)
{

	/*
	 * 远程网络调用都通过go的“消息总线”完成，调用MSG_BUS_SendRequest向指定的远程服务和接口发送请求包，
	 * 然后调用MSG_BUS_RecvResponse收取响应的应答
	 *
	 *   int MSG_BUS_SendRequest(const char * svc_name,  int svc_name_len,
	 *   				const char * intf_name, int intf_name_len, 
	 *   				const unsigned char * request,
	 *   				int req_len);
	 *   
	 *   int MSG_BUS_RecvResponse(unsigned char * response, int resp_max_len);
	 *
	 */
	printf("in C do_business...\n");
	MSG_BUS_SendRequest("oidb",  4, "addfrd", 6, "request", 7);

	unsigned char response[1024];
	MSG_BUS_RecvResponse(response , 1024);


	return 0;
}

