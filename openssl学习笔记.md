#### 1、命令行工具

##### 1.1 对称加密

```shell
openssl enc -e -aes-128-cbc -K '11223344556677889900112233445566' -iv '00000000000000000000000000000000' -in .profile -out xxx.dat 
openssl enc -d -aes-128-cbc -K '11223344556677889900112233445566' -iv '00000000000000000000000000000000' -in xxx.dat
openssl enc --help
```

#### 1.2 消息摘要

```shell
openssl dgst --help
openssl dgst -md5 .profile
openssl dgst -sha256 .profile
```

#### 1.3 生成DH公私钥

```shell
openssl dhparam -outform PEM -out xxx.dat  -2
openssl dhparam -inform PEM -in xxx.dat  -C
openssl dhparam --help
#新版本openssl命令行工具已经不能生成DH的公私钥了，只能产生p和g两个参数，不知道为何这样设计
```

#### 1.4 生成ECDH公私钥

```shell
openssl ecparam -help
openssl ecparam -list_curves
openssl ecparam -genkey    -name secp521r1 -outform PEM -out priv.pem
openssl ec -in priv.pem -pubout
openssl ec -in priv.pem -pubout -text
```

#### 1.5 生成DSA 公私钥

```shell
openssl dsaparam  -genkey -outform PEM -out priv.pem 1024
openssl dsa -in priv.pem  -pubout
```

#### 1.6 生成RSA公私钥并加解密

```shell
openssl genrsa  -out priv.pem 1024
openssl rsa  -in  priv.pem  -pubout -out pub.pem
openssl rsautl -encrypt -pubin -keyform PEM  -inkey pub.pem  -in plain.txt  -out cipher.txt
openssl rsautl -decrypt  -keyform PEM  -inkey priv.pem  -in cipher.txt
```



### 2、代码

#### 2.1 DH

```c
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "hex_dump.h"
#include <openssl/dh.h>
#include <openssl/err.h>
#include <openssl/rand.h>

using namespace std;

#define PRIME_LEN (512)

int main()
{
    int ret1, ret2;
    DH * handle1=NULL, *handle2=NULL;
    unsigned char key1[1024];
    unsigned char key2[1024];
    int keylen, code1, code2;
    int err;

    unsigned char buf[100];
    BIGNUM * bignum = NULL;

   if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }

    handle1 = DH_new();
    handle2 = DH_new();
    if (handle1 == NULL || handle2 == NULL)
    {
        fprintf(stderr, "dh_new failed\n");
        goto end;
    }

    ret1 = DH_generate_parameters_ex(handle1, PRIME_LEN, DH_GENERATOR_2, NULL);
    ret2 = DH_generate_parameters_ex(handle2, PRIME_LEN, DH_GENERATOR_2, NULL);
    if (ret1 == 0||ret2 == 0)
    {
        fprintf(stderr, "DH_generate_parameters_ex failed\n");
        goto end;
    }

    ret1 = DH_check(handle1, &code1);
    ret2 = DH_check(handle2, &code2);
    if (ret1 == 0 || ret2 == 0 || code1 || code2)
    {
        fprintf(stderr, "DH_check failed\n");
        goto end;
    }

    BN_copy(handle1->g, handle2->g);
    BN_copy(handle1->p, handle2->p);
 
    
    ret1 = DH_generate_key(handle1);
    ret2 = DH_generate_key(handle2);
    if (ret1 != 1||ret2 !=1 || handle1->pub_key == NULL || handle2->pub_key == NULL)
    {
        fprintf(stderr, "DH_generate_key failed %d %d\n", ret1, ret2);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    keylen = DH_size(handle1);
    printf("key len:%d\n", keylen);

    ret1 = DH_compute_key(key1, handle1->pub_key, handle2);
    ret2 = DH_compute_key(key2, handle2->pub_key, handle1);
    if (ret1 != keylen || ret2 != keylen )
    {
        fprintf(stderr, "DH_compute_key failed %d %d\n", ret1, ret2);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        
        goto end;
    }
    if (memcmp(key1, key2, keylen) != 0)
    {
        fprintf(stderr, "key diff!\n");
       
    }
    dash::hex_dump(key1, keylen, std::cout);
    printf("\n");
    dash::hex_dump(key2, keylen, std::cout);
    

end:
    if (handle1)
    {
        DH_free(handle1);
    }
    if (handle2)
    {
        DH_free(handle2);
    }

}

```

#### 2.2 DSA算法

```c
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "hex_dump.h"
#include <openssl/dsa.h>
#include <openssl/err.h>
#include <openssl/rand.h>

using namespace std;

#define PRIME_LEN (512)

int main()
{
    int ret;
    DSA * handle=NULL;
    int keylen;
    int err;

    unsigned char buf[100];
   

    unsigned char dgst[20] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0};
    unsigned char sig[1024];
    unsigned int siglen = sizeof(sig);

    if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    

    handle = DSA_new();
    if (handle == NULL)
    {
        fprintf(stderr, "dsa_new failed\n");
        goto end;
    }

    ret = DSA_generate_parameters_ex(handle, PRIME_LEN, NULL, 0, NULL, NULL, NULL );
    if (ret != 1 )
    {
        fprintf(stderr, "DSA_generate_parameters_ex failed\n");
        goto end;
    }

    
    ret = DSA_generate_key(handle);
    if (ret != 1 )
    {
        fprintf(stderr, "DSA_generate_key failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    keylen = DSA_size(handle);
    printf("key len:%d\n", keylen);

    
    ret = DSA_sign(0, dgst, 20, sig, &siglen, handle);
    if (ret != 1)
    {
        fprintf(stderr, "DSA_sign failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    dash::hex_dump(sig, siglen, std::cout);
    //dgst[0] = 11;
    ret = DSA_verify(0, dgst, 20, sig, siglen, handle);
    if (ret == 1)
    {
        fprintf(stdout, "match!\n");
    }
    else if (ret == 0)
    {
        fprintf(stdout, "mismatch!\n");
    }
    else
    {
        fprintf(stderr, "DSA_verify failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    

end:
    if (handle)
    {
        DSA_free(handle);
    }
}

```
#### 2.3 RSA算法

```c
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "hex_dump.h"
#include <openssl/rsa.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/objects.h>

using namespace std;

#define PRIME_LEN (512)

int main()
{
    int ret;
    RSA * handle=NULL;
    int keylen;
    int err;

    unsigned char dgst[20] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0};
    unsigned char sig[1024];
    unsigned int siglen = sizeof(sig);
    unsigned char cipher[1024];
    int len;

    

    unsigned char buf[100];
    if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    handle =  RSA_generate_key(1024, RSA_3, NULL, NULL);
    if (handle == NULL)
    {
        fprintf(stderr, "RSA_generate_key failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
   
    keylen = RSA_size(handle);
    printf("key len:%d\n", keylen);

    // digital signature
    ret = RSA_sign(NID_sha1, dgst, 20, sig, &siglen, handle);
    if (ret != 1)
    {
        fprintf(stderr, "RSA_sign failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    dash::hex_dump(sig, siglen, std::cout);
    //dgst[0] = 11;
    ret = RSA_verify(NID_sha1, dgst, 20, sig, siglen, handle);
    if (ret == 1)
    {
        fprintf(stdout, "match!\n");
    }
    else if (ret == 0)
    {
        fprintf(stdout, "mismatch!\n");
    }
    else
    {
        fprintf(stderr, "RSA_verify failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    // challenge or transport key
    len = RSA_public_encrypt(20, dgst, cipher, handle, RSA_PKCS1_OAEP_PADDING );
    if (len <= 0)
    {
        fprintf(stderr, "RSA_public_encrypt failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    len = RSA_private_decrypt(len, cipher, dgst, handle, RSA_PKCS1_OAEP_PADDING);
    if (len != 20)
    {
        fprintf(stderr, "RSA_private_decrypt failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    dash::hex_dump(dgst, len, std::cout);

    // signature
    len = RSA_private_encrypt(20, dgst, cipher, handle, RSA_PKCS1_PADDING);
    if (len <= 0)
    {
        fprintf(stderr, "RSA_private_encrypt failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    len = RSA_public_decrypt(len, cipher, dgst, handle, RSA_PKCS1_PADDING);
    if (len != 20)
    {
        fprintf(stderr, "RSA_public_decrypt failed %d\n", ret);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    printf("RSA_public_decrypt succeed.\n");
    dash::hex_dump(dgst, len, std::cout);

end:
    if (handle)
    {
        RSA_free(handle);
    }
  

}
```

#### 2.4 ECDH and ECDSA
```c
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "hex_dump.h"
#include <openssl/ec.h>
#include <openssl/ecdh.h>
#include <openssl/ecdsa.h>

#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/objects.h>

using namespace std;

#define PRIME_LEN (512)

int main()
{
    int i, ret;
    EC_KEY * eckey1 = NULL, *eckey2 = NULL;
    int keylen;
    int err, ret1, ret2;
    ECDSA_SIG *sig = NULL;

    unsigned char dgst[20] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0};

    unsigned char buf[200];
    if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }

    eckey1 = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
    eckey2 = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
    if (eckey1 == NULL || eckey2 == NULL)
    {
        fprintf(stderr, "EC_KEY_new_by_curve_name failed %d\n", ret1);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    
    ret1 = EC_KEY_generate_key(eckey1);
    ret2 = EC_KEY_generate_key(eckey2);
    if (ret1 != 1 || ret2 != 1)
    {
        fprintf(stderr, "EC_KEY_generate_key failed %d\n", ret1);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    ret1 = EC_KEY_check_key(eckey1);
    ret2 = EC_KEY_check_key(eckey2);
    if (ret1 != 1 || ret2 != 1)
    {
        fprintf(stderr, "EC_KEY_check_key failed %d\n", ret1);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    // ECDH key agreement
    keylen = ECDH_compute_key(buf, sizeof(buf), EC_KEY_get0_public_key(eckey1), eckey2, NULL);
    if (keylen <= 0)
    {
        fprintf(stderr, "ECDH_compute_key failed %d\n", keylen);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    dash::hex_dump(buf, keylen, std::cout);
    printf("\n\n");

    keylen = ECDH_compute_key(buf, sizeof(buf), EC_KEY_get0_public_key(eckey2), eckey1, NULL);
    if (keylen <= 0)
    {
        fprintf(stderr, "ECDH_compute_key failed %d\n", keylen);
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    dash::hex_dump(buf, keylen, std::cout);
    printf("\n\n");

    // ECDSA sign and verify
    sig = ECDSA_do_sign(dgst, 20, eckey1);
    if (sig == NULL)
    {
        fprintf(stderr, "ECDSA_do_sign failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    //dgst[0] = 3;
    ret = ECDSA_do_verify(dgst, 20, sig, eckey1);
    if (ret == 1)
    {
        printf("match!\n");
    }
    else if (ret == 0)
    {
        printf("mismatch!\n");
    }
    else 
    {
        fprintf(stderr, "ECDSA_do_verify failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    // another functions, ECDSA_sign() ECDSA_verify exist .

end:
    if (eckey1)
    {
        EC_KEY_free(eckey1);
    }
    if (eckey2)
    {
        EC_KEY_free(eckey2);
    }
    if (sig)
    {
        ECDSA_SIG_free(sig);
    }
  

}
```
### 3、辅助类

#### 3.1 随机数

RAND_seed()  RAND_load_file() RAND_bytes()...

#### 3.2 大整数运算

BN_new()  BN_add() BN_sub() BN_bn2bin() BN_dec2bn()...

不能表示有理数无理数，只能表示整数。如果要用到小数，可以考虑GMP：https://gmplib.org/

#### 3.3 IO抽象类

BIO_xxx...

### 4、对称密码算法的代码

Stream ciphers  are essentially just cryptographic pseudorandom number generators. They use a starting seed as a  key to produce a stream of random bits known as the keystream. To encrypt data, one takes the  plaintext and simply XORs it with the keystream。

流式密码算法本质上是一个随机数发生器，使用种子初始化后不断的产生随机bit，即密钥流。用密钥流和明文数据做简单的异或即完成了加密。相比分组密码算法，流式密码算法的优势是运算快。为了避免出错，流式密码算法通常伴随MAC的使用。

openssl提供封装过后的EVP_xxx相关函数，使得对称加解密更加方便：

```c
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "hex_dump.h"
#include <openssl/evp.h>

#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/objects.h>

using namespace std;

#define PRIME_LEN (512)

int main()
{
    int i, ret;
    
    int keylen;
    int err, ret1, ret2;
   
    BIGNUM * a = NULL;
    EVP_CIPHER_CTX ctx;
    unsigned char key[16];
    unsigned char iv[16];
    char plaintext[1024] = "It returns 0 if the pseudorandom number generator cannot be seeded securely.";
    unsigned char ciphertext[1024];
    int offset = 0;
    int len = sizeof(ciphertext);
    int cipherlen;

    if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    RAND_bytes(key, sizeof(key));
    RAND_bytes(iv, sizeof(iv));
    // encrypt
    ret = EVP_EncryptInit(&ctx, EVP_aes_128_cbc(), key, iv);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptInit failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    ret = EVP_EncryptUpdate(&ctx, ciphertext+offset, &len, (const unsigned char *)plaintext, strlen(plaintext));
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    len = sizeof(ciphertext) - offset;
    ret = EVP_EncryptFinal(&ctx, ciphertext+offset, &len);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    cipherlen = offset;
    printf("cipher text len:%d\n", cipherlen);

    dash::hex_dump(ciphertext, offset, std::cout);

    // decrypt
    ret = EVP_DecryptInit(&ctx, EVP_aes_128_cbc(), key, iv);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptInit failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset = 0;
    len = sizeof(plaintext);
    ret = EVP_DecryptUpdate(&ctx, (unsigned char*)plaintext+offset, &len, ciphertext, cipherlen);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    printf("decrypt len:%d\n", len);
    offset += len;
    len = sizeof(plaintext) - offset;
    ret = EVP_DecryptFinal(&ctx, (unsigned char*)plaintext+offset, &len);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptFinal failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    printf("plain text len:%d\n", offset);

    dash::hex_dump(plaintext, offset, std::cout);

end:
    return 0;
      
  

}

```


