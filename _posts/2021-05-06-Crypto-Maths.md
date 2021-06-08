---
layout: post
title: Crypto Maths
---

__"Find factors, get money" -[Notorious T.K.G.](
https://blog.cloudflare.com/a-relatively-easy-to-understand-primer-on-elliptic-curve-cryptography/)__
<!-- __Chapter 4 summary: Ethereum Book__ -->

Can you represent 8,018,009 as a product of two prime numbers? 

You can use whatever calculator or program you like. But you've only got 10 seconds. I'll wait.

****

I couldn't do it either. Nor can a computer.

This is the guiding principle behind Crypto Maths.

In this post we'll go from a private key to an address using all the mathematical functions in between. Much of this comes from Chapter 4 of the ethereumbook. 

# Private Keys

*"Not your keys not your coins"*

Your private key is calculated from a secure source of randomness. It involves essentially picking a number between 1 and 2256. 

Ethereum's software uses the system's random number generator to product 256 random bits. 

That's it, really. Public key generation is not so easy, however. 

# Public Keys

You've heard many definitions of a Public Key. But here's the real one: 

*An Ethereum public key is a point on an elliptic curve, meaning it is a set of x and y coordinates that satisfy the elliptic curve equation. - Etheruem book. __(Rephrase this)__*

By now you're probably thinking: What's an elliptical curve? 

## Elliptical Curve

Ethereuem uses the same elliptical curve as Bitcoin (secp256k1).

Here's the equation: 

`y^2 mod(p) = ( x^3 + 7 ) mod (p)`

![img](https://hackernoon.com/hn-images/1*O8g1ldH8j2gUm1TfQdY31w.png)

The mod p indicates that this curve is only valid over the field of prime order p.

So in reality it looks more disjoint and less smooth. For `p=17` the graph looks something like this:

![img](https://github.com/ethereumbook/ethereumbook/blob/develop/images/ec_over_small_prime_field.png)

The private key is "plugged in" uses this curve to generate a public key.

## Generating a Public Key

TODO: this doesn't make sense. The equation above isn't referenced here at all.

Your private key (`k`) uses the following equation to generate a public key.

`K = k * G`

Where G is a predetermined point on the curve (called the generator point)

A more intuitive way of expressing this is as follows:

`K = (k*G1, k*G2)` where `G1` and `G2` are the x and y coordinates respectively.

```
Example (how does this work?)
Private Key: f8f8a2f43c8376ccb0871305060d7b27b0554d2cc72bccf41b2705608452f315

K = (f8f8a2f43c8376ccb0871305060d7b27b0554d2cc72bccf41b2705608452f315 * G1, f8f8a2f43c8376ccb0871305060d7b27b0554d2cc72bccf41b2705608452f315 * G2)

K = (6e145ccef1033dea239875dd00dfb4fee6e3348b84985c92f103444683bae07b, 83b5c38e5e2b0c8529d7fa3f64d46daa1ece2d9ac14cab9477d042c84c32ccd0)
```

## Serialization

Ethereum uses uncompressed keys. So here's how we can serialize the keys:

`serialized_key = 04 + x-coordinate (32 bytes/64 hex) + y-coordinate (32 bytes/64 hex)`

```
Example:

serialized_key = 04 + 6e145ccef1033dea239875dd00dfb4fee6e3348b84985c92f103444683bae07b + 83b5c38e5e2b0c8529d7fa3f64d46daa1ece2d9ac14cab9477d042c84c32ccd0

serialized_key = 046e145ccef1033dea239875dd00dfb4fee6e3348b84985c92f103444683bae07b83b5c38e5e2b0c8529d7fa3f64d46daa1ece2d9ac14cab9477d042c84c32ccd0

```

<!-- ## Python Code
Here's the code to the public key creation in python. From [stackoverflow](https://stackoverflow.com/questions/59243185/generating-elliptic-curve-private-key-in-python-with-the-cryptography-library).

```python
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key = private_key.public_key()
# serializing into PEM
rsa_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
print(rsa_pem.decode()) # print public key

``` -->

Once the public key is obtained we can derive the address. Now people can send you money!

# Cryptographic Hash Function

__"Much more than encryption algorithms, one-way hash functions are the workhorses of modern cryptography. - [Bruce Schneier](https://www.schneier.com/essays/archives/2004/08/cryptanalysis_of_md5.html)"__

Remember the intro? We saw that representing `8018009` as a product of two primes is difficult. 

But `2003 * 4003` is easy.

In other words you can go from `x->y`. But you can't go from `y->x` feasibly. This is what one-way means. 

Ethereum uses the Keccak-256 one-way hash function. Keccak-256 converts the public key into a unique address. 

An address cannot be used to obtain the public key.

```
Keccak256(K) = 2a5bc342ed616b5ba5732269001d3f1ef827552ae1114027bd3ecf1f086ba0f9
``` 

Only the last 20 bytes are used as an address. The prefix `0x` is usually added:

```
address = 0x001d3f1ef827552ae1114027bd3ecf1f086ba0f9
```

Done! 

# Conclusion/How to make Trillions of Dollars

Crypto Maths is very interesting. And it leads to an important consideration:

If you want to make trillions of dollars find a way to reverse engineer all of this.

Specifically: 

- obtain the public key from the address.
- obtain the private key from the public key.

<!-- But good luck - you'll need it. -->

It would be *extremely* difficult (virtually impossible). And it would probably spell the end of Cryptocurrencies. 

But hey, that's the price of progress. You can thank me later.

Hope this article was useful. If it was please let me know and I'll make more of these!


