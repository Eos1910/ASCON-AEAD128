#ifndef ASCON_API_H
#define ASCON_API_H

#include <stdint.h>

#define CRYPTO_ALGNAME "Ascon-AEAD128"

#define CRYPTO_KEYBYTES 16
#define CRYPTO_NSECBYTES 0
#define CRYPTO_NPUBBYTES 16
#define CRYPTO_ABYTES 16
#define CRYPTO_NOOVERLAP 1

int crypto_aead_encrypt(
    unsigned char *c,
    unsigned long long *clen,
    const unsigned char *m,
    unsigned long long mlen,
    const unsigned char *ad,
    unsigned long long adlen,
    const unsigned char *nsec,
    const unsigned char *npub,
    const unsigned char *k
);

int crypto_aead_decrypt(
    unsigned char *m,
    unsigned long long *mlen,
    unsigned char *nsec,
    const unsigned char *c,
    unsigned long long clen,
    const unsigned char *ad,
    unsigned long long adlen,
    const unsigned char *npub,
    const unsigned char *k
);

uint8_t ascon_keybyte_absorb_tighttrigger(
    const uint8_t *key,
    const uint8_t *nonce,
    uint8_t target_byte
);

#endif