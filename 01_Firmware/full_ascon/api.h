#ifndef ASCON_API_H
#define ASCON_API_H

#include <stdint.h>

/*
 * ASCON-AEAD-128 API settings
 *
 * Key size   : 16 bytes = 128 bits
 * Nonce size : 16 bytes = 128 bits
 * Tag size   : 16 bytes = 128 bits
 */

#define CRYPTO_ALGNAME "Ascon-AEAD128"

#define CRYPTO_KEYBYTES 16
#define CRYPTO_NSECBYTES 0
#define CRYPTO_NPUBBYTES 16
#define CRYPTO_ABYTES 16
#define CRYPTO_NOOVERLAP 1

/*
 * Standard AEAD encryption function.
 *
 * c     : output ciphertext || tag
 * clen  : output ciphertext length + tag length
 * m     : plaintext
 * mlen  : plaintext length
 * ad    : associated data
 * adlen : associated data length
 * nsec  : unused secret nonce
 * npub  : public nonce, 16 bytes
 * k     : key, 16 bytes
 */
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

/*
 * Standard AEAD decryption function.
 *
 * Returns 0 if tag verification succeeds.
 * Returns -1 if tag verification fails.
 */
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

/*
 * Tight-trigger helper for side-channel dataset collection.
 *
 * This function should be implemented in encrypt.c.
 *
 * key         : 16-byte ASCON key
 * nonce       : 16-byte ASCON nonce
 * target_byte : selected key byte from 0 to 15
 *
 * It returns a 1-byte checksum so the compiler does not optimize away
 * the computation.
 */
uint8_t ascon_keybyte_absorb_tighttrigger(
    const uint8_t *key,
    const uint8_t *nonce,
    uint8_t target_byte
);

#endif