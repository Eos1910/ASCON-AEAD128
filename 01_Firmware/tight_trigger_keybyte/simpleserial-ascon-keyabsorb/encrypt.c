#include <stdint.h>
#include <string.h>

#include "api.h"
#include "hal.h"

/*
 * Compact standalone Ascon-AEAD128 implementation for ChipWhisperer/XMEGA.
 *
 * This file provides:
 *   crypto_aead_encrypt()
 *   crypto_aead_decrypt()
 *   ascon_keybyte_absorb_tighttrigger()
 *
 * The tight-trigger helper is for side-channel dataset collection.
 */

#define ASCON_AEAD_RATE 16

#define ASCON_TAG_SIZE 16
#define ASCON_PA_ROUNDS 12
#define ASCON_PB_ROUNDS 8

#define ASCON_AEAD_VARIANT 1

#define ASCON_128A_IV \
    (((uint64_t)(ASCON_AEAD_VARIANT) << 0) | \
     ((uint64_t)(ASCON_PA_ROUNDS) << 16) | \
     ((uint64_t)(ASCON_PB_ROUNDS) << 20) | \
     ((uint64_t)(ASCON_TAG_SIZE * 8) << 24) | \
     ((uint64_t)(ASCON_AEAD_RATE) << 40))

typedef struct {
    uint64_t x[5];
} ascon_state_t;

static uint64_t ROR(uint64_t x, int n)
{
    return (x >> n) | (x << ((64 - n) & 63));
}

static uint8_t GETBYTE(uint64_t x, int i)
{
    return (uint8_t)(x >> (8 * i));
}

static uint64_t SETBYTE(uint8_t b, int i)
{
    return ((uint64_t)b) << (8 * i);
}

static uint64_t PAD(int i)
{
    return SETBYTE(0x01, i);
}

static uint64_t DSEP(void)
{
    return SETBYTE(0x80, 7);
}

static uint64_t LOADBYTES(const uint8_t *bytes, int n)
{
    int i;
    uint64_t x = 0;

    for (i = 0; i < n; i++) {
        x |= SETBYTE(bytes[i], i);
    }

    return x;
}

static void STOREBYTES(uint8_t *bytes, uint64_t x, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        bytes[i] = GETBYTE(x, i);
    }
}

static uint64_t CLEARBYTES(uint64_t x, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        x &= ~SETBYTE(0xff, i);
    }

    return x;
}

static void ROUND(ascon_state_t *s, uint8_t C)
{
    ascon_state_t t;

    s->x[2] ^= C;

    s->x[0] ^= s->x[4];
    s->x[4] ^= s->x[3];
    s->x[2] ^= s->x[1];

    t.x[0] = s->x[0] ^ (~s->x[1] & s->x[2]);
    t.x[1] = s->x[1] ^ (~s->x[2] & s->x[3]);
    t.x[2] = s->x[2] ^ (~s->x[3] & s->x[4]);
    t.x[3] = s->x[3] ^ (~s->x[4] & s->x[0]);
    t.x[4] = s->x[4] ^ (~s->x[0] & s->x[1]);

    t.x[1] ^= t.x[0];
    t.x[0] ^= t.x[4];
    t.x[3] ^= t.x[2];
    t.x[2] = ~t.x[2];

    s->x[0] = t.x[0] ^ ROR(t.x[0], 19) ^ ROR(t.x[0], 28);
    s->x[1] = t.x[1] ^ ROR(t.x[1], 61) ^ ROR(t.x[1], 39);
    s->x[2] = t.x[2] ^ ROR(t.x[2], 1) ^ ROR(t.x[2], 6);
    s->x[3] = t.x[3] ^ ROR(t.x[3], 10) ^ ROR(t.x[3], 17);
    s->x[4] = t.x[4] ^ ROR(t.x[4], 7) ^ ROR(t.x[4], 41);
}

static void P12(ascon_state_t *s)
{
    ROUND(s, 0xf0);
    ROUND(s, 0xe1);
    ROUND(s, 0xd2);
    ROUND(s, 0xc3);
    ROUND(s, 0xb4);
    ROUND(s, 0xa5);
    ROUND(s, 0x96);
    ROUND(s, 0x87);
    ROUND(s, 0x78);
    ROUND(s, 0x69);
    ROUND(s, 0x5a);
    ROUND(s, 0x4b);
}

static void P8(ascon_state_t *s)
{
    ROUND(s, 0xb4);
    ROUND(s, 0xa5);
    ROUND(s, 0x96);
    ROUND(s, 0x87);
    ROUND(s, 0x78);
    ROUND(s, 0x69);
    ROUND(s, 0x5a);
    ROUND(s, 0x4b);
}

static void ascon_initialize(ascon_state_t *s, const uint8_t *key, const uint8_t *nonce)
{
    uint64_t K0 = LOADBYTES(key, 8);
    uint64_t K1 = LOADBYTES(key + 8, 8);
    uint64_t N0 = LOADBYTES(nonce, 8);
    uint64_t N1 = LOADBYTES(nonce + 8, 8);

    s->x[0] = ASCON_128A_IV;
    s->x[1] = K0;
    s->x[2] = K1;
    s->x[3] = N0;
    s->x[4] = N1;

    P12(s);

    s->x[3] ^= K0;
    s->x[4] ^= K1;
}

static void ascon_process_ad(ascon_state_t *s, const uint8_t *ad, unsigned long long adlen)
{
    if (adlen > 0) {
        while (adlen >= ASCON_AEAD_RATE) {
            s->x[0] ^= LOADBYTES(ad, 8);
            s->x[1] ^= LOADBYTES(ad + 8, 8);

            P8(s);

            ad += ASCON_AEAD_RATE;
            adlen -= ASCON_AEAD_RATE;
        }

        if (adlen >= 8) {
            s->x[0] ^= LOADBYTES(ad, 8);
            s->x[1] ^= LOADBYTES(ad + 8, adlen - 8);
            s->x[1] ^= PAD(adlen - 8);
        } else {
            s->x[0] ^= LOADBYTES(ad, adlen);
            s->x[0] ^= PAD(adlen);
        }

        P8(s);
    }

    s->x[4] ^= DSEP();
}

static void ascon_finalize(ascon_state_t *s, const uint8_t *key)
{
    uint64_t K0 = LOADBYTES(key, 8);
    uint64_t K1 = LOADBYTES(key + 8, 8);

    s->x[2] ^= K0;
    s->x[3] ^= K1;

    P12(s);

    s->x[3] ^= K0;
    s->x[4] ^= K1;
}

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
)
{
    ascon_state_t s;
    unsigned long long remaining;

    (void)nsec;

    *clen = mlen + CRYPTO_ABYTES;

    ascon_initialize(&s, k, npub);
    ascon_process_ad(&s, ad, adlen);

    remaining = mlen;

    while (remaining >= ASCON_AEAD_RATE) {
        s.x[0] ^= LOADBYTES(m, 8);
        s.x[1] ^= LOADBYTES(m + 8, 8);

        STOREBYTES(c, s.x[0], 8);
        STOREBYTES(c + 8, s.x[1], 8);

        P8(&s);

        m += ASCON_AEAD_RATE;
        c += ASCON_AEAD_RATE;
        remaining -= ASCON_AEAD_RATE;
    }

    if (remaining >= 8) {
        s.x[0] ^= LOADBYTES(m, 8);
        s.x[1] ^= LOADBYTES(m + 8, remaining - 8);

        STOREBYTES(c, s.x[0], 8);
        STOREBYTES(c + 8, s.x[1], remaining - 8);

        s.x[1] ^= PAD(remaining - 8);
    } else {
        s.x[0] ^= LOADBYTES(m, remaining);

        STOREBYTES(c, s.x[0], remaining);

        s.x[0] ^= PAD(remaining);
    }

    c += remaining;

    ascon_finalize(&s, k);

    STOREBYTES(c, s.x[3], 8);
    STOREBYTES(c + 8, s.x[4], 8);

    return 0;
}

static int constant_time_tag_check(const uint8_t *a, const uint8_t *b, int len)
{
    int i;
    uint8_t diff = 0;

    for (i = 0; i < len; i++) {
        diff |= a[i] ^ b[i];
    }

    return diff == 0 ? 0 : -1;
}

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
)
{
    ascon_state_t s;
    unsigned long long remaining;
    uint8_t tag[CRYPTO_ABYTES];

    (void)nsec;

    if (clen < CRYPTO_ABYTES) {
        return -1;
    }

    *mlen = clen - CRYPTO_ABYTES;
    remaining = *mlen;

    ascon_initialize(&s, k, npub);
    ascon_process_ad(&s, ad, adlen);

    while (remaining >= ASCON_AEAD_RATE) {
        uint64_t c0 = LOADBYTES(c, 8);
        uint64_t c1 = LOADBYTES(c + 8, 8);

        STOREBYTES(m, s.x[0] ^ c0, 8);
        STOREBYTES(m + 8, s.x[1] ^ c1, 8);

        s.x[0] = c0;
        s.x[1] = c1;

        P8(&s);

        c += ASCON_AEAD_RATE;
        m += ASCON_AEAD_RATE;
        remaining -= ASCON_AEAD_RATE;
    }

    if (remaining >= 8) {
        uint64_t c0 = LOADBYTES(c, 8);
        uint64_t c1 = LOADBYTES(c + 8, remaining - 8);

        STOREBYTES(m, s.x[0] ^ c0, 8);
        STOREBYTES(m + 8, s.x[1] ^ c1, remaining - 8);

        s.x[0] = c0;
        s.x[1] = CLEARBYTES(s.x[1], remaining - 8);
        s.x[1] |= c1;
        s.x[1] ^= PAD(remaining - 8);
    } else {
        uint64_t c0 = LOADBYTES(c, remaining);

        STOREBYTES(m, s.x[0] ^ c0, remaining);

        s.x[0] = CLEARBYTES(s.x[0], remaining);
        s.x[0] |= c0;
        s.x[0] ^= PAD(remaining);
    }

    c += remaining;

    ascon_finalize(&s, k);

    STOREBYTES(tag, s.x[3], 8);
    STOREBYTES(tag + 8, s.x[4], 8);

    return constant_time_tag_check(tag, c, CRYPTO_ABYTES);
}

/*
 * Side-channel helper.
 *
 * This is the function used by the SimpleSerial p command.
 * It creates a byte-wise key-loading point so one selected key byte can be
 * captured with a tight trigger.
 *
 * This is not a normal public ASCON API function. It is only for profiling
 * and key-byte leakage dataset collection.
 */
uint8_t ascon_keybyte_absorb_tighttrigger(
    const uint8_t *key,
    const uint8_t *nonce,
    uint8_t target_byte
)
{
    volatile uint64_t K0 = 0;
    volatile uint64_t K1 = 0;
    uint64_t N0;
    uint64_t N1;
    ascon_state_t s;
    uint8_t i;
    uint8_t checksum;

    if (target_byte >= CRYPTO_KEYBYTES) {
        return 0xff;
    }

    /*
     * Load key byte-by-byte.
     * Trigger only around the selected key byte.
     */
    for (i = 0; i < CRYPTO_KEYBYTES; i++) {
        if (i == target_byte) {
            trigger_high();
        }

        if (i < 8) {
            K0 |= SETBYTE(key[i], i);
        } else {
            K1 |= SETBYTE(key[i], i - 8);
        }

        if (i == target_byte) {
            trigger_low();
        }
    }

    N0 = LOADBYTES(nonce, 8);
    N1 = LOADBYTES(nonce + 8, 8);

    s.x[0] = ASCON_128A_IV;
    s.x[1] = K0;
    s.x[2] = K1;
    s.x[3] = N0;
    s.x[4] = N1;

    P12(&s);

    s.x[3] ^= K0;
    s.x[4] ^= K1;

    /*
     * Small checksum to prevent optimization and confirm execution.
     * This is not the key.
     */
    checksum = 0;
    checksum ^= (uint8_t)(s.x[0]);
    checksum ^= (uint8_t)(s.x[1] >> 8);
    checksum ^= (uint8_t)(s.x[2] >> 16);
    checksum ^= (uint8_t)(s.x[3] >> 24);
    checksum ^= (uint8_t)(s.x[4] >> 32);
    checksum ^= key[target_byte];

    return checksum;
}