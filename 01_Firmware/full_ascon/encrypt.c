/*
 * encrypt.c - ASCON-AEAD-128 implementation for ChipWhisperer/XMEGA FYP use
 *
 * Algorithm: Ascon-AEAD128, NIST SP 800-232 style reference implementation.
 * Interface: eBACS/LWC crypto_aead_encrypt() and crypto_aead_decrypt().
 *
 * This file is self-contained: it does not need separate ascon.h/constants.h/word.h files.
 * It matches the api.h used in the ASCON_FYP scaffold:
 *   CRYPTO_KEYBYTES  = 16
 *   CRYPTO_NPUBBYTES = 16
 *   CRYPTO_ABYTES    = 16
 *
 * Notes for the side-channel experiment:
 *   - crypto_aead_encrypt() is the real ASCON-AEAD-128 encryption routine.
 *   - ascon_keybyte_absorb_tighttrigger() is a helper for your byte-wise key-recovery
 *     trace collection. It performs the real ASCON initialization and then applies
 *     the second key-injection step byte by byte so that a selected key byte can be
 *     isolated in the firmware trigger.
 *
 * Implementation basis:
 *   - NIST SP 800-232 Ascon-AEAD128 specification.
 *   - ascon-c reference implementation structure, released as CC0-1.0.
 */

#include <stdint.h>
#include <string.h>
#include "api.h"

/* -------------------------------------------------------------------------
 * ASCON constants
 * ------------------------------------------------------------------------- */

#ifndef CRYPTO_KEYBYTES
#define CRYPTO_KEYBYTES 16
#endif

#ifndef CRYPTO_NPUBBYTES
#define CRYPTO_NPUBBYTES 16
#endif

#ifndef CRYPTO_ABYTES
#define CRYPTO_ABYTES 16
#endif

#define ASCON_AEAD_VARIANT 1
#define ASCON_TAG_SIZE 16
#define ASCON_128A_RATE 16
#define ASCON_PA_ROUNDS 12
#define ASCON_128A_PB_ROUNDS 8

#define ASCON_128A_IV \
    (((uint64_t)(ASCON_AEAD_VARIANT) << 0) | \
     ((uint64_t)(ASCON_PA_ROUNDS) << 16) | \
     ((uint64_t)(ASCON_128A_PB_ROUNDS) << 20) | \
     ((uint64_t)(ASCON_TAG_SIZE * 8) << 24) | \
     ((uint64_t)(ASCON_128A_RATE) << 40))

/* -------------------------------------------------------------------------
 * Optional trigger macros for the key-byte helper.
 *
 * If you want the trigger to be inside ascon_keybyte_absorb_tighttrigger(),
 * compile with -DASCON_ENABLE_KEYBYTE_TRIGGER and make sure hal.h is available.
 * If not defined, these macros do nothing.
 * ------------------------------------------------------------------------- */

#ifdef ASCON_ENABLE_KEYBYTE_TRIGGER
#include "hal.h"
#define ASCON_TRIGGER_HIGH() trigger_high()
#define ASCON_TRIGGER_LOW()  trigger_low()
#else
#define ASCON_TRIGGER_HIGH() do { } while (0)
#define ASCON_TRIGGER_LOW()  do { } while (0)
#endif

/* -------------------------------------------------------------------------
 * Internal state and word utilities
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t x[5];
} ascon_state_t;

static uint64_t ascon_ror(uint64_t x, int n)
{
    return (x >> n) | (x << ((-n) & 63));
}

#define GETBYTE(x, i) ((uint8_t)((uint64_t)(x) >> (8 * (i))))
#define SETBYTE(b, i) ((uint64_t)((uint8_t)(b)) << (8 * (i)))
#define PAD(i)        SETBYTE(0x01, (i))
#define DSEP()        SETBYTE(0x80, 7)

static uint64_t load_bytes(const uint8_t *bytes, int n)
{
    int i;
    uint64_t x = 0;
    for (i = 0; i < n; ++i) {
        x |= SETBYTE(bytes[i], i);
    }
    return x;
}

static void store_bytes(uint8_t *bytes, uint64_t x, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        bytes[i] = GETBYTE(x, i);
    }
}

static uint64_t clear_bytes(uint64_t x, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        x &= ~SETBYTE(0xff, i);
    }
    return x;
}

/* -------------------------------------------------------------------------
 * ASCON permutation
 * ------------------------------------------------------------------------- */

static void ascon_round(ascon_state_t *s, uint8_t C)
{
    ascon_state_t t;

    /* Add round constant. */
    s->x[2] ^= C;

    /* Substitution layer. */
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

    /* Linear diffusion layer. */
    s->x[0] = t.x[0] ^ ascon_ror(t.x[0], 19) ^ ascon_ror(t.x[0], 28);
    s->x[1] = t.x[1] ^ ascon_ror(t.x[1], 61) ^ ascon_ror(t.x[1], 39);
    s->x[2] = t.x[2] ^ ascon_ror(t.x[2], 1)  ^ ascon_ror(t.x[2], 6);
    s->x[3] = t.x[3] ^ ascon_ror(t.x[3], 10) ^ ascon_ror(t.x[3], 17);
    s->x[4] = t.x[4] ^ ascon_ror(t.x[4], 7)  ^ ascon_ror(t.x[4], 41);
}

static void P12(ascon_state_t *s)
{
    ascon_round(s, 0xf0);
    ascon_round(s, 0xe1);
    ascon_round(s, 0xd2);
    ascon_round(s, 0xc3);
    ascon_round(s, 0xb4);
    ascon_round(s, 0xa5);
    ascon_round(s, 0x96);
    ascon_round(s, 0x87);
    ascon_round(s, 0x78);
    ascon_round(s, 0x69);
    ascon_round(s, 0x5a);
    ascon_round(s, 0x4b);
}

static void P8(ascon_state_t *s)
{
    ascon_round(s, 0xb4);
    ascon_round(s, 0xa5);
    ascon_round(s, 0x96);
    ascon_round(s, 0x87);
    ascon_round(s, 0x78);
    ascon_round(s, 0x69);
    ascon_round(s, 0x5a);
    ascon_round(s, 0x4b);
}

/* -------------------------------------------------------------------------
 * Small helpers
 * ------------------------------------------------------------------------- */

static void ascon_initialize(ascon_state_t *s, const uint8_t *npub, const uint8_t *k,
                             uint64_t *K0_out, uint64_t *K1_out)
{
    const uint64_t K0 = load_bytes(k, 8);
    const uint64_t K1 = load_bytes(k + 8, 8);
    const uint64_t N0 = load_bytes(npub, 8);
    const uint64_t N1 = load_bytes(npub + 8, 8);

    s->x[0] = ASCON_128A_IV;
    s->x[1] = K0;
    s->x[2] = K1;
    s->x[3] = N0;
    s->x[4] = N1;

    P12(s);

    s->x[3] ^= K0;
    s->x[4] ^= K1;

    if (K0_out) *K0_out = K0;
    if (K1_out) *K1_out = K1;
}

static uint8_t ascon_verify_tag(const uint8_t *a, const uint8_t *b, unsigned int len)
{
    uint8_t diff = 0;
    unsigned int i;
    for (i = 0; i < len; ++i) {
        diff |= a[i] ^ b[i];
    }
    return diff;
}

static uint8_t ascon_state_checksum(const ascon_state_t *s)
{
    uint8_t out = 0;
    uint8_t tmp[40];
    int i;

    store_bytes(tmp,      s->x[0], 8);
    store_bytes(tmp + 8,  s->x[1], 8);
    store_bytes(tmp + 16, s->x[2], 8);
    store_bytes(tmp + 24, s->x[3], 8);
    store_bytes(tmp + 32, s->x[4], 8);

    for (i = 0; i < 40; ++i) {
        out ^= (uint8_t)(tmp[i] + (uint8_t)(i * 13u));
    }
    return out;
}

/* -------------------------------------------------------------------------
 * ASCON-AEAD-128 encryption
 * ------------------------------------------------------------------------- */

int crypto_aead_encrypt(
    unsigned char *c, unsigned long long *clen,
    const unsigned char *m, unsigned long long mlen,
    const unsigned char *ad, unsigned long long adlen,
    const unsigned char *nsec,
    const unsigned char *npub,
    const unsigned char *k)
{
    ascon_state_t s;
    uint64_t K0, K1;
    unsigned char *c_start = c;

    (void)nsec;

    *clen = mlen + CRYPTO_ABYTES;

    ascon_initialize(&s, npub, k, &K0, &K1);

    /* Associated data. */
    if (adlen) {
        while (adlen >= ASCON_128A_RATE) {
            s.x[0] ^= load_bytes(ad, 8);
            s.x[1] ^= load_bytes(ad + 8, 8);
            P8(&s);
            ad += ASCON_128A_RATE;
            adlen -= ASCON_128A_RATE;
        }

        if (adlen >= 8) {
            s.x[0] ^= load_bytes(ad, 8);
            s.x[1] ^= load_bytes(ad + 8, (int)(adlen - 8));
            s.x[1] ^= PAD((int)(adlen - 8));
        } else {
            s.x[0] ^= load_bytes(ad, (int)adlen);
            s.x[0] ^= PAD((int)adlen);
        }
        P8(&s);
    }

    /* Domain separation. */
    s.x[4] ^= DSEP();

    /* Plaintext. */
    while (mlen >= ASCON_128A_RATE) {
        s.x[0] ^= load_bytes(m, 8);
        s.x[1] ^= load_bytes(m + 8, 8);
        store_bytes(c, s.x[0], 8);
        store_bytes(c + 8, s.x[1], 8);

        P8(&s);
        m += ASCON_128A_RATE;
        c += ASCON_128A_RATE;
        mlen -= ASCON_128A_RATE;
    }

    if (mlen >= 8) {
        s.x[0] ^= load_bytes(m, 8);
        s.x[1] ^= load_bytes(m + 8, (int)(mlen - 8));
        store_bytes(c, s.x[0], 8);
        store_bytes(c + 8, s.x[1], (int)(mlen - 8));
        s.x[1] ^= PAD((int)(mlen - 8));
    } else {
        s.x[0] ^= load_bytes(m, (int)mlen);
        store_bytes(c, s.x[0], (int)mlen);
        s.x[0] ^= PAD((int)mlen);
    }
    c += mlen;

    /* Finalization. */
    s.x[2] ^= K0;
    s.x[3] ^= K1;
    P12(&s);
    s.x[3] ^= K0;
    s.x[4] ^= K1;

    /* Tag. */
    store_bytes(c, s.x[3], 8);
    store_bytes(c + 8, s.x[4], 8);

    (void)c_start;
    return 0;
}

/* -------------------------------------------------------------------------
 * ASCON-AEAD-128 decryption
 * ------------------------------------------------------------------------- */

int crypto_aead_decrypt(
    unsigned char *m, unsigned long long *mlen,
    unsigned char *nsec,
    const unsigned char *c, unsigned long long clen,
    const unsigned char *ad, unsigned long long adlen,
    const unsigned char *npub,
    const unsigned char *k)
{
    ascon_state_t s;
    uint64_t K0, K1;
    uint8_t tag[CRYPTO_ABYTES];
    unsigned long long text_len;

    (void)nsec;

    if (clen < CRYPTO_ABYTES) {
        return -1;
    }

    text_len = clen - CRYPTO_ABYTES;
    *mlen = text_len;

    ascon_initialize(&s, npub, k, &K0, &K1);

    /* Associated data. */
    if (adlen) {
        while (adlen >= ASCON_128A_RATE) {
            s.x[0] ^= load_bytes(ad, 8);
            s.x[1] ^= load_bytes(ad + 8, 8);
            P8(&s);
            ad += ASCON_128A_RATE;
            adlen -= ASCON_128A_RATE;
        }

        if (adlen >= 8) {
            s.x[0] ^= load_bytes(ad, 8);
            s.x[1] ^= load_bytes(ad + 8, (int)(adlen - 8));
            s.x[1] ^= PAD((int)(adlen - 8));
        } else {
            s.x[0] ^= load_bytes(ad, (int)adlen);
            s.x[0] ^= PAD((int)adlen);
        }
        P8(&s);
    }

    /* Domain separation. */
    s.x[4] ^= DSEP();

    clen = text_len;

    /* Ciphertext. */
    while (clen >= ASCON_128A_RATE) {
        uint64_t c0 = load_bytes(c, 8);
        uint64_t c1 = load_bytes(c + 8, 8);

        store_bytes(m, s.x[0] ^ c0, 8);
        store_bytes(m + 8, s.x[1] ^ c1, 8);
        s.x[0] = c0;
        s.x[1] = c1;

        P8(&s);
        m += ASCON_128A_RATE;
        c += ASCON_128A_RATE;
        clen -= ASCON_128A_RATE;
    }

    if (clen >= 8) {
        uint64_t c0 = load_bytes(c, 8);
        uint64_t c1 = load_bytes(c + 8, (int)(clen - 8));

        store_bytes(m, s.x[0] ^ c0, 8);
        store_bytes(m + 8, s.x[1] ^ c1, (int)(clen - 8));
        s.x[0] = c0;
        s.x[1] = clear_bytes(s.x[1], (int)(clen - 8));
        s.x[1] |= c1;
        s.x[1] ^= PAD((int)(clen - 8));
    } else {
        uint64_t c0 = load_bytes(c, (int)clen);

        store_bytes(m, s.x[0] ^ c0, (int)clen);
        s.x[0] = clear_bytes(s.x[0], (int)clen);
        s.x[0] |= c0;
        s.x[0] ^= PAD((int)clen);
    }
    c += clen;

    /* Finalization. */
    s.x[2] ^= K0;
    s.x[3] ^= K1;
    P12(&s);
    s.x[3] ^= K0;
    s.x[4] ^= K1;

    store_bytes(tag, s.x[3], 8);
    store_bytes(tag + 8, s.x[4], 8);

    if (ascon_verify_tag(c, tag, CRYPTO_ABYTES) != 0) {
        *mlen = 0;
        return -1;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Key-byte helper for ChipWhisperer tight-trigger collection
 * -------------------------------------------------------------------------
 * This function is for your dataset collection firmware.
 *
 * It follows the ASCON initialization path:
 *   S = IV || K || N
 *   P12(S)
 *   S[3] ^= K0
 *   S[4] ^= K1
 *
 * For byte-wise profiling, the second key-injection step is performed one byte
 * at a time. If ASCON_ENABLE_KEYBYTE_TRIGGER is defined, the trigger is raised
 * only around the selected byte injection.
 * ------------------------------------------------------------------------- */

uint8_t ascon_keybyte_absorb_tighttrigger(const uint8_t *key,
                                          const uint8_t *nonce,
                                          uint8_t target_byte)
{
    ascon_state_t s;
    uint64_t K0, K1, N0, N1;
    uint8_t i;

    if (target_byte >= CRYPTO_KEYBYTES) {
        return 0xff;
    }

    K0 = load_bytes(key, 8);
    K1 = load_bytes(key + 8, 8);
    N0 = load_bytes(nonce, 8);
    N1 = load_bytes(nonce + 8, 8);

    s.x[0] = ASCON_128A_IV;
    s.x[1] = K0;
    s.x[2] = K1;
    s.x[3] = N0;
    s.x[4] = N1;

    /* Real ASCON initialization permutation. */
    P12(&s);

    /* Second key injection, byte by byte, with optional tight trigger. */
    for (i = 0; i < CRYPTO_KEYBYTES; ++i) {
        uint8_t kb = key[i];

        if (i == target_byte) {
            ASCON_TRIGGER_HIGH();
        }

        if (i < 8) {
            s.x[3] ^= SETBYTE(kb, i);
        } else {
            s.x[4] ^= SETBYTE(kb, i - 8);
        }

        if (i == target_byte) {
            ASCON_TRIGGER_LOW();
        }
    }

    return ascon_state_checksum(&s) ^ key[target_byte];
}
