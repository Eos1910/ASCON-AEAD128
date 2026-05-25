#include <stdint.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"
#include "api.h"

/*
 * ASCON-AEAD-128 tight-trigger firmware for ChipWhisperer-Lite XMEGA.
 *
 * SimpleSerial commands:
 *
 *   k <16 bytes> : load 128-bit ASCON key
 *   n <16 bytes> : load 128-bit ASCON nonce
 *   p <1 byte>   : run tight-trigger key-byte operation for target byte 0 to 15
 *   e <16 bytes> : optional full ASCON encryption sanity test
 *
 * Dataset collection should use:
 *
 *   k = random key
 *   n = fixed or random nonce
 *   p = target byte index
 *
 * The p command returns 1 byte only.
 * That 1 byte is not the key. It is a checksum/status byte used to keep the
 * computation alive and verify that the target returned something.
 */

#define KEY_LEN CRYPTO_KEYBYTES
#define NONCE_LEN CRYPTO_NPUBBYTES

#define TEST_PLAINTEXT_LEN 16
#define TEST_CIPHERTEXT_LEN (TEST_PLAINTEXT_LEN + CRYPTO_ABYTES)

static uint8_t key[KEY_LEN];
static uint8_t nonce[NONCE_LEN];

/*
 * Command: k
 * Load 16-byte ASCON key.
 */
uint8_t get_key(uint8_t *data, uint8_t len)
{
    if (len != KEY_LEN) {
        return 0x01;
    }

    memcpy(key, data, KEY_LEN);
    return 0x00;
}

/*
 * Command: n
 * Load 16-byte ASCON nonce.
 */
uint8_t get_nonce(uint8_t *data, uint8_t len)
{
    if (len != NONCE_LEN) {
        return 0x02;
    }

    memcpy(nonce, data, NONCE_LEN);
    return 0x00;
}

/*
 * Command: p
 * Run selected key-byte tight-trigger operation.
 *
 * data[0] = target byte, from 0 to 15.
 *
 * The actual trigger should happen inside encrypt.c, inside
 * ascon_keybyte_absorb_tighttrigger(), if encrypt.c was compiled with:
 *
 *   -DASCON_ENABLE_KEYBYTE_TRIGGER
 */
uint8_t run_keybyte_trigger(uint8_t *data, uint8_t len)
{
    uint8_t target_byte;
    uint8_t result;

    if (len != 1) {
        return 0x03;
    }

    target_byte = data[0];

    if (target_byte >= KEY_LEN) {
        return 0x04;
    }

    result = ascon_keybyte_absorb_tighttrigger(key, nonce, target_byte);

    /*
     * Return one byte so Python knows the operation completed.
     * This is not the recovered key.
     */
    simpleserial_put('r', 1, &result);

    return 0x00;
}

/*
 * Command: e
 * Optional full ASCON-AEAD-128 encryption sanity test.
 *
 * Input:
 *   16-byte plaintext
 *
 * Output:
 *   32 bytes = 16-byte ciphertext + 16-byte tag
 *
 * This command is useful to confirm that encrypt.c is working.
 * It is not the main dataset collection command.
 */
uint8_t run_full_encrypt(uint8_t *data, uint8_t len)
{
    uint8_t ciphertext[TEST_CIPHERTEXT_LEN];
    unsigned long long clen = 0;

    if (len != TEST_PLAINTEXT_LEN) {
        return 0x05;
    }

    memset(ciphertext, 0, sizeof(ciphertext));

    /*
     * Trigger around full encryption only for sanity testing.
     * For tight-trigger dataset collection, use command p instead.
     */
    trigger_high();

    crypto_aead_encrypt(
        ciphertext,
        &clen,
        data,
        TEST_PLAINTEXT_LEN,
        NULL,
        0,
        NULL,
        nonce,
        key
    );

    trigger_low();

    if (clen != TEST_CIPHERTEXT_LEN) {
        return 0x06;
    }

    simpleserial_put('r', TEST_CIPHERTEXT_LEN, ciphertext);

    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    memset(key, 0, sizeof(key));
    memset(nonce, 0, sizeof(nonce));

    simpleserial_init();

    /*
     * Main dataset collection commands.
     */
    simpleserial_addcmd('k', KEY_LEN, get_key);
    simpleserial_addcmd('n', NONCE_LEN, get_nonce);
    simpleserial_addcmd('p', 1, run_keybyte_trigger);

    /*
     * Optional sanity test command.
     */
    simpleserial_addcmd('e', TEST_PLAINTEXT_LEN, run_full_encrypt);

    while (1) {
        simpleserial_get();
    }

    return 0;
}