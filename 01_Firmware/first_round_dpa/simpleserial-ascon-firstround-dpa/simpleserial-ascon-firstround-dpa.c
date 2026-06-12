#include <stdint.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"
#include "api.h"

/*
 * ASCON first-round DPA/CPA firmware for ChipWhisperer-Lite XMEGA.
 *
 * SimpleSerial commands:
 *
 *   k <16 bytes> : load fixed ASCON 128-bit key
 *   n <16 bytes> : load random ASCON 128-bit nonce
 *   p <0 bytes>  : run first ASCON initialization round capture
 *
 * This firmware is for the traditional DPA/CPA baseline:
 *   fixed key + random nonce + first initialization round trigger
 */

#define KEY_LEN CRYPTO_KEYBYTES
#define NONCE_LEN CRYPTO_NPUBBYTES

static uint8_t key[KEY_LEN];
static uint8_t nonce[NONCE_LEN];

uint8_t get_key(uint8_t *data, uint8_t len)
{
    if (len != KEY_LEN) {
        return 0x01;
    }

    memcpy(key, data, KEY_LEN);
    return 0x00;
}

uint8_t get_nonce(uint8_t *data, uint8_t len)
{
    if (len != NONCE_LEN) {
        return 0x02;
    }

    memcpy(nonce, data, NONCE_LEN);
    return 0x00;
}

uint8_t run_first_round_capture(uint8_t *data, uint8_t len)
{
    uint8_t result;

    (void)data;

    /*
     * p command should have no payload.
     */
    if (len != 0) {
        return 0x03;
    }

    result = ascon_first_round_dpa_capture(key, nonce);

    simpleserial_put('r', 1, &result);

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

    simpleserial_addcmd('k', KEY_LEN, get_key);
    simpleserial_addcmd('n', NONCE_LEN, get_nonce);
    simpleserial_addcmd('p', 0, run_first_round_capture);

    while (1) {
        simpleserial_get();
    }

    return 0;
}