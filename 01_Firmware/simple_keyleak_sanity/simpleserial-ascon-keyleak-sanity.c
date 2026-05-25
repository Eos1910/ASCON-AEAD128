#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <string.h>

static uint8_t key[16];

uint8_t get_key(uint8_t *data, uint8_t len) {
    if (len != 16) return 1;
    memcpy(key, data, 16);
    return 0;
}

uint8_t run_leak(uint8_t *data, uint8_t len) {
    if (len != 1) return 1;
    uint8_t idx = data[0] & 0x0F;
    volatile uint8_t x = 0;
    trigger_high();
    for (volatile uint16_t i = 0; i < 200; i++) {
        x ^= key[idx];
    }
    trigger_low();
    simpleserial_put('r', 1, (uint8_t *)&x);
    return 0;
}

int main(void) {
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();

    memset(key, 0, sizeof(key));
    simpleserial_addcmd('k', 16, get_key);
    simpleserial_addcmd('p', 1, run_leak);

    while (1) {
        simpleserial_get();
    }
}
