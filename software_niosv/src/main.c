/*
 * Nios V geometry telemetry example.
 * This polls the geometry IP only; no CNN class decision is implemented yet.
 */
#include <stdio.h>
#include <stdint.h>
#include "system.h"
#include "io.h"

#ifndef LAUNDRY_GEOM_IP_0_BASE
#error "Define LAUNDRY_GEOM_IP_0_BASE in the generated Platform Designer system.h"
#endif

#define REG_STATUS       0u
#define REG_AREA         1u
#define REG_BBOX         2u
#define REG_TWO_RUN_ROWS 3u
#define REG_WH           4u
#define STATUS_DONE      (1u << 2)

int main(void)
{
    uint32_t done_reported = 0u;
    puts("Laundry Shape Sorter - Cyclone V / Nios V geometry monitor");

    for (;;) {
        uint32_t status = IORD_32DIRECT(LAUNDRY_GEOM_IP_0_BASE, REG_STATUS * 4u);
        if ((status & STATUS_DONE) == 0u) {
            done_reported = 0u;
            continue;
        }
        if (done_reported != 0u) {
            continue;
        }
        done_reported = 1u;

        uint32_t area = IORD_32DIRECT(LAUNDRY_GEOM_IP_0_BASE, REG_AREA * 4u) & 0x1FFFu;
        uint32_t bbox = IORD_32DIRECT(LAUNDRY_GEOM_IP_0_BASE, REG_BBOX * 4u);
        uint32_t runs = IORD_32DIRECT(LAUNDRY_GEOM_IP_0_BASE, REG_TWO_RUN_ROWS * 4u) & 0xFFu;
        uint32_t wh = IORD_32DIRECT(LAUNDRY_GEOM_IP_0_BASE, REG_WH * 4u);

        printf("FRAME DONE area=%lu bbox=(%lu,%lu)-(%lu,%lu) W=%lu H=%lu two-run=%lu\n",
            (unsigned long)area,
            (unsigned long)(bbox & 0xFFu),
            (unsigned long)((bbox >> 16) & 0xFFu),
            (unsigned long)((bbox >> 8) & 0xFFu),
            (unsigned long)((bbox >> 24) & 0xFFu),
            (unsigned long)(wh & 0xFFu),
            (unsigned long)((wh >> 8) & 0xFFu),
            (unsigned long)runs);
        puts("Geometry telemetry only; CNN classification is not connected yet.");
    }
    return 0;
}
