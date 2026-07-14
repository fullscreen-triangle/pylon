#!/usr/bin/env python3
"""
TR-064 collector — polls both FritzBoxes and exposes Prometheus metrics on :9101.

Reads credentials from environment variables (set in .env.local or export them):
  FRITZ_HOST         IP of main router   (default: 192.168.178.1)
  FRITZ_PASSWORD     admin password for main router
  FRITZ_HOST_2       IP of repeater      (default: 192.168.178.2)
  FRITZ_PASSWORD_2   admin password for repeater (often same as main)
"""

import os
import time
import logging
from dotenv import load_dotenv
from fritzconnection.lib.fritzstatus import FritzStatus
from fritzconnection.lib.fritzhosts import FritzHosts
from prometheus_client import start_http_server, Gauge, Counter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fritz_collector")

FRITZ_HOST      = os.getenv("FRITZ_HOST", "192.168.178.1")
FRITZ_PASSWORD  = os.getenv("FRITZ_PASSWORD", "")
FRITZ_HOST_2    = os.getenv("FRITZ_HOST_2", "192.168.178.2")
FRITZ_PASSWORD_2 = os.getenv("FRITZ_PASSWORD_2", FRITZ_PASSWORD)

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
PORT          = int(os.getenv("COLLECTOR_PORT", "9101"))

# ── Gauges ────────────────────────────────────────────────────────────────────

upstream_bps = Gauge("fritz_upstream_bps",
    "Current upstream bit rate", ["host"])
downstream_bps = Gauge("fritz_downstream_bps",
    "Current downstream bit rate", ["host"])
upstream_max_bps = Gauge("fritz_upstream_max_bps",
    "Max upstream bit rate", ["host"])
downstream_max_bps = Gauge("fritz_downstream_max_bps",
    "Max downstream bit rate", ["host"])

bytes_sent = Counter("fritz_bytes_sent_total",
    "Total bytes sent", ["host"])
bytes_received = Counter("fritz_bytes_received_total",
    "Total bytes received", ["host"])

connected_devices = Gauge("fritz_connected_devices",
    "Number of active devices on the network", ["host"])

uptime_seconds = Gauge("fritz_uptime_seconds",
    "Router uptime in seconds", ["host"])

is_connected = Gauge("fritz_wan_connected",
    "1 if WAN link is up, 0 otherwise", ["host"])

# ── Collection ────────────────────────────────────────────────────────────────

def collect_status(host: str, password: str, label: str):
    try:
        fc = FritzStatus(address=host, password=password)

        upstream_bps.labels(host=label).set(fc.transmission_rate[0])
        downstream_bps.labels(host=label).set(fc.transmission_rate[1])
        upstream_max_bps.labels(host=label).set(fc.max_bit_rate[0])
        downstream_max_bps.labels(host=label).set(fc.max_bit_rate[1])

        sent = fc.bytes_sent
        recv = fc.bytes_received
        # Counter.inc only — use _value.set for absolute counters from router
        bytes_sent.labels(host=label)._value.set(sent)
        bytes_received.labels(host=label)._value.set(recv)

        uptime_seconds.labels(host=label).set(fc.uptime)
        is_connected.labels(host=label).set(1 if fc.is_connected else 0)

        log.info(f"[{label}] up={fc.transmission_rate[0]/1000:.0f}kbps "
                 f"down={fc.transmission_rate[1]/1000:.0f}kbps "
                 f"uptime={fc.uptime}s")
    except Exception as e:
        log.warning(f"[{label}] status poll failed: {e}")
        is_connected.labels(host=label).set(0)


def collect_hosts(host: str, password: str, label: str):
    try:
        fh = FritzHosts(address=host, password=password)
        active = sum(1 for d in fh.get_hosts_info() if d.get("status"))
        connected_devices.labels(host=label).set(active)
        log.info(f"[{label}] active devices: {active}")
    except Exception as e:
        log.warning(f"[{label}] hosts poll failed: {e}")


def poll():
    collect_status(FRITZ_HOST,   FRITZ_PASSWORD,  "fritz-main")
    collect_hosts( FRITZ_HOST,   FRITZ_PASSWORD,  "fritz-main")
    collect_status(FRITZ_HOST_2, FRITZ_PASSWORD_2, "fritz-repeater")


if __name__ == "__main__":
    log.info(f"Starting Fritz collector on :{PORT}, polling every {POLL_INTERVAL}s")
    start_http_server(PORT)
    while True:
        poll()
        time.sleep(POLL_INTERVAL)
