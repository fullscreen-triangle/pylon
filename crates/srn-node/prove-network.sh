#!/usr/bin/env bash
# prove-network.sh
#
# Demonstrates two properties on a local machine using two node processes:
#
#   Property 1 — PRIVATE NETWORK
#     Two nodes can communicate; /network/probe returns reachable=true for
#     each peer and reports the peer's partition coordinates.
#
#   Property 2 — INFORMATION PROPAGATION
#     A glyph emitted from node A causes node B to evaluate it in B's own
#     receiver frame and write the result to B's Γ registry.
#     The laptop (node A) then fetches that result from node B's /fetch endpoint.
#     Gossip propagates a binding to all peers simultaneously.

set -e

BINARY="./target/release/srn-node"
if [ ! -f "$BINARY" ]; then
  echo "Building release binary..."
  cargo build --release --quiet
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Sango Rine Shumba — Network Proof               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Start two nodes on localhost ─────────────────────────────────────────────

echo "▶  Starting Node A  (2,1,0,+)  on :7700  — represents this laptop"
"$BINARY" --n 2 --l 1 --m 0 --s 1 --bind 0.0.0.0:7700 2>/dev/null &
NODE_A=$!

echo "▶  Starting Node B  (1,0,0,+)  on :7701  — represents the Chromebook"
"$BINARY" --n 1 --l 0 --m 0 --s 1 --bind 0.0.0.0:7701 2>/dev/null &
NODE_B=$!

sleep 1  # wait for both to be ready

BASE_A="http://localhost:7700"
BASE_B="http://localhost:7701"

cleanup() {
  kill $NODE_A $NODE_B 2>/dev/null
}
trap cleanup EXIT

# ── Register peers with each other ──────────────────────────────────────────

echo ""
echo "── Registering peers ────────────────────────────────────────────────────"

curl -sf -X POST "$BASE_A/peers" \
  -H 'Content-Type: application/json' \
  -d '{"name":"chromebook","addr":"localhost:7701","n":1,"l":0,"m":0,"s":1}' \
  | python3 -m json.tool

curl -sf -X POST "$BASE_B/peers" \
  -H 'Content-Type: application/json' \
  -d '{"name":"laptop","addr":"localhost:7700","n":2,"l":1,"m":0,"s":1}' \
  | python3 -m json.tool

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  PROPERTY 1: PRIVATE NETWORK"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Node A probes all peers.  Expected: chromebook reachable=true"
echo ""

PROBE=$(curl -sf "$BASE_A/network/probe")
echo "$PROBE" | python3 -m json.tool

REACHABLE=$(echo "$PROBE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['peers_reachable'])")
TOTAL=$(echo "$PROBE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['peers_probed'])")
HEALTHY=$(echo "$PROBE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['network_healthy'])")

echo ""
if [ "$HEALTHY" = "True" ]; then
  echo "  ✓ PROPERTY 1 PROVED: all $REACHABLE/$TOTAL peers reachable"
else
  echo "  ✗ PROPERTY 1 FAILED: only $REACHABLE/$TOTAL peers reachable"
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  PROPERTY 2A: INFORMATION PROPAGATION — emit (delegate)"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Node A emits a glyph to Node B (chromebook)."
echo "  Node B evaluates it in its own receiver frame (n=1,l=0,m=0,s=+)."
echo "  Node A then fetches the result from Node B's registry."
echo ""

EMIT=$(curl -sf -X POST "$BASE_A/network/emit" \
  -H 'Content-Type: application/json' \
  -d '{
    "peer_name": "chromebook",
    "name": "ring-phone",
    "target_n": 1, "target_l": 0, "target_m": 0, "target_s": 1,
    "not_guard": "do-not-ring",
    "body": "ring device at self-coords",
    "to_target": "return"
  }')

echo "Emit response from chromebook:"
echo "$EMIT" | python3 -m json.tool

# Fetch B's registry to see the result recorded there
echo ""
echo "Fetching Node B's Γ registry for its own coords (1,0,0,+):"
FETCH_B=$(curl -sf "$BASE_B/fetch/1/0/0/1" || echo '{"error":"not found"}')
echo "$FETCH_B" | python3 -m json.tool

ACCEPTED=$(echo "$EMIT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('accepted',False))" 2>/dev/null || echo "False")

echo ""
if [ "$ACCEPTED" = "True" ]; then
  echo "  ✓ PROPERTY 2A PROVED: glyph delegated to chromebook and evaluated there"
else
  echo "  ✗ PROPERTY 2A: chromebook did not accept the glyph"
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  PROPERTY 2B: INFORMATION PROPAGATION — gossip"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Node A gossips 'caller-id: +263771000000' to all peers."
echo "  Node B receives it and its environment now contains the binding."
echo "  A subsequent glyph from Node A with not_guard='caller-id' can"
echo "  be rejected by Node B (because it now knows the value)."
echo ""

GOSSIP=$(curl -sf -X POST "$BASE_A/network/gossip" \
  -H 'Content-Type: application/json' \
  -d '{"key":"caller-id","value":"+263771000000"}')

echo "Gossip result:"
echo "$GOSSIP" | python3 -m json.tool

DELIVERED=$(echo "$GOSSIP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['peers_delivered'])" 2>/dev/null || echo "0")

echo ""
if [ "$DELIVERED" -ge 1 ] 2>/dev/null; then
  echo "  ✓ PROPERTY 2B PROVED: binding propagated to $DELIVERED peer(s)"
else
  echo "  ✗ PROPERTY 2B: gossip delivery failed"
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  PROPERTY 2C: INFORMATION PROPAGATION — fetch-remote"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Node A evaluates a glyph locally (phone-only sensor data simulation)."
echo "  Node B fetches that result from Node A's registry."
echo ""

# First: A evaluates something locally that only A would know
curl -sf -X POST "$BASE_A/eval" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "battery-level",
    "target_n": 2, "target_l": 1, "target_m": 0, "target_s": 1,
    "not_guard": "battery-blocked",
    "body": "read local battery sensor",
    "to_target": "self"
  }' > /dev/null

echo "Node B fetches Node A's (laptop) latest registry entry remotely:"
REMOTE=$(curl -sf "$BASE_B/network/fetch-remote/laptop/2/1/0/1" || echo '{"error":"not found"}')
echo "$REMOTE" | python3 -m json.tool

HAS_RESULT=$(echo "$REMOTE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('error' not in d)" 2>/dev/null || echo "False")

echo ""
if [ "$HAS_RESULT" = "True" ]; then
  echo "  ✓ PROPERTY 2C PROVED: Node B fetched Node A's local registry entry"
else
  echo "  ✗ PROPERTY 2C: remote fetch returned nothing"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  Final status of both nodes"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Node A (laptop, 2,1,0,+):"
curl -sf "$BASE_A/status" | python3 -m json.tool
echo ""
echo "Node B (chromebook, 1,0,0,+):"
curl -sf "$BASE_B/status" | python3 -m json.tool
