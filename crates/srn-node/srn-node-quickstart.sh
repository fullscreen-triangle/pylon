#!/usr/bin/env bash
# SRN node quick-start — run this from crates/srn-node/ on the Chromebook
# (or any Linux machine with Rust + cargo installed)

set -e

echo "=== Building srn-node ==="
cargo build --release

echo ""
echo "=== Starting node at (1,0,0,+1) on :7700 ==="
./target/release/srn-node --n 1 --l 0 --m 0 --s 1 --bind 0.0.0.0:7700 &
NODE_PID=$!
sleep 1

echo ""
echo "=== Status ==="
curl -s http://localhost:7700/status | python3 -m json.tool

echo ""
echo "=== Evaluate a glyph expression ==="
curl -s -X POST http://localhost:7700/eval \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "hello-forest",
    "target_n": 1, "target_l": 0, "target_m": 0, "target_s": 1,
    "not_guard": "blocked",
    "body": "echo self",
    "to_target": "return"
  }' | python3 -m json.tool

echo ""
echo "=== Install a binding ==="
curl -s -X POST http://localhost:7700/install \
  -H 'Content-Type: application/json' \
  -d '{"key": "blocked", "value": true}' | python3 -m json.tool

echo ""
echo "=== Evaluate again — guard fires this time ==="
curl -s -X POST http://localhost:7700/eval \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "hello-forest",
    "target_n": 1, "target_l": 0, "target_m": 0, "target_s": 1,
    "not_guard": "blocked",
    "body": "echo self",
    "to_target": "return"
  }' | python3 -m json.tool

echo ""
echo "=== Schedule a task (residue-driven scheduler) ==="
curl -s -X POST http://localhost:7700/schedule \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-1",
    "complexity": 64,
    "op_types": 4,
    "initial_residue": 80.0,
    "threshold": 5.0,
    "descent_per_tick": 10.0
  }' | python3 -m json.tool

echo ""
echo "=== Status after scheduler run ==="
curl -s http://localhost:7700/status | python3 -m json.tool

echo ""
echo "Stopping node..."
kill $NODE_PID
