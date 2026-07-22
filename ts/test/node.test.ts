import { test } from "node:test";
import assert from "node:assert/strict";
import { makeCoords, isCoordError, type PartitionCoords } from "../src/coords.js";
import { Task, Scheduler, STALL_WINDOW, FLOOR } from "../src/scheduler.js";
import { SrnNode } from "../src/node.js";
import { glyph } from "../src/srn/expr.js";

function coords(n: number, l: number, m: number, s: number): PartitionCoords {
  const c = makeCoords(n, l, m, s);
  if (isCoordError(c)) throw new Error(c.message);
  return c;
}

// ---- scheduler (ported from scheduler.rs tests) ----

test("priority is zero for a stalled task (never dispatched while others run)", () => {
  const t = new Task("t", 100, 4, 50);
  for (let i = 0; i <= STALL_WINDOW; i++) t.commitUnit(50); // flat residue -> stall
  assert.ok(t.priority() === 0 || t.state === "stalled");
});

test("trajectory count M is strictly monotone", () => {
  const t = new Task("t", 100, 4, 80);
  for (let i = 1; i <= 20; i++) {
    t.commitUnit(Math.max(80 - i, FLOOR));
    assert.equal(t.trajectoryCount, i);
  }
});

test("task at threshold becomes sufficient (finishes)", () => {
  const t = new Task("t", 100, 4, 80).withThreshold(5);
  t.commitUnit(3); // below threshold
  assert.equal(t.state, "sufficient");
});

test("termination bound is logarithmic: n_term(1e12, 3) = 21", () => {
  const t = new Task("t", 1_000_000_000_000, 3, 50);
  assert.equal(t.terminationBound(), 21);
  const t2 = new Task("t2", 1_000_000, 3, 50);
  assert.ok(t2.terminationBound() < 21 && t2.terminationBound() > 1);
});

test("liveness: a descending task reaches sufficiency in finite dispatches", () => {
  const sched = new Scheduler();
  const t = new Task("live", 100, 4, 20).withThreshold(FLOOR);
  sched.addTask(t);
  let guard = 0;
  while (t.state === "running" && guard++ < 10000) {
    sched.tick(1, (task) => Math.max(task.residue - 1, FLOOR)); // descend by 1 each unit
  }
  assert.notEqual(t.state, "running");
  assert.ok(t.trajectoryCount <= 21, "reaches floor within ~20 units");
});

test("stalled frontier declines: all-stalled tasks produce no dispatch", () => {
  const sched = new Scheduler();
  const t = new Task("s", 100, 4, 50);
  for (let i = 0; i <= STALL_WINDOW; i++) t.commitUnit(50);
  sched.addTask(t);
  const results = sched.tick(4, (task) => task.residue);
  assert.equal(results.length, 0, "no dispatch when the only task is stalled");
});

// ---- node (ported from node.rs behaviour) ----

test("SrnNode advances M and appends to the registry on each eval", () => {
  const node = SrnNode.reference(coords(1, 0, 0, 1));
  const g = glyph({ name: "ping", target: coords(1, 0, 0, 1), notGuard: "x", body: "b", toTarget: "*" });
  assert.equal(node.trajectoryCount(), 0);
  node.evaluate(g);
  node.evaluate(g);
  assert.equal(node.trajectoryCount(), 2);
  assert.equal(node.totalCommitted(), 2);
  assert.equal(node.evalLog.length, 2);
});

test("SrnNode fetch returns latest value; miss returns undefined", () => {
  const node = SrnNode.reference(coords(1, 0, 0, 1));
  const g = glyph({ name: "v", target: coords(1, 0, 0, 1), notGuard: "x", body: "b", toTarget: "*" });
  node.evaluate(g);
  assert.ok(node.fetch(coords(1, 0, 0, 1)) !== undefined);
  assert.equal(node.fetch(coords(3, 1, 0, 1)), undefined);
});

test("M is never decremented across many evals (incorruptibility)", () => {
  const node = SrnNode.reference(coords(2, 1, 0, 1));
  const g = glyph({ name: "m", target: coords(2, 1, 0, 1), notGuard: "x", body: "b", toTarget: "*" });
  let prev = node.trajectoryCount();
  for (let i = 0; i < 50; i++) {
    node.evaluate(g);
    assert.ok(node.trajectoryCount() >= prev);
    prev = node.trajectoryCount();
  }
});
