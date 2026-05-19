import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";
import { useEffect, useRef, useState, useCallback } from "react";

// ─── WGSL: Agent struct (32 bytes, 16-byte aligned) ─────────────────────────
const AGENT_STRUCT = /* wgsl */`
struct Agent {
  sk: f32, st: f32, se: f32, phase: f32,
  omega: f32, _p0: f32, _p1: f32, _p2: f32,
}`;

// ─── WGSL: Kuramoto phase update ─────────────────────────────────────────────
const KURAMOTO_WGSL = /* wgsl */`
${AGENT_STRUCT}
struct Params { n: u32, dt: f32, K: f32, t: f32 }

@group(0) @binding(0) var<storage, read>       src:    array<Agent>;
@group(0) @binding(1) var<storage, read_write> dst:    array<Agent>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let ai = src[i];
  let si = vec3f(ai.sk, ai.st, ai.se);
  var sum = 0.0;
  for (var j = 0u; j < params.n; j++) {
    if (j == i) { continue; }
    let aj = src[j];
    let sj = vec3f(aj.sk, aj.st, aj.se);
    let num = dot(si, sj);
    let den = length(si) * length(sj) + 1e-6;
    let w   = (num / den + 1.0) * 0.5;
    sum += w * sin(aj.phase - ai.phase);
  }
  let dphase = ai.omega + (params.K / f32(params.n - 1u)) * sum;
  dst[i] = Agent(ai.sk, ai.st, ai.se,
                 ai.phase + params.dt * dphase,
                 ai.omega, 0.0, 0.0, 0.0);
}`;

// ─── WGSL: SpectralDP similarity matrix ─────────────────────────────────────
const SPECTRALDP_WGSL = /* wgsl */`
${AGENT_STRUCT}

@group(0) @binding(0) var<storage, read>       agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> sim:    array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = arrayLength(&agents);
  let i = gid.x; let j = gid.y;
  if (i >= n || j >= n) { return; }
  let si = vec3f(agents[i].sk, agents[i].st, agents[i].se);
  let sj = vec3f(agents[j].sk, agents[j].st, agents[j].se);
  let cosine = dot(si, sj) / (length(si) * length(sj) + 1e-6);
  sim[i * n + j] = (cosine + 1.0) * 0.5;
}`;

// ─── WGSL: Kuramoto order parameter R_ens ────────────────────────────────────
const RENS_WGSL = /* wgsl */`
${AGENT_STRUCT}
struct Result { r_cos: f32, r_sin: f32, r_ens: f32, _pad: f32 }

@group(0) @binding(0) var<storage, read>       agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> result: Result;

@compute @workgroup_size(1)
fn main() {
  let n = arrayLength(&agents);
  var c = 0.0; var s = 0.0;
  for (var i = 0u; i < n; i++) {
    c += cos(agents[i].phase);
    s += sin(agents[i].phase);
  }
  let fn_ = f32(n);
  result.r_cos = c / fn_;
  result.r_sin = s / fn_;
  result.r_ens = sqrt(c * c + s * s) / fn_;
}`;

const N_AGENTS = 8;
const AGENT_STRIDE = 8; // f32s per agent

function coordRegime(r) {
  if (r < 0.30) return { name: "Turbulent",      color: "#ef4444", friction: 1.00 };
  if (r < 0.50) return { name: "Aperture",        color: "#f97316", friction: 0.72 };
  if (r < 0.80) return { name: "Cascade",         color: "#eab308", friction: 0.40 };
  if (r < 0.95) return { name: "Coherent",        color: "#22c55e", friction: 0.12 };
  return              { name: "Phase-Locked ✦",   color: "#58E6D9", friction: 0.00 };
}

// ─── CPU fallback simulation ─────────────────────────────────────────────────
function cpuStep(agents, K, dt) {
  const n = agents.length;
  const next = agents.map((ai) => {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (j === agents.indexOf(ai)) continue;
      const aj = agents[j];
      const dot = ai.sk * aj.sk + ai.st * aj.st + ai.se * aj.se;
      const lenA = Math.hypot(ai.sk, ai.st, ai.se) + 1e-6;
      const lenB = Math.hypot(aj.sk, aj.st, aj.se) + 1e-6;
      const w = (dot / (lenA * lenB) + 1) * 0.5;
      sum += w * Math.sin(aj.phase - ai.phase);
    }
    const dphase = ai.omega + (K / (n - 1)) * sum;
    return { ...ai, phase: ai.phase + dt * dphase };
  });
  return next;
}

function cpuSpectralDP(agents) {
  const n = agents.length;
  const mat = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    const ai = agents[i];
    for (let j = 0; j < n; j++) {
      const aj = agents[j];
      const dot = ai.sk * aj.sk + ai.st * aj.st + ai.se * aj.se;
      const lenA = Math.hypot(ai.sk, ai.st, ai.se) + 1e-6;
      const lenB = Math.hypot(aj.sk, aj.st, aj.se) + 1e-6;
      mat[i * n + j] = (dot / (lenA * lenB) + 1) * 0.5;
    }
  }
  return mat;
}

function cpuRens(agents) {
  let c = 0, s = 0;
  for (const a of agents) { c += Math.cos(a.phase); s += Math.sin(a.phase); }
  return Math.hypot(c, s) / agents.length;
}

function makeInitialAgents() {
  const agents = [];
  for (let i = 0; i < N_AGENTS; i++) {
    agents.push({
      sk: 0.3 + Math.random() * 0.6,
      st: 0.3 + Math.random() * 0.6,
      se: 0.3 + Math.random() * 0.6,
      phase: Math.random() * 2 * Math.PI,
      omega: (Math.random() - 0.5) * 2.0,
    });
  }
  return agents;
}

// ─── Canvas draw ─────────────────────────────────────────────────────────────
function drawScene(canvas, agents, simMatrix, rEns, history, darkMode) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const bg = darkMode ? "#1b1b1b" : "#f5f5f0";
  const fg = darkMode ? "#e8e8e8" : "#1b1b1b";
  const panel = darkMode ? "#242424" : "#ffffff";
  const n = agents.length;
  const regime = coordRegime(rEns);

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const PAD = 20;
  const HEADER = 54;
  const FOOTER = 80;
  const availH = H - HEADER - FOOTER - PAD * 2;
  const halfW = Math.floor(W / 2) - PAD;

  // ── Header bar ────────────────────────────────────────────────────────────
  ctx.fillStyle = darkMode ? "#2a2a2a" : "#f0ede8";
  ctx.beginPath();
  roundRect(ctx, PAD, PAD, W - PAD * 2, HEADER - PAD, 8);
  ctx.fill();

  ctx.fillStyle = regime.color;
  ctx.font = "bold 15px monospace";
  ctx.fillText(`Regime: ${regime.name}`, PAD + 16, PAD + 22);

  // R_ens label
  ctx.fillStyle = fg;
  ctx.font = "12px monospace";
  ctx.fillText(`R_ens`, PAD + 16, PAD + 42);

  // R_ens bar
  const barX = PAD + 70; const barW = 180; const barH = 10;
  ctx.fillStyle = darkMode ? "#3a3a3a" : "#ddd";
  ctx.beginPath();
  roundRect(ctx, barX, PAD + 34, barW, barH, 4);
  ctx.fill();
  ctx.fillStyle = regime.color;
  ctx.beginPath();
  roundRect(ctx, barX, PAD + 34, barW * rEns, barH, 4);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.font = "bold 12px monospace";
  ctx.fillText(rEns.toFixed(3), barX + barW + 8, PAD + 44);

  // Friction
  ctx.fillStyle = fg;
  ctx.font = "12px monospace";
  ctx.fillText(`Friction: ${regime.friction.toFixed(2)}`, W - PAD - 180, PAD + 22);
  ctx.fillText(`Agents: ${n}`, W - PAD - 180, PAD + 42);

  const LY = HEADER + PAD;

  // ── Left panel: Agent scatter (Sk × Se) ──────────────────────────────────
  const LW = halfW;
  const LH = availH;
  ctx.fillStyle = panel;
  ctx.beginPath();
  roundRect(ctx, PAD, LY, LW, LH, 10);
  ctx.fill();

  // Axis labels
  ctx.fillStyle = fg;
  ctx.font = "11px monospace";
  ctx.globalAlpha = 0.5;
  ctx.fillText("Sk →", PAD + 8, LY + LH - 6);
  ctx.save();
  ctx.translate(PAD + 14, LY + LH - 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Se →", 0, 0);
  ctx.restore();
  ctx.globalAlpha = 1;

  // Grid lines
  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.06;
  ctx.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    const gx = PAD + 30 + (g / 4) * (LW - 40);
    const gy = LY + 10 + (g / 4) * (LH - 20);
    ctx.beginPath(); ctx.moveTo(gx, LY + 10); ctx.lineTo(gx, LY + LH - 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD + 30, gy); ctx.lineTo(PAD + LW - 10, gy); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Connection lines between similar agents
  if (simMatrix) {
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const ai = agents[i];
      const ax = PAD + 30 + ai.sk * (LW - 40);
      const ay = LY + LH - 10 - ai.se * (LH - 20);
      for (let j = i + 1; j < n; j++) {
        const sim = simMatrix[i * n + j];
        if (sim < 0.65) continue;
        const aj = agents[j];
        const bx = PAD + 30 + aj.sk * (LW - 40);
        const by = LY + LH - 10 - aj.se * (LH - 20);
        ctx.globalAlpha = (sim - 0.65) * 2.0 * 0.4;
        ctx.strokeStyle = regime.color;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  // Agent dots
  for (let i = 0; i < n; i++) {
    const a = agents[i];
    const ax = PAD + 30 + a.sk * (LW - 40);
    const ay = LY + LH - 10 - a.se * (LH - 20);
    const hue = ((a.phase % (2 * Math.PI)) / (2 * Math.PI)) * 360;
    const r = 8 + a.st * 5;

    // Glow
    const grd = ctx.createRadialGradient(ax, ay, 0, ax, ay, r * 2);
    grd.addColorStop(0, `hsla(${hue},80%,65%,0.4)`);
    grd.addColorStop(1, `hsla(${hue},80%,65%,0)`);
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(ax, ay, r * 2, 0, Math.PI * 2); ctx.fill();

    // Circle
    ctx.fillStyle = `hsl(${hue},80%,60%)`;
    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(ax, ay, r, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    ctx.globalAlpha = 1;

    // Index label
    ctx.fillStyle = fg;
    ctx.font = "bold 10px monospace";
    ctx.fillText(String(i), ax - 3, ay + 4);
  }

  // Panel title
  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.4;
  ctx.font = "10px monospace";
  ctx.fillText("S-ENTROPY SPACE  (Sk × Se)", PAD + 8, LY + 14);
  ctx.globalAlpha = 1;

  // ── Right panel: SpectralDP similarity heatmap ────────────────────────────
  const RX = PAD + halfW + PAD;
  const RW = W - RX - PAD;
  const RH = availH;
  ctx.fillStyle = panel;
  ctx.beginPath();
  roundRect(ctx, RX, LY, RW, RH, 10);
  ctx.fill();

  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.4;
  ctx.font = "10px monospace";
  ctx.fillText("SPECTRALDP SIMILARITY", RX + 8, LY + 14);
  ctx.globalAlpha = 1;

  if (simMatrix) {
    const cellW = (RW - 24) / n;
    const cellH = (RH - 32) / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = simMatrix[i * n + j];
        const hue = v * 140; // 0=red(0) 1=green(140)
        const lum = 30 + v * 25;
        ctx.fillStyle = `hsl(${hue},70%,${lum}%)`;
        ctx.fillRect(
          RX + 12 + j * cellW,
          LY + 20 + i * cellH,
          cellW - 1,
          cellH - 1
        );
      }
    }
    // Row/col labels
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.4;
    ctx.font = "9px monospace";
    for (let i = 0; i < n; i++) {
      ctx.fillText(String(i), RX + 12 + i * cellW + cellW / 3, LY + 20 + n * cellH + 10);
      ctx.fillText(String(i), RX + 3, LY + 20 + i * cellH + cellH / 1.5);
    }
    ctx.globalAlpha = 1;
  }

  // ── Footer: R_ens history sparkline ──────────────────────────────────────
  const FY = LY + availH + PAD;
  const FW = W - PAD * 2;
  ctx.fillStyle = panel;
  ctx.beginPath();
  roundRect(ctx, PAD, FY, FW, FOOTER - PAD, 10);
  ctx.fill();

  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.4;
  ctx.font = "10px monospace";
  ctx.fillText("R_ens HISTORY", PAD + 12, FY + 14);
  ctx.globalAlpha = 1;

  if (history.length > 1) {
    const hW = FW - 100;
    const hH = FOOTER - PAD - 24;
    const hX = PAD + 12;
    const hY = FY + 18;

    // Reference lines
    [0.3, 0.5, 0.8, 0.95].forEach((threshold) => {
      const ty = hY + hH - threshold * hH;
      ctx.strokeStyle = fg;
      ctx.globalAlpha = 0.08;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(hX, ty); ctx.lineTo(hX + hW, ty); ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.globalAlpha = 1;
    ctx.lineWidth = 2;
    ctx.strokeStyle = regime.color;
    ctx.beginPath();
    history.forEach((v, idx) => {
      const x = hX + (idx / (history.length - 1)) * hW;
      const y = hY + hH - v * hH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels for thresholds
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.3;
    ctx.font = "9px monospace";
    ["0.3", "0.5", "0.8", "0.95"].forEach((t) => {
      const ty = hY + hH - parseFloat(t) * hH;
      ctx.fillText(t, hX + hW + 4, ty + 4);
    });
    ctx.globalAlpha = 1;
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ─── WebGPU init + pipeline ──────────────────────────────────────────────────
async function initWebGPU(agentData) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const agentBuf0 = device.createBuffer({
    size: N_AGENTS * AGENT_STRIDE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const agentBuf1 = device.createBuffer({
    size: N_AGENTS * AGENT_STRIDE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const simBuf = device.createBuffer({
    size: N_AGENTS * N_AGENTS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const resultBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: Math.max(N_AGENTS * AGENT_STRIDE * 4, N_AGENTS * N_AGENTS * 4, 16),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const paramBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Upload initial agent data
  const arr = new Float32Array(N_AGENTS * AGENT_STRIDE);
  agentData.forEach((a, i) => {
    arr[i * AGENT_STRIDE + 0] = a.sk;
    arr[i * AGENT_STRIDE + 1] = a.st;
    arr[i * AGENT_STRIDE + 2] = a.se;
    arr[i * AGENT_STRIDE + 3] = a.phase;
    arr[i * AGENT_STRIDE + 4] = a.omega;
  });
  device.queue.writeBuffer(agentBuf0, 0, arr);

  const kuraModule = device.createShaderModule({ code: KURAMOTO_WGSL });
  const dpModule   = device.createShaderModule({ code: SPECTRALDP_WGSL });
  const rensModule = device.createShaderModule({ code: RENS_WGSL });

  const kuraPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: kuraModule, entryPoint: "main" },
  });
  const dpPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: dpModule, entryPoint: "main" },
  });
  const rensPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: rensModule, entryPoint: "main" },
  });

  return {
    device, agentBuf0, agentBuf1, simBuf, resultBuf, readBuf, paramBuf,
    kuraPipeline, dpPipeline, rensPipeline,
    front: 0,
  };
}

function encodeGPUFrame(gpu, K, dt, t, tick) {
  const { device, paramBuf, kuraPipeline, dpPipeline, rensPipeline,
          agentBuf0, agentBuf1, simBuf, resultBuf } = gpu;

  // Update params uniform
  const params = new Float32Array([N_AGENTS, dt, K, t]);
  const paramsU32 = new Uint32Array(params.buffer);
  paramsU32[0] = N_AGENTS;
  device.queue.writeBuffer(paramBuf, 0, params);

  const srcBuf = gpu.front === 0 ? agentBuf0 : agentBuf1;
  const dstBuf = gpu.front === 0 ? agentBuf1 : agentBuf0;

  const enc = device.createCommandEncoder();

  // Pass 1: Kuramoto
  {
    const bg = device.createBindGroup({
      layout: kuraPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: dstBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(kuraPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N_AGENTS / 64));
    pass.end();
  }

  // Pass 2: SpectralDP
  {
    const bg = device.createBindGroup({
      layout: dpPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dstBuf } },
        { binding: 1, resource: { buffer: simBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(dpPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N_AGENTS / 8), Math.ceil(N_AGENTS / 8));
    pass.end();
  }

  // Pass 3: R_ens (every 5 ticks)
  if (tick % 5 === 0) {
    const bg = device.createBindGroup({
      layout: rensPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dstBuf } },
        { binding: 1, resource: { buffer: resultBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(rensPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  device.queue.submit([enc.finish()]);
  gpu.front = 1 - gpu.front;
  return { srcBuf: dstBuf, simBuf, resultBuf };
}

async function readbackGPU(gpu, buffers, tick) {
  const { device, readBuf } = gpu;
  const agentSize = N_AGENTS * AGENT_STRIDE * 4;
  const simSize   = N_AGENTS * N_AGENTS * 4;
  const rensSize  = 16;

  // Read agents
  const enc1 = device.createCommandEncoder();
  enc1.copyBufferToBuffer(buffers.srcBuf, 0, readBuf, 0, agentSize);
  device.queue.submit([enc1.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ, 0, agentSize);
  const agentArr = new Float32Array(readBuf.getMappedRange(0, agentSize).slice());
  readBuf.unmap();

  const agents = [];
  for (let i = 0; i < N_AGENTS; i++) {
    agents.push({
      sk:    agentArr[i * AGENT_STRIDE + 0],
      st:    agentArr[i * AGENT_STRIDE + 1],
      se:    agentArr[i * AGENT_STRIDE + 2],
      phase: agentArr[i * AGENT_STRIDE + 3],
      omega: agentArr[i * AGENT_STRIDE + 4],
    });
  }

  // Read similarity matrix
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(buffers.simBuf, 0, readBuf, 0, simSize);
  device.queue.submit([enc2.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ, 0, simSize);
  const simMatrix = new Float32Array(readBuf.getMappedRange(0, simSize).slice());
  readBuf.unmap();

  // Read R_ens
  let rEns = null;
  if (tick % 5 === 0) {
    const enc3 = device.createCommandEncoder();
    enc3.copyBufferToBuffer(buffers.resultBuf, 0, readBuf, 0, rensSize);
    device.queue.submit([enc3.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ, 0, rensSize);
    const rArr = new Float32Array(readBuf.getMappedRange(0, rensSize).slice());
    readBuf.unmap();
    rEns = rArr[2];
  }

  return { agents, simMatrix, rEns };
}

// ─── Stat card component ──────────────────────────────────────────────────────
const StatCard = ({ label, value, accent, sub }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="relative rounded-2xl border border-solid border-dark/20 bg-light p-5
    dark:border-light/20 dark:bg-dark"
  >
    <div className="absolute top-0 -right-2 -z-10 h-[102%] w-[101%] rounded-[1.5rem]
    rounded-br-3xl bg-dark dark:bg-light" />
    <p className="text-xs font-medium uppercase tracking-widest text-dark/40 dark:text-light/40">{label}</p>
    <p className={`mt-1 text-xl font-bold ${accent ? "text-primary dark:text-primaryDark" : "text-dark dark:text-light"}`}>
      {value}
    </p>
    {sub && <p className="mt-0.5 text-xs text-dark/40 dark:text-light/40">{sub}</p>}
  </motion.div>
);

// ─── Main page ────────────────────────────────────────────────────────────────
export default function Shaders() {
  const canvasRef = useRef(null);
  const gpuRef    = useRef(null);
  const cpuRef    = useRef({ agents: makeInitialAgents() });
  const rafRef    = useRef(null);
  const tickRef   = useRef(0);
  const kRef      = useRef(0.5);
  const histRef   = useRef([]);
  const rEnsRef   = useRef(0);

  const [gpuSupported, setGpuSupported] = useState(null);
  const [darkMode, setDarkMode]         = useState(false);
  const [K, setKState]                  = useState(0.5);
  const [liveStats, setLiveStats]       = useState({
    rEns: 0, regime: "Turbulent", friction: 1.0, tick: 0,
  });

  // Dark mode observer
  useEffect(() => {
    const check = () => setDarkMode(document.documentElement.classList.contains("dark"));
    check();
    const obs = new MutationObserver(check);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);

  // K slider → kRef (no re-render per frame)
  const handleK = useCallback((e) => {
    const v = parseFloat(e.target.value);
    kRef.current = v;
    setKState(v);
  }, []);

  // Reset agents
  const handleReset = useCallback(() => {
    cpuRef.current.agents = makeInitialAgents();
    histRef.current = [];
    rEnsRef.current = 0;
    if (gpuRef.current) {
      // Re-upload initial agents
      const { device, agentBuf0, agentBuf1 } = gpuRef.current;
      const arr = new Float32Array(N_AGENTS * AGENT_STRIDE);
      cpuRef.current.agents.forEach((a, i) => {
        arr[i * AGENT_STRIDE + 0] = a.sk;
        arr[i * AGENT_STRIDE + 1] = a.st;
        arr[i * AGENT_STRIDE + 2] = a.se;
        arr[i * AGENT_STRIDE + 3] = a.phase;
        arr[i * AGENT_STRIDE + 4] = a.omega;
      });
      device.queue.writeBuffer(agentBuf0, 0, arr);
      device.queue.writeBuffer(agentBuf1, 0, arr);
      gpuRef.current.front = 0;
    }
    tickRef.current = 0;
  }, []);

  // Main simulation + draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let simMatrix = cpuSpectralDP(cpuRef.current.agents);
    let destroyed = false;

    (async () => {
      // Try WebGPU
      const gpu = await initWebGPU(cpuRef.current.agents);
      if (!destroyed) {
        gpuRef.current = gpu;
        setGpuSupported(!!gpu);
      }

      let lastStatUpdate = 0;
      const DT = 0.016;

      const frame = async (ts) => {
        if (destroyed) return;
        const tick = tickRef.current++;
        const K = kRef.current;

        if (gpu && !destroyed) {
          // GPU path
          const buffers = encodeGPUFrame(gpu, K, DT, tick * DT, tick);
          const rb = await readbackGPU(gpu, buffers, tick);
          if (destroyed) return;
          cpuRef.current.agents = rb.agents;
          simMatrix = rb.simMatrix;
          if (rb.rEns !== null) rEnsRef.current = rb.rEns;
        } else {
          // CPU fallback
          cpuRef.current.agents = cpuStep(cpuRef.current.agents, K, DT);
          simMatrix = cpuSpectralDP(cpuRef.current.agents);
          rEnsRef.current = cpuRens(cpuRef.current.agents);
        }

        const rEns = rEnsRef.current;
        histRef.current.push(rEns);
        if (histRef.current.length > 200) histRef.current.shift();

        // Draw
        drawScene(canvas, cpuRef.current.agents, simMatrix, rEns, histRef.current, darkMode);

        // Update React state every ~500ms
        if (ts - lastStatUpdate > 500) {
          lastStatUpdate = ts;
          const r = coordRegime(rEns);
          setLiveStats({ rEns, regime: r.name, friction: r.friction, tick });
        }

        rafRef.current = requestAnimationFrame(frame);
      };

      rafRef.current = requestAnimationFrame(frame);
    })();

    return () => {
      destroyed = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (gpuRef.current) {
        gpuRef.current.device.destroy();
        gpuRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resize canvas to container
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width  = Math.floor(rect.width);
      canvas.height = Math.floor(rect.width * 0.56);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  const regime = coordRegime(liveStats.rEns);

  return (
    <>
      <Head>
        <title>Purpose-Driven Shader VM | Pylon Framework</title>
        <meta
          name="description"
          content="Browser-native WebGPU implementation of the Pylon PDSVM: Kuramoto phase dynamics and SpectralDP similarity over S-entropy coordinates, running live in your GPU."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Purpose-Driven Shader VM"
            className="mb-6 !text-7xl !leading-tight lg:!text-6xl sm:!text-5xl xs:!text-3xl sm:mb-4"
          />

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mx-auto mb-4 max-w-3xl text-center text-base font-medium leading-relaxed
            text-dark/70 dark:text-light/70"
          >
            Eight agents evolve on the 5-dimensional coordination manifold
            M&nbsp;=&nbsp;[0,1]&times;[0,2&pi;&sup2;]&times;[0,1]&sup3;. Every frame, two GPU
            compute passes run:{" "}
            <strong>Kuramoto</strong> updates agent phases via S-entropy–weighted coupling, and{" "}
            <strong>SpectralDP</strong> computes the full pairwise similarity matrix. Watch
            coordination friction drop discontinuously to zero as R&#x2093;&#x2099;&#x2e;
            crosses 0.95.
          </motion.p>

          {/* GPU badge */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mb-8 flex items-center justify-center gap-3"
          >
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500 opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-green-600" />
            </span>
            <span className="text-xs font-semibold uppercase tracking-widest text-green-600 dark:text-green-400">
              {gpuSupported === null && "Initialising GPU..."}
              {gpuSupported === true  && "WebGPU compute — GPU-accelerated"}
              {gpuSupported === false && "CPU simulation — WebGPU unavailable in this browser"}
            </span>
          </motion.div>

          {/* ── Controls ─────────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mb-6 flex flex-wrap items-center justify-center gap-8"
          >
            <label className="flex flex-col items-center gap-2">
              <span className="text-xs font-medium uppercase tracking-widest text-dark/50 dark:text-light/50">
                Coupling K = {K.toFixed(2)}
              </span>
              <input
                type="range" min="0" max="4" step="0.05"
                value={K} onChange={handleK}
                className="w-56 accent-primary"
              />
              <span className="text-xs text-dark/40 dark:text-light/40">
                0 = free &nbsp;|&nbsp; 4 = phase-locked
              </span>
            </label>

            <button
              onClick={handleReset}
              className="rounded-xl border border-solid border-dark/20 bg-light px-5 py-2.5
              text-sm font-semibold text-dark transition hover:bg-dark hover:text-light
              dark:border-light/20 dark:bg-dark dark:text-light dark:hover:bg-light dark:hover:text-dark"
            >
              Randomise Agents
            </button>
          </motion.div>

          {/* ── Canvas ───────────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="relative mb-8 w-full overflow-hidden rounded-2xl border border-solid
            border-dark/20 bg-light dark:border-light/20 dark:bg-dark"
          >
            <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem]
            rounded-br-3xl bg-dark dark:bg-light md:-right-2 md:w-[101%]" />
            <canvas
              ref={canvasRef}
              className="block w-full"
              style={{ imageRendering: "pixelated" }}
            />
          </motion.div>

          {/* ── Live stat cards ────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-12 grid grid-cols-4 gap-5 lg:grid-cols-2 sm:grid-cols-1"
          >
            <StatCard
              label="Kuramoto R_ens"
              value={liveStats.rEns.toFixed(4)}
              accent
              sub="Order parameter [0,1]"
            />
            <StatCard
              label="Coordination Regime"
              value={liveStats.regime}
              sub="5 regimes: turbulent → phase-locked"
            />
            <StatCard
              label="Coordination Friction"
              value={liveStats.friction.toFixed(2)}
              sub="Drops to 0 at phase-lock"
            />
            <StatCard
              label="Coupling K"
              value={K.toFixed(2)}
              sub={`K_c ≈ 2σ_ω/π — critical coupling`}
            />
          </motion.div>

          {/* ── Theory panel ────────────────────────────────────────────── */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-16 grid grid-cols-3 gap-6 md:grid-cols-1"
          >
            {[
              {
                title: "SpectralDP",
                color: "#B63E96",
                body: "Fragment-shader dot product in S-entropy space (Sk, St, Se). Identifies action-cell equivalence: agents with cosine similarity > 0.65 share a common coordination cell. Complexity O(1) per pair — the heatmap updates every GPU frame.",
              },
              {
                title: "Kuramoto on S-Entropy",
                color: "#58E6D9",
                body: "Phase update: dφᵢ/dt = ωᵢ + (K/N)Σⱼ wᵢⱼ sin(φⱼ−φᵢ) where wᵢⱼ = ½(cos(sᵢ,sⱼ)+1). S-entropy similarity gates coupling: coherent agents in the same partition cell synchronise faster. R_ens = |Σ eⁱφ|/N.",
              },
              {
                title: "Friction → Zero",
                color: "#3b82f6",
                body: "Coordination friction f = 1 − R_ens for R_ens < 0.95, then drops discontinuously to 0 at phase-lock. This is the synchronisation phase transition: partition extinction. Drag K above ≈ 1.5 to drive the ensemble through cascade into phase-lock.",
              },
            ].map(({ title, color, body }) => (
              <div
                key={title}
                className="relative rounded-2xl border border-solid border-dark/20
                bg-light p-6 dark:border-light/20 dark:bg-dark"
              >
                <div className="absolute top-0 -right-2 -z-10 h-[102%] w-[101%] rounded-[1.5rem]
                rounded-br-3xl bg-dark dark:bg-light" />
                <div
                  className="mb-3 h-1 w-12 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <h3 className="mb-2 text-base font-bold text-dark dark:text-light">{title}</h3>
                <p className="text-sm font-medium leading-relaxed text-dark/60 dark:text-light/60">{body}</p>
              </div>
            ))}
          </motion.section>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-8 text-center text-sm font-medium text-dark/50 dark:text-light/50"
          >
            All computation runs in your browser.{" "}
            {gpuSupported
              ? "WebGPU compute shaders execute on your GPU — zero server round-trips."
              : "CPU fallback active — Chrome 113+ or Edge 113+ required for GPU mode."}
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
