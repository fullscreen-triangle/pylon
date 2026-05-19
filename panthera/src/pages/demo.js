import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import { useEffect, useRef, useState, useCallback } from "react";

// ─── Node registry ────────────────────────────────────────────────────────────
const NODES = [
  { id: 0, name: "You",    color: "#B63E96", initial: "Y" },
  { id: 1, name: "Node α", color: "#58E6D9", initial: "α" },
  { id: 2, name: "Node β", color: "#3b82f6", initial: "β" },
  { id: 3, name: "Node γ", color: "#22c55e", initial: "γ" },
  { id: 4, name: "Node δ", color: "#f59e0b", initial: "δ" },
  { id: 5, name: "Node ε", color: "#ef4444", initial: "ε" },
  { id: 6, name: "Node ζ", color: "#8b5cf6", initial: "ζ" },
  { id: 7, name: "Node η", color: "#ec4899", initial: "η" },
];

// ─── Coordination regime ──────────────────────────────────────────────────────
function coordRegime(r) {
  if (r < 0.30) return { name: "Turbulent",    color: "#ef4444", friction: 1.00 };
  if (r < 0.50) return { name: "Aperture",     color: "#f97316", friction: 0.72 };
  if (r < 0.80) return { name: "Cascade",      color: "#eab308", friction: 0.40 };
  if (r < 0.95) return { name: "Coherent",     color: "#22c55e", friction: 0.12 };
  return              { name: "Phase-Locked",  color: "#58E6D9", friction: 0.00 };
}

// ─── S-entropy from message content ──────────────────────────────────────────
function sEntropy(text) {
  let h1 = 5381, h2 = 52711, h3 = 9973;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = (((h1 << 5) + h1) ^ c) >>> 0;
    h2 = (((h2 << 7) + h2) ^ (c * 7)) >>> 0;
    h3 = (((h3 << 3) + h3) ^ (c * 13)) >>> 0;
  }
  return {
    sk: 0.15 + (h1 / 0xffffffff) * 0.70,
    st: 0.15 + (h2 / 0xffffffff) * 0.70,
    se: 0.15 + (h3 / 0xffffffff) * 0.70,
  };
}

// ─── Routing helpers ──────────────────────────────────────────────────────────
function deliveryMs(rEns) {
  const base = 4 + Math.random() * 12;
  return Math.round(base * (1 + (1 - rEns) * 3.5));
}

function backwardHops({ sk, st, se }) {
  const d = Math.sqrt((sk - 0.5) ** 2 + (st - 0.5) ** 2 + (se - 0.5) ** 2);
  return Math.max(2, Math.round(3 + d * 14));
}

// ─── Auto-response generation ─────────────────────────────────────────────────
function autoResponse(fromNode, se, netState, K) {
  const pool = [
    `S-entropy (${se.sk.toFixed(2)}, ${se.st.toFixed(2)}, ${se.se.toFixed(2)}) indexed. Action-cell confirmed.`,
    `Received in ${netState.regime} regime. R_ens = ${netState.rEns.toFixed(3)}.`,
    `Backward trajectory: ${backwardHops(se)} hops. O(log M) navigation complete.`,
    `Thermodynamic signature: clean. No entropy injection detected.`,
    `Phase synchronisation at ${(netState.rEns * 100).toFixed(1)}%. Coordination friction: ${netState.friction.toFixed(2)}.`,
    `Common action-cell identified. Disjoint decoders converge on shared cell.`,
    `Second Law compliance: satisfied. Coupling K = ${K.toFixed(2)}.`,
    `Route optimised via ternary trie. Delivery guaranteed in ${netState.regime} regime.`,
    `Cell-truth preserved: receiver floor β > 0. Message integrity: verified.`,
    `${fromNode.name} → Node 0. S-entropy distance: ${(Math.hypot(se.sk - 0.5, se.st - 0.5, se.se - 0.5)).toFixed(3)}.`,
  ];
  return pool[Math.floor(Math.random() * pool.length)];
}

// ─── Kuramoto step (CPU) ──────────────────────────────────────────────────────
function kuramotoStep(phases, omegas, K, dt = 0.05) {
  const n = phases.length;
  return phases.map((phi, i) => {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      sum += Math.sin(phases[j] - phi);
    }
    return phi + dt * (omegas[i] + (K / (n - 1)) * sum);
  });
}

function kuramotoRens(phases) {
  let c = 0, s = 0;
  phases.forEach(p => { c += Math.cos(p); s += Math.sin(p); });
  return Math.hypot(c, s) / phases.length;
}

// ─── Message bubble component ─────────────────────────────────────────────────
function Bubble({ msg, nodes }) {
  const isOutgoing = msg.from === 0;
  const node = nodes.find(n => n.id === (isOutgoing ? msg.to : msg.from));
  const regime = coordRegime(msg.rEns ?? 0.4);

  return (
    <div className={`flex flex-col mb-3 ${isOutgoing ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[72%] rounded-2xl px-4 py-2.5 text-sm font-medium leading-relaxed
          ${isOutgoing
            ? "rounded-br-sm text-white"
            : "rounded-bl-sm bg-light text-dark dark:bg-dark dark:text-light border border-solid border-dark/10 dark:border-light/10"
          }`}
        style={isOutgoing ? { backgroundColor: node?.color ?? "#B63E96" } : {}}
      >
        {msg.content}
      </div>
      <div className={`mt-1 flex items-center gap-2 text-[10px] font-mono text-dark/35 dark:text-light/35 ${isOutgoing ? "flex-row-reverse" : "flex-row"}`}>
        <span>
          S({msg.sk.toFixed(2)}, {msg.st.toFixed(2)}, {msg.se.toFixed(2)})
        </span>
        <span style={{ color: regime.color }} className="font-semibold">
          {regime.name}
        </span>
        <span>{msg.latencyMs}ms</span>
        <span>{msg.hops} hops</span>
        <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function Demo() {
  const [selectedNode, setSelectedNode]     = useState(1);
  const [inputText, setInputText]           = useState("");
  const [conversations, setConversations]   = useState(() =>
    Object.fromEntries(NODES.slice(1).map(n => [n.id, []]))
  );
  const [K, setKState]                      = useState(1.2);
  const [networkState, setNetworkState]     = useState({
    rEns: 0.35, regime: "Aperture", friction: 0.72,
  });

  const kRef            = useRef(1.2);
  const msgIdRef        = useRef(0);
  const messagesEndRef  = useRef(null);
  const netRef          = useRef({ rEns: 0.35, regime: "Aperture", friction: 0.72 });
  const histRef         = useRef([0.35]);
  const simRef          = useRef({
    phases: Array.from({ length: 8 }, () => Math.random() * 2 * Math.PI),
    omegas: Array.from({ length: 8 }, () => (Math.random() - 0.5) * 2.0),
  });
  const tickRef         = useRef(0);

  // ── Kuramoto background simulation ─────────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(() => {
      tickRef.current++;
      const { phases, omegas } = simRef.current;
      const newPhases = kuramotoStep(phases, omegas, kRef.current);
      simRef.current.phases = newPhases;

      if (tickRef.current % 4 === 0) {
        const rEns = kuramotoRens(newPhases);
        const r    = coordRegime(rEns);
        const next = { rEns, regime: r.name, friction: r.friction };
        netRef.current = next;
        histRef.current.push(rEns);
        if (histRef.current.length > 120) histRef.current.shift();
        setNetworkState(next);
      }
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // ── Seed greeting messages on mount ────────────────────────────────────────
  useEffect(() => {
    const greetings = [
      { node: 1, text: "Node α online. S-entropy coordinates synced." },
      { node: 3, text: "Node γ ready. Awaiting coordination from primary node." },
      { node: 5, text: "Node ε connected. Backward trajectory: initialised." },
    ];
    const initConvs = Object.fromEntries(NODES.slice(1).map(n => [n.id, []]));
    greetings.forEach(({ node, text }) => {
      const se = sEntropy(text);
      initConvs[node].push({
        id: msgIdRef.current++,
        from: node, to: 0,
        content: text,
        timestamp: Date.now() - 5000 + node * 300,
        ...se, rEns: 0.35, latencyMs: 18, hops: backwardHops(se),
      });
    });
    setConversations(initConvs);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Auto-scroll ─────────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, selectedNode]);

  // ── K slider ────────────────────────────────────────────────────────────────
  const handleK = useCallback((e) => {
    const v = parseFloat(e.target.value);
    kRef.current = v;
    setKState(v);
  }, []);

  // ── Send message ────────────────────────────────────────────────────────────
  const sendMessage = useCallback(() => {
    const text = inputText.trim();
    if (!text) return;
    const se      = sEntropy(text);
    const rEns    = netRef.current.rEns;
    const latency = deliveryMs(rEns);
    const hops    = backwardHops(se);

    const outMsg = {
      id: msgIdRef.current++,
      from: 0, to: selectedNode,
      content: text,
      timestamp: Date.now(),
      ...se, rEns, latencyMs: latency, hops,
    };

    setConversations(prev => ({
      ...prev,
      [selectedNode]: [...(prev[selectedNode] ?? []), outMsg],
    }));
    setInputText("");

    // Auto-response
    const delay = 900 + Math.random() * 1800;
    setTimeout(() => {
      const net    = netRef.current;
      const fromNode = NODES.find(n => n.id === selectedNode);
      const rText  = autoResponse(fromNode, se, net, kRef.current);
      const rSe    = sEntropy(rText);
      const inMsg  = {
        id: msgIdRef.current++,
        from: selectedNode, to: 0,
        content: rText,
        timestamp: Date.now(),
        ...rSe, rEns: net.rEns, latencyMs: deliveryMs(net.rEns), hops: backwardHops(rSe),
      };
      setConversations(prev => ({
        ...prev,
        [selectedNode]: [...(prev[selectedNode] ?? []), inMsg],
      }));
    }, delay);
  }, [inputText, selectedNode]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }, [sendMessage]);

  const currentRegime = coordRegime(networkState.rEns);
  const targetNode    = NODES.find(n => n.id === selectedNode);
  const messages      = conversations[selectedNode] ?? [];

  return (
    <>
      <Head>
        <title>Pylon Mesh Messenger | Live Demo</title>
        <meta
          name="description"
          content="Browser-native Pylon messaging demo. Messages are routed via backward trajectory completion and S-entropy coordinates. Watch coordination regime transitions in real time."
        />
      </Head>

      <TransitionEffect />

      <main className="flex w-full flex-col dark:text-light" style={{ minHeight: "calc(100vh - 5.5rem)" }}>
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div className="border-b border-solid border-dark/10 px-6 py-4 dark:border-light/10">
          <div className="mx-auto flex max-w-7xl items-baseline gap-4">
            <h1 className="text-2xl font-bold text-dark dark:text-light">Pylon Mesh Messenger</h1>
            <p className="text-sm font-medium text-dark/50 dark:text-light/50">
              Messages routed via S-entropy coordinates and backward trajectory completion
            </p>
          </div>
        </div>

        {/* ── Three-column layout ─────────────────────────────────────────── */}
        <div className="mx-auto flex w-full max-w-7xl flex-1 gap-0 overflow-hidden px-6 py-4" style={{ height: "calc(100vh - 9rem)" }}>

          {/* ── Left: Node list ─────────────────────────────────────────── */}
          <aside className="flex w-48 flex-shrink-0 flex-col rounded-l-2xl border border-solid border-dark/15 bg-light dark:border-light/15 dark:bg-dark overflow-hidden">
            <div className="border-b border-solid border-dark/10 px-4 py-3 dark:border-light/10">
              <p className="text-xs font-semibold uppercase tracking-widest text-dark/40 dark:text-light/40">Nodes</p>
            </div>
            <div className="flex-1 overflow-y-auto">
              {NODES.slice(1).map(node => {
                const unread = (conversations[node.id] ?? []).filter(m => m.from === node.id).length;
                const active = selectedNode === node.id;
                return (
                  <button
                    key={node.id}
                    onClick={() => setSelectedNode(node.id)}
                    className={`flex w-full items-center gap-3 px-4 py-3 text-left transition-colors
                      ${active
                        ? "bg-dark/8 dark:bg-light/8"
                        : "hover:bg-dark/4 dark:hover:bg-light/4"
                      }`}
                  >
                    <span
                      className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold text-white"
                      style={{ backgroundColor: node.color }}
                    >
                      {node.initial}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className={`truncate text-sm font-semibold ${active ? "text-dark dark:text-light" : "text-dark/70 dark:text-light/70"}`}>
                        {node.name}
                      </p>
                      <p className="text-[10px] text-dark/35 dark:text-light/35">
                        {unread} msg{unread !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <span
                      className="h-2 w-2 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: node.color, opacity: 0.7 }}
                    />
                  </button>
                );
              })}
            </div>
          </aside>

          {/* ── Centre: Chat area ────────────────────────────────────────── */}
          <section className="flex flex-1 flex-col border-y border-solid border-dark/15 bg-light/60 dark:border-light/15 dark:bg-dark/60 overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 border-b border-solid border-dark/10 px-5 py-3 dark:border-light/10">
              <span
                className="flex h-9 w-9 items-center justify-center rounded-full text-base font-bold text-white flex-shrink-0"
                style={{ backgroundColor: targetNode?.color }}
              >
                {targetNode?.initial}
              </span>
              <div>
                <p className="font-bold text-dark dark:text-light">{targetNode?.name}</p>
                <p className="text-xs font-mono text-dark/40 dark:text-light/40">
                  Node {selectedNode} · {currentRegime.name} · R_ens {networkState.rEns.toFixed(3)}
                </p>
              </div>
              <div className="ml-auto flex items-center gap-2">
                <span
                  className="rounded-full px-2.5 py-0.5 text-[10px] font-bold text-white"
                  style={{ backgroundColor: currentRegime.color }}
                >
                  {currentRegime.name}
                </span>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-dark/30 dark:text-light/30">
                    Send a message to {targetNode?.name} to begin coordination
                  </p>
                </div>
              ) : (
                messages.map(msg => (
                  <Bubble key={msg.id} msg={msg} nodes={NODES} />
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-solid border-dark/10 px-4 py-3 dark:border-light/10">
              <div className="flex items-end gap-3">
                <textarea
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Message ${targetNode?.name}…`}
                  rows={1}
                  className="flex-1 resize-none rounded-xl border border-solid border-dark/20 bg-light
                  px-4 py-2.5 text-sm font-medium text-dark placeholder-dark/30 outline-none
                  transition focus:border-primary dark:border-light/20 dark:bg-dark dark:text-light
                  dark:placeholder-light/30 dark:focus:border-primaryDark"
                  style={{ minHeight: "2.5rem", maxHeight: "6rem" }}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputText.trim()}
                  className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl
                  text-white transition disabled:opacity-30"
                  style={{ backgroundColor: targetNode?.color }}
                >
                  <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4 rotate-90">
                    <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                  </svg>
                </button>
              </div>
              <p className="mt-1.5 text-[10px] font-mono text-dark/25 dark:text-light/25">
                Enter to send · Each message gets S-entropy coords and backward-trajectory routing
              </p>
            </div>
          </section>

          {/* ── Right: Network panel ─────────────────────────────────────── */}
          <aside className="flex w-56 flex-shrink-0 flex-col rounded-r-2xl border border-solid border-dark/15 bg-light dark:border-light/15 dark:bg-dark overflow-hidden">
            <div className="border-b border-solid border-dark/10 px-4 py-3 dark:border-light/10">
              <p className="text-xs font-semibold uppercase tracking-widest text-dark/40 dark:text-light/40">Network</p>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
              {/* R_ens */}
              <div>
                <div className="mb-1 flex items-baseline justify-between">
                  <p className="text-xs font-medium text-dark/50 dark:text-light/50">R_ens</p>
                  <p className="font-mono text-sm font-bold" style={{ color: currentRegime.color }}>
                    {networkState.rEns.toFixed(4)}
                  </p>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-dark/10 dark:bg-light/10">
                  <div
                    className="h-full rounded-full transition-all duration-200"
                    style={{ width: `${networkState.rEns * 100}%`, backgroundColor: currentRegime.color }}
                  />
                </div>
                {/* Sparkline */}
                <svg className="mt-2 w-full" height="32" viewBox={`0 0 120 32`}>
                  <polyline
                    fill="none"
                    stroke={currentRegime.color}
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={histRef.current
                      .slice(-60)
                      .map((v, i, arr) => `${(i / (arr.length - 1)) * 120},${32 - v * 28}`)
                      .join(" ")}
                  />
                  {[0.3, 0.5, 0.8, 0.95].map(t => (
                    <line key={t} x1="0" x2="120" y1={32 - t * 28} y2={32 - t * 28}
                      stroke="currentColor" strokeWidth="0.5" strokeDasharray="2,3"
                      className="text-dark/15 dark:text-light/15" />
                  ))}
                </svg>
              </div>

              {/* Regime */}
              <div>
                <p className="mb-1 text-xs font-medium text-dark/50 dark:text-light/50">Regime</p>
                <span
                  className="inline-block rounded-full px-3 py-1 text-xs font-bold text-white"
                  style={{ backgroundColor: currentRegime.color }}
                >
                  {currentRegime.name}
                </span>
                <div className="mt-2 grid grid-cols-2 gap-1.5 text-[10px] font-mono text-dark/40 dark:text-light/40">
                  {[["Turbulent", 0.3], ["Aperture", 0.5], ["Cascade", 0.8], ["Coherent", 0.95]].map(([name, thresh]) => (
                    <span key={name} className={networkState.rEns >= thresh ? "opacity-100" : "opacity-30"}>
                      {name}
                    </span>
                  ))}
                </div>
              </div>

              {/* Friction */}
              <div>
                <p className="mb-1 text-xs font-medium text-dark/50 dark:text-light/50">Coordination Friction</p>
                <p className="font-mono text-xl font-bold text-dark dark:text-light">
                  {currentRegime.friction.toFixed(2)}
                </p>
                {currentRegime.friction === 0 && (
                  <p className="mt-0.5 text-[10px] text-teal-500 font-semibold">Zero-friction delivery</p>
                )}
              </div>

              {/* K slider */}
              <div>
                <div className="mb-1 flex items-baseline justify-between">
                  <p className="text-xs font-medium text-dark/50 dark:text-light/50">Coupling K</p>
                  <p className="font-mono text-xs font-bold text-dark dark:text-light">{K.toFixed(2)}</p>
                </div>
                <input
                  type="range" min="0" max="4" step="0.05"
                  value={K} onChange={handleK}
                  className="w-full accent-primary"
                />
                <p className="mt-1 text-[10px] text-dark/30 dark:text-light/30">
                  K_c ≈ 2σ_ω/π · drag right for phase-lock
                </p>
              </div>

              {/* Security badge */}
              <div className="rounded-xl border border-solid border-dark/10 bg-dark/4 p-3 dark:border-light/10 dark:bg-light/4">
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-dark/40 dark:text-light/40">
                  Thermodynamic Security
                </p>
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-green-500" />
                  <p className="text-[10px] font-mono text-dark/60 dark:text-light/60">
                    No entropy injection
                  </p>
                </div>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className="h-2 w-2 rounded-full bg-green-500" />
                  <p className="text-[10px] font-mono text-dark/60 dark:text-light/60">
                    Second Law satisfied
                  </p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </>
  );
}
