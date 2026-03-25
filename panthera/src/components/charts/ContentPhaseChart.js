import React, { useRef, useEffect, useState } from "react";

const ContentPhaseChart = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 400 });

  // Phase transition data: order parameter vs temperature
  const data = (() => {
    const pts = [];
    for (let i = 0; i < 50; i++) {
      const T = 5.0 - i * (4.99 / 49);
      let psi;
      if (T > 3.42) {
        psi = 0.02 + Math.random() * 0.03; // Gas phase
      } else if (T > 2.65) {
        psi = 0.05 + (3.42 - T) / (3.42 - 2.65) * 0.45; // Liquid phase (transition)
      } else {
        psi = 0.5 + (2.65 - T) / 2.65 * 0.5; // Crystal phase
      }
      psi = Math.min(1.0, psi + (Math.random() - 0.5) * 0.02);
      pts.push({ T: parseFloat(T.toFixed(3)), psi: parseFloat(psi.toFixed(4)) });
    }
    return pts;
  })();

  useEffect(() => {
    import("d3").then((d3) => { d3Ref.current = d3; renderChart(); });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) setDimensions({ width, height: Math.min(400, Math.max(280, width * 0.55)) });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => { renderChart(); }, [dimensions]);

  function renderChart() {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current) return;
    const darkMode = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
    const { width, height } = dimensions;
    const margin = { top: 35, right: 30, bottom: 55, left: 65 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const accentColor = darkMode ? primaryDark : primary;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 5.5]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1.1]).range([innerH, 0]);

    // Grid
    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Phase background regions
    const gasColor = darkMode ? "rgba(255,100,100,0.08)" : "rgba(255,100,100,0.08)";
    const liquidColor = darkMode ? "rgba(100,100,255,0.08)" : "rgba(100,100,255,0.08)";
    const crystalColor = darkMode ? "rgba(88,230,217,0.08)" : "rgba(182,62,150,0.08)";

    // Crystal region
    g.append("rect").attr("x", 0).attr("y", 0).attr("width", xScale(2.65)).attr("height", innerH)
      .attr("fill", crystalColor);
    // Liquid region
    g.append("rect").attr("x", xScale(2.65)).attr("y", 0).attr("width", xScale(3.42) - xScale(2.65)).attr("height", innerH)
      .attr("fill", liquidColor);
    // Gas region
    g.append("rect").attr("x", xScale(3.42)).attr("y", 0).attr("width", innerW - xScale(3.42)).attr("height", innerH)
      .attr("fill", gasColor);

    // Phase labels
    g.append("text").attr("x", xScale(1.3)).attr("y", yScale(1.05))
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").style("font-weight", "bold").style("opacity", 0.6).text("Crystal");
    g.append("text").attr("x", xScale(3.03)).attr("y", yScale(1.05))
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").style("font-weight", "bold").style("opacity", 0.6).text("Liquid");
    g.append("text").attr("x", xScale(4.3)).attr("y", yScale(1.05))
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").style("font-weight", "bold").style("opacity", 0.6).text("Gas");

    // Critical temperature lines
    [{ T: 3.42, label: "T_c = 3.42" }, { T: 2.65, label: "T_m = 2.65" }].forEach(({ T, label }) => {
      g.append("line")
        .attr("x1", xScale(T)).attr("y1", 0).attr("x2", xScale(T)).attr("y2", innerH)
        .attr("stroke", accentColor).attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
      g.append("text")
        .attr("x", xScale(T) + 5).attr("y", yScale(0.95))
        .style("fill", accentColor).style("font-size", "11px").style("font-weight", "bold").text(label);
    });

    // Line
    const line = d3.line().x(d => xScale(d.T)).y(d => yScale(d.psi)).curve(d3.curveMonotoneX);
    const sortedData = [...data].sort((a, b) => a.T - b.T);

    g.append("path")
      .datum(sortedData)
      .attr("fill", "none")
      .attr("stroke", accentColor)
      .attr("stroke-width", 2.5)
      .attr("d", line);

    // Points
    g.selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", d => xScale(d.T))
      .attr("cy", d => yScale(d.psi))
      .attr("r", 3.5)
      .attr("fill", accentColor)
      .attr("fill-opacity", 0.8);

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Temperature T");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 16)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Order Parameter \u03A8");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 20)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "14px").style("font-weight", "bold")
      .text("Phase Transition: Order Parameter vs Temperature");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentPhaseChart;
