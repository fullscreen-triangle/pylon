import React, { useRef, useEffect, useState } from "react";

const ContentVarianceChart = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 400 });

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
    const margin = { top: 35, right: 130, bottom: 55, left: 65 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";

    // 4 decay curves for T0 = 1, 5, 10, 20
    const T0s = [1, 5, 10, 20];
    const taus = [0.499, 0.499, 0.500, 0.500];
    const colors = ["#B63E96", "#58E6D9", "#FF6B6B", "#4ECDC4"];

    // Generate data for each curve
    const curves = T0s.map((T0, idx) => {
      const pts = [];
      const tau = taus[idx];
      for (let t = 0; t <= 5; t += 0.05) {
        const deltaT = T0 * Math.exp(-t / tau);
        pts.push({ t: parseFloat(t.toFixed(3)), deltaT: parseFloat(deltaT.toFixed(4)) });
      }
      return { T0, tau, pts, color: colors[idx] };
    });

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 5]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 22]).range([innerH, 0]);

    // Grid
    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(6))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Curves
    const lineGen = d3.line().x(d => xScale(d.t)).y(d => yScale(d.deltaT)).curve(d3.curveMonotoneX);

    curves.forEach((curve, idx) => {
      // Line
      g.append("path")
        .datum(curve.pts)
        .attr("fill", "none")
        .attr("stroke", curve.color)
        .attr("stroke-width", 2.5)
        .attr("d", lineGen);

      // Sample points along the curve
      const samplePts = curve.pts.filter((d, i) => i % 10 === 0);
      g.selectAll(`circle.c${idx}`)
        .data(samplePts)
        .join("circle")
        .attr("class", `c${idx}`)
        .attr("cx", d => xScale(d.t))
        .attr("cy", d => yScale(d.deltaT))
        .attr("r", 3)
        .attr("fill", curve.color)
        .attr("fill-opacity", 0.8);
    });

    // tau annotation
    const tauX = xScale(0.5);
    g.append("line")
      .attr("x1", tauX).attr("y1", 0).attr("x2", tauX).attr("y2", innerH)
      .attr("stroke", textColor).attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.4);
    g.append("text")
      .attr("x", tauX + 5).attr("y", yScale(21))
      .style("fill", textColor).style("font-size", "11px").style("opacity", 0.6)
      .text("\u03C4 \u2248 0.499 ms");

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Time t (ms)");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 16)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("\u0394T(t) = T\u2080 exp(-t/\u03C4)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 20)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "14px").style("font-weight", "bold")
      .text("Variance Restoration: Exponential Decay (\u03C4 = 0.499 ms)");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${width - margin.right + 10}, ${margin.top + 10})`);
    curves.forEach((curve, i) => {
      const row = lg.append("g").attr("transform", `translate(0, ${i * 24})`);
      row.append("line").attr("x1", 0).attr("y1", 6).attr("x2", 20).attr("y2", 6)
        .attr("stroke", curve.color).attr("stroke-width", 2.5);
      row.append("text").attr("x", 25).attr("y", 10)
        .style("fill", textColor).style("font-size", "10px")
        .text(`T\u2080=${curve.T0}`);
    });
    lg.append("text").attr("x", 0).attr("y", curves.length * 24 + 15)
      .style("fill", textColor).style("font-size", "9px").style("opacity", 0.7)
      .text("\u03C4 \u2248 0.5 ms (all)");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentVarianceChart;
