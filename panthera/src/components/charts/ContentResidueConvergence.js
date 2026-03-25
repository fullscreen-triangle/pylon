import React, { useRef, useEffect, useState } from "react";

const ContentResidueConvergence = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 320 });

  useEffect(() => {
    import("d3").then((d3) => { d3Ref.current = d3; renderChart(); });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) setDimensions({ width, height: Math.min(320, Math.max(220, width * 0.5)) });
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
    const margin = { top: 30, right: 130, bottom: 50, left: 55 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";

    // Triple convergence: oscillatory, categorical, parametric
    // convergence_rate = 1.0, divergence_rate = 0.923
    const nPts = 50;
    const colors = [darkMode ? primaryDark : primary, "#FF6B6B", "#4ECDC4"];
    const labels = ["\u03B5_osc (oscillatory)", "\u03B5_cat (categorical)", "\u03B5_par (parametric)"];

    const curves = [
      // Oscillatory convergence - rapid
      Array.from({ length: nPts }, (_, i) => ({
        step: i,
        error: 1.0 * Math.exp(-i * 0.15) + Math.sin(i * 0.8) * 0.02 * Math.exp(-i * 0.1),
      })),
      // Categorical convergence - moderate
      Array.from({ length: nPts }, (_, i) => ({
        step: i,
        error: 1.0 * Math.exp(-i * 0.12) + Math.cos(i * 0.6) * 0.015 * Math.exp(-i * 0.08),
      })),
      // Parametric convergence - slowest
      Array.from({ length: nPts }, (_, i) => ({
        step: i,
        error: 1.0 * Math.exp(-i * 0.1) + Math.sin(i * 1.2) * 0.01 * Math.exp(-i * 0.06),
      })),
    ];

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, nPts]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([0.001, 1.2]).range([innerH, 0]);

    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(4))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    const line = d3.line().x(d => xScale(d.step)).y(d => yScale(Math.max(0.001, d.error))).curve(d3.curveMonotoneX);

    curves.forEach((curve, idx) => {
      g.append("path").datum(curve).attr("fill", "none").attr("stroke", colors[idx]).attr("stroke-width", 2).attr("d", line);
    });

    // Convergence target line
    g.append("line").attr("x1", 0).attr("y1", yScale(0.01)).attr("x2", innerW).attr("y2", yScale(0.01))
      .attr("stroke", textColor).attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.3);
    g.append("text").attr("x", innerW).attr("y", yScale(0.01) - 5).attr("text-anchor", "end")
      .style("fill", textColor).style("font-size", "9px").style("opacity", 0.5).text("convergence threshold");

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(4, ".0e"))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Iteration Step");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Error (log)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("G\u00F6delian Residue: Triple Convergence");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${width - margin.right + 8}, ${margin.top + 5})`);
    labels.forEach((label, i) => {
      const row = lg.append("g").attr("transform", `translate(0,${i * 20})`);
      row.append("line").attr("x1", 0).attr("y1", 5).attr("x2", 16).attr("y2", 5).attr("stroke", colors[i]).attr("stroke-width", 2);
      row.append("text").attr("x", 20).attr("y", 9).style("fill", textColor).style("font-size", "9px").text(label);
    });
    lg.append("text").attr("x", 0).attr("y", labels.length * 20 + 12).style("fill", textColor).style("font-size", "9px").style("opacity", 0.7).text("Conv: 100%");
    lg.append("text").attr("x", 0).attr("y", labels.length * 20 + 24).style("fill", textColor).style("font-size", "9px").style("opacity", 0.7).text("Div: 92.3%");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentResidueConvergence;
