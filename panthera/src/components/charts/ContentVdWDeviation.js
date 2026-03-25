import React, { useRef, useEffect, useState } from "react";

const ContentVdWDeviation = () => {
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
    const margin = { top: 30, right: 20, bottom: 50, left: 55 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const accentColor = darkMode ? "#58E6D9" : "#B63E96";

    const B2 = -1.310;
    const pts = [];
    for (let i = 0; i < 25; i++) {
      const rho = 0.02 * (i + 1);
      const Zideal = 1.0;
      const Zvirial = 1 + B2 * rho;
      const deviation = Zvirial - Zideal;
      pts.push({ rho: parseFloat(rho.toFixed(3)), deviation: parseFloat(deviation.toFixed(4)) });
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, 0.55]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([-0.7, 0.05]).range([innerH, 0]);

    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Zero line
    g.append("line").attr("x1", 0).attr("y1", yScale(0)).attr("x2", innerW).attr("y2", yScale(0))
      .attr("stroke", textColor).attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.3);

    // Deviation curve
    const area = d3.area().x(d => xScale(d.rho)).y0(yScale(0)).y1(d => yScale(d.deviation)).curve(d3.curveMonotoneX);
    g.append("path").datum(pts).attr("fill", accentColor).attr("fill-opacity", 0.15).attr("d", area);

    const line = d3.line().x(d => xScale(d.rho)).y(d => yScale(d.deviation)).curve(d3.curveMonotoneX);
    g.append("path").datum(pts).attr("fill", "none").attr("stroke", accentColor).attr("stroke-width", 2).attr("d", line);

    g.selectAll("circle").data(pts.filter((d, i) => i % 3 === 0)).join("circle")
      .attr("cx", d => xScale(d.rho)).attr("cy", d => yScale(d.deviation)).attr("r", 3).attr("fill", accentColor);

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Density \u03C1");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Z - 1 (deviation)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Deviation from Ideal Gas (B\u2082 = -1.310)");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentVdWDeviation;
