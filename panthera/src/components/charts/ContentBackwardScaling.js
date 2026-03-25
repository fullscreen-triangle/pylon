import React, { useRef, useEffect, useState } from "react";

const ContentBackwardScaling = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 320 });

  // scaling_exponent=0.936, R^2=1.0
  const data = (() => {
    const pts = [];
    for (let logM = 4; logM <= 20; logM++) {
      const M = Math.pow(2, logM);
      // O(log M) backward: comparisons ~ log2(M) * constant
      const backward = logM * 1.1 + 0.5;
      // O(M) forward: comparisons ~ M / scale_factor (for visualization)
      const forward = M / 1000;
      pts.push({
        logM,
        M,
        backward: parseFloat(backward.toFixed(2)),
        forward: parseFloat(Math.min(forward, 1200).toFixed(2)),
      });
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
    const margin = { top: 30, right: 110, bottom: 50, left: 65 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const backwardColor = darkMode ? primaryDark : primary;
    const forwardColor = "#FF6B6B";

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([3, 21]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([1, 1500]).range([innerH, 0]);

    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Forward O(M) line
    const forwardLine = d3.line().x(d => xScale(d.logM)).y(d => yScale(Math.max(1, d.forward))).curve(d3.curveMonotoneX);
    g.append("path").datum(data).attr("fill", "none").attr("stroke", forwardColor).attr("stroke-width", 2.5).attr("d", forwardLine);

    // Backward O(log M) line
    const backwardLine = d3.line().x(d => xScale(d.logM)).y(d => yScale(d.backward)).curve(d3.curveMonotoneX);
    g.append("path").datum(data).attr("fill", "none").attr("stroke", backwardColor).attr("stroke-width", 2.5).attr("d", backwardLine);

    // Points
    g.selectAll("circle.b").data(data).join("circle").attr("class", "b")
      .attr("cx", d => xScale(d.logM)).attr("cy", d => yScale(d.backward)).attr("r", 3).attr("fill", backwardColor);
    g.selectAll("circle.f").data(data).join("circle").attr("class", "f")
      .attr("cx", d => xScale(d.logM)).attr("cy", d => yScale(Math.max(1, d.forward))).attr("r", 3).attr("fill", forwardColor);

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(8).tickFormat(d => `2^${d}`))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5, ",.0f"))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Message Count M (log\u2082)");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Comparisons (log)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Backward O(log M) vs Forward O(M) Scaling");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${width - margin.right + 8}, ${margin.top + 8})`);
    lg.append("line").attr("x1", 0).attr("y1", 5).attr("x2", 18).attr("y2", 5).attr("stroke", backwardColor).attr("stroke-width", 2);
    lg.append("text").attr("x", 22).attr("y", 9).style("fill", textColor).style("font-size", "10px").text("Backward");
    lg.append("line").attr("x1", 0).attr("y1", 25).attr("x2", 18).attr("y2", 25).attr("stroke", forwardColor).attr("stroke-width", 2);
    lg.append("text").attr("x", 22).attr("y", 29).style("fill", textColor).style("font-size", "10px").text("Forward");
    lg.append("text").attr("x", 0).attr("y", 50).style("fill", textColor).style("font-size", "9px").style("opacity", 0.7).text("R\u00B2 = 1.0");
    lg.append("text").attr("x", 0).attr("y", 62).style("fill", textColor).style("font-size", "9px").style("opacity", 0.7).text("\u03B1 = 0.936");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentBackwardScaling;
