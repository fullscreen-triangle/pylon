import React, { useRef, useEffect, useState } from "react";

const ContentVarianceDecay = () => {
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
    const margin = { top: 30, right: 110, bottom: 50, left: 55 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const colors = ["#B63E96", "#58E6D9", "#FF6B6B", "#4ECDC4"];
    const T0s = [1, 5, 10, 20];
    const taus = [0.499, 0.499, 0.500, 0.500];

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, 4]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 22]).range([innerH, 0]);

    T0s.forEach((T0, idx) => {
      const pts = [];
      for (let t = 0; t <= 4; t += 0.04) {
        pts.push({ t, val: T0 * Math.exp(-t / taus[idx]) });
      }
      const line = d3.line().x(d => xScale(d.t)).y(d => yScale(d.val)).curve(d3.curveMonotoneX);
      g.append("path").datum(pts).attr("fill", "none").attr("stroke", colors[idx]).attr("stroke-width", 2).attr("d", line);
    });

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Time (ms)");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("\u0394T");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Variance Restoration (\u03C4 \u2248 0.5 ms)");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${width - margin.right + 8}, ${margin.top + 8})`);
    T0s.forEach((T0, i) => {
      const row = lg.append("g").attr("transform", `translate(0,${i * 20})`);
      row.append("line").attr("x1", 0).attr("y1", 5).attr("x2", 16).attr("y2", 5).attr("stroke", colors[i]).attr("stroke-width", 2);
      row.append("text").attr("x", 20).attr("y", 9).style("fill", textColor).style("font-size", "10px").text(`T\u2080=${T0}`);
    });
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentVarianceDecay;
