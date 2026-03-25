import React, { useRef, useEffect, useState } from "react";

const ContentPhaseOrderParam = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 320 });

  const data = (() => {
    const pts = [];
    for (let i = 0; i < 50; i++) {
      const T = 5.0 - i * (4.99 / 49);
      let psi;
      if (T > 3.42) psi = 0.02 + Math.sin(i * 0.8) * 0.02;
      else if (T > 2.65) psi = 0.05 + (3.42 - T) / (3.42 - 2.65) * 0.45 + Math.sin(i * 1.5) * 0.02;
      else psi = 0.5 + (2.65 - T) / 2.65 * 0.5 + Math.sin(i * 2.1) * 0.01;
      pts.push({ T: parseFloat(T.toFixed(3)), psi: Math.min(1, Math.max(0, parseFloat(psi.toFixed(4)))) });
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
    const margin = { top: 30, right: 20, bottom: 50, left: 55 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const accentColor = darkMode ? "#58E6D9" : "#B63E96";

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, 5.5]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1.1]).range([innerH, 0]);

    // Phase backgrounds
    g.append("rect").attr("x", 0).attr("y", 0).attr("width", xScale(2.65)).attr("height", innerH).attr("fill", darkMode ? "rgba(88,230,217,0.06)" : "rgba(182,62,150,0.06)");
    g.append("rect").attr("x", xScale(2.65)).attr("y", 0).attr("width", xScale(3.42) - xScale(2.65)).attr("height", innerH).attr("fill", "rgba(100,100,255,0.06)");
    g.append("rect").attr("x", xScale(3.42)).attr("y", 0).attr("width", innerW - xScale(3.42)).attr("height", innerH).attr("fill", "rgba(255,100,100,0.06)");

    // Phase labels
    g.append("text").attr("x", xScale(1.3)).attr("y", 15).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "10px").style("opacity", 0.5).text("Crystal");
    g.append("text").attr("x", xScale(3.03)).attr("y", 15).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "10px").style("opacity", 0.5).text("Liquid");
    g.append("text").attr("x", xScale(4.3)).attr("y", 15).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "10px").style("opacity", 0.5).text("Gas");

    // Critical lines
    [{ T: 3.42, l: "T_c" }, { T: 2.65, l: "T_m" }].forEach(({ T, l }) => {
      g.append("line").attr("x1", xScale(T)).attr("y1", 0).attr("x2", xScale(T)).attr("y2", innerH)
        .attr("stroke", accentColor).attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");
      g.append("text").attr("x", xScale(T)).attr("y", innerH + 28).attr("text-anchor", "middle")
        .style("fill", accentColor).style("font-size", "10px").style("font-weight", "bold").text(`${l}=${T}`);
    });

    // Line + points
    const sorted = [...data].sort((a, b) => a.T - b.T);
    const line = d3.line().x(d => xScale(d.T)).y(d => yScale(d.psi)).curve(d3.curveMonotoneX);
    g.append("path").datum(sorted).attr("fill", "none").attr("stroke", accentColor).attr("stroke-width", 2).attr("d", line);
    g.selectAll("circle").data(data).join("circle").attr("cx", d => xScale(d.T)).attr("cy", d => yScale(d.psi)).attr("r", 2.5).attr("fill", accentColor);

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Temperature T");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("\u03A8");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Phase Order Parameter vs Temperature");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentPhaseOrderParam;
