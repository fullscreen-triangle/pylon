import React, { useRef, useEffect, useState } from "react";

const ContentByzantineThreshold = () => {
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
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const thermoColor = darkMode ? primaryDark : primary;
    const pbftColor = "#FF6B6B";

    // Thermodynamic: success rate stays at 1.0 up to f=0.51, then drops
    const thermoData = [];
    for (let f = 0; f <= 0.7; f += 0.01) {
      let sr;
      if (f <= 0.51) sr = 1.0;
      else sr = Math.max(0, 1.0 - (f - 0.51) / 0.15);
      thermoData.push({ f: parseFloat(f.toFixed(2)), sr: parseFloat(sr.toFixed(3)) });
    }

    // PBFT: success rate stays at 1.0 up to f=0.34, then drops
    const pbftData = [];
    for (let f = 0; f <= 0.7; f += 0.01) {
      let sr;
      if (f <= 0.34) sr = 1.0;
      else sr = Math.max(0, 1.0 - (f - 0.34) / 0.12);
      pbftData.push({ f: parseFloat(f.toFixed(2)), sr: parseFloat(sr.toFixed(3)) });
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const xScale = d3.scaleLinear().domain([0, 0.7]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1.1]).range([innerH, 0]);

    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Thermodynamic line
    const line = d3.line().x(d => xScale(d.f)).y(d => yScale(d.sr)).curve(d3.curveMonotoneX);
    g.append("path").datum(thermoData).attr("fill", "none").attr("stroke", thermoColor).attr("stroke-width", 2.5).attr("d", line);

    // PBFT line
    g.append("path").datum(pbftData).attr("fill", "none").attr("stroke", pbftColor).attr("stroke-width", 2.5).attr("stroke-dasharray", "6,3").attr("d", line);

    // Threshold markers
    [{ f: 0.51, label: "Thermo: 0.51", color: thermoColor }, { f: 0.34, label: "PBFT: 0.34", color: pbftColor }].forEach(({ f, label, color }) => {
      g.append("line").attr("x1", xScale(f)).attr("y1", 0).attr("x2", xScale(f)).attr("y2", innerH)
        .attr("stroke", color).attr("stroke-width", 1.5).attr("stroke-dasharray", "3,3").attr("opacity", 0.6);
      g.append("text").attr("x", xScale(f)).attr("y", -5).attr("text-anchor", "middle")
        .style("fill", color).style("font-size", "10px").style("font-weight", "bold").text(label);
    });

    // Advantage region
    g.append("rect").attr("x", xScale(0.34)).attr("y", 0).attr("width", xScale(0.51) - xScale(0.34)).attr("height", innerH)
      .attr("fill", thermoColor).attr("fill-opacity", 0.08);
    g.append("text").attr("x", (xScale(0.34) + xScale(0.51)) / 2).attr("y", innerH - 10)
      .attr("text-anchor", "middle").style("fill", thermoColor).style("font-size", "9px").style("opacity", 0.7).text("Advantage");

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(7).tickFormat(d3.format(".0%")))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".0%")))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Fault Fraction f");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Success Rate");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Byzantine Tolerance: Thermodynamic vs PBFT");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${width - margin.right + 8}, ${margin.top + 8})`);
    lg.append("line").attr("x1", 0).attr("y1", 5).attr("x2", 18).attr("y2", 5).attr("stroke", thermoColor).attr("stroke-width", 2);
    lg.append("text").attr("x", 22).attr("y", 9).style("fill", textColor).style("font-size", "10px").text("Thermo");
    lg.append("line").attr("x1", 0).attr("y1", 25).attr("x2", 18).attr("y2", 25).attr("stroke", pbftColor).attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
    lg.append("text").attr("x", 22).attr("y", 29).style("fill", textColor).style("font-size", "10px").text("PBFT");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentByzantineThreshold;
