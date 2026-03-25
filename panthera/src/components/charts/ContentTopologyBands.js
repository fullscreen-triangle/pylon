import React, { useRef, useEffect, useState } from "react";

const ContentTopologyBands = () => {
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
        if (width > 0) setDimensions({ width, height: Math.min(350, Math.max(250, width * 0.5)) });
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
    const margin = { top: 30, right: 30, bottom: 50, left: 65 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const accentColor = darkMode ? primaryDark : primary;

    // Phonon dispersion: omega^2(q) = omega_0^2 [1 - cos(qa)]
    // Two bands: acoustic and optical
    const omega0 = 1.0;
    const a = 1.0;
    const bandGap = 0.586;

    const acousticBand = [];
    const opticalBand = [];
    const nPts = 100;
    for (let i = 0; i <= nPts; i++) {
      const q = -Math.PI / a + (2 * Math.PI / a) * i / nPts;
      const qNorm = q / (Math.PI / a); // normalized to [-1, 1]
      const omega_acoustic = omega0 * Math.sqrt(Math.abs(1 - Math.cos(q * a)));
      const omega_optical = omega0 * Math.sqrt(Math.abs(1 - Math.cos(q * a))) + bandGap + 0.5;
      acousticBand.push({ q: qNorm, omega: omega_acoustic });
      opticalBand.push({ q: qNorm, omega: omega_optical });
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([-1.1, 1.1]).range([0, innerW]);
    const yMax = d3.max(opticalBand, d => d.omega) * 1.1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(6))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Band gap shading
    const acousticMax = d3.max(acousticBand, d => d.omega);
    const opticalMin = d3.min(opticalBand, d => d.omega);
    g.append("rect")
      .attr("x", 0).attr("y", yScale(opticalMin))
      .attr("width", innerW).attr("height", yScale(acousticMax) - yScale(opticalMin))
      .attr("fill", "#FF6B6B").attr("fill-opacity", 0.1);

    g.append("text")
      .attr("x", innerW - 5).attr("y", (yScale(opticalMin) + yScale(acousticMax)) / 2 + 4)
      .attr("text-anchor", "end")
      .style("fill", "#FF6B6B").style("font-size", "10px").style("font-weight", "bold")
      .text(`Band gap = ${bandGap}`);

    // Acoustic band
    const line = d3.line().x(d => xScale(d.q)).y(d => yScale(d.omega)).curve(d3.curveMonotoneX);
    g.append("path").datum(acousticBand).attr("fill", "none").attr("stroke", accentColor).attr("stroke-width", 2.5).attr("d", line);

    // Optical band
    g.append("path").datum(opticalBand).attr("fill", "none").attr("stroke", "#4ECDC4").attr("stroke-width", 2.5).attr("d", line);

    // Annotations
    g.append("text").attr("x", xScale(0.5)).attr("y", yScale(0.6))
      .style("fill", accentColor).style("font-size", "10px").text("Acoustic");
    g.append("text").attr("x", xScale(0.5)).attr("y", yScale(opticalMin + 0.4))
      .style("fill", "#4ECDC4").style("font-size", "10px").text("Optical");

    // Berry phase annotation
    g.append("text").attr("x", xScale(-0.8)).attr("y", yScale(yMax * 0.95))
      .style("fill", textColor).style("font-size", "10px").style("opacity", 0.7)
      .text("Winding # = 1, Berry \u03C6 = \u03C0");

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d === 0 ? "\u0393" : d === -1 ? "-\u03C0/a" : d === 1 ? "\u03C0/a" : ""))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Wavevector q");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Frequency \u03C9");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18).attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold").text("Phonon Band Structure (Topological Protection)");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentTopologyBands;
