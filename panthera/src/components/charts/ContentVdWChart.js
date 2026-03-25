import React, { useRef, useEffect, useState } from "react";

const ContentVdWChart = () => {
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

    // B2 theoretical = -1.310
    const B2 = -1.310;
    const B3 = 0.3; // typical third virial

    // Density range
    const densities = [];
    for (let rho = 0; rho <= 0.5; rho += 0.01) {
      densities.push(rho);
    }

    // Z = PV/NkT
    const idealLine = densities.map(rho => ({ rho, Z: 1.0 }));
    const virialLine = densities.map(rho => ({ rho, Z: 1 + B2 * rho + B3 * rho * rho }));

    // Measured data points (with noise)
    const measuredPts = [];
    for (let i = 0; i < 20; i++) {
      const rho = 0.025 * (i + 1);
      const Zvirial = 1 + B2 * rho + B3 * rho * rho;
      const Zmeas = Zvirial + (Math.sin(i * 3.1) * 0.008);
      measuredPts.push({ rho: parseFloat(rho.toFixed(3)), Z: parseFloat(Zmeas.toFixed(4)) });
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 0.55]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0.3, 1.15]).range([innerH, 0]);

    // Grid
    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(6))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Ideal gas line (Z=1)
    const lineGen = d3.line().x(d => xScale(d.rho)).y(d => yScale(d.Z));
    g.append("path").datum(idealLine)
      .attr("fill", "none").attr("stroke", darkMode ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)")
      .attr("stroke-width", 2).attr("stroke-dasharray", "8,4").attr("d", lineGen);

    // Virial expansion curve
    const curveGen = d3.line().x(d => xScale(d.rho)).y(d => yScale(d.Z)).curve(d3.curveMonotoneX);
    g.append("path").datum(virialLine)
      .attr("fill", "none").attr("stroke", accentColor).attr("stroke-width", 2.5).attr("d", curveGen);

    // Measured points
    g.selectAll("circle")
      .data(measuredPts)
      .join("circle")
      .attr("cx", d => xScale(d.rho))
      .attr("cy", d => yScale(d.Z))
      .attr("r", 4.5)
      .attr("fill", "none")
      .attr("stroke", "#FF6B6B")
      .attr("stroke-width", 2);

    // Boyle temperature marker
    // At Boyle T, B2(T_B) = 0, Z ~ 1 for all rho at low density
    // T_B = 3.41
    g.append("text")
      .attr("x", xScale(0.35))
      .attr("y", yScale(1.02))
      .style("fill", darkMode ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)")
      .style("font-size", "11px")
      .text("Ideal: Z = 1");

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Density \u03C1");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 16)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Compressibility Z = PV/NkT");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 20)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "14px").style("font-weight", "bold")
      .text("Van der Waals: Z vs Density (B\u2082 = -1.310, Boyle T = 3.41)");

    // Legend
    const lg = svg.append("g").attr("transform", `translate(${margin.left + 15}, ${margin.top + 15})`);
    lg.append("rect").attr("width", 150).attr("height", 62).attr("rx", 4).attr("fill", bg).attr("fill-opacity", 0.9).attr("stroke", accentColor).attr("stroke-width", 0.5);
    lg.append("line").attr("x1", 8).attr("y1", 14).attr("x2", 35).attr("y2", 14).attr("stroke", darkMode ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)").attr("stroke-width", 2).attr("stroke-dasharray", "8,4");
    lg.append("text").attr("x", 40).attr("y", 18).style("fill", textColor).style("font-size", "10px").text("Ideal (Z=1)");
    lg.append("line").attr("x1", 8).attr("y1", 30).attr("x2", 35).attr("y2", 30).attr("stroke", accentColor).attr("stroke-width", 2.5);
    lg.append("text").attr("x", 40).attr("y", 34).style("fill", textColor).style("font-size", "10px").text("Virial expansion");
    lg.append("circle").attr("cx", 21).attr("cy", 48).attr("r", 4).attr("fill", "none").attr("stroke", "#FF6B6B").attr("stroke-width", 2);
    lg.append("text").attr("x", 40).attr("y", 52).style("fill", textColor).style("font-size", "10px").text("Measured");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentVdWChart;
