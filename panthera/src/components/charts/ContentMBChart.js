import React, { useRef, useEffect, useState } from "react";

const ContentMBChart = () => {
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

    // MB distribution for T=1.0 (kT=1)
    const kT = 1.0;
    const mbPdf = (v) => 4 * Math.PI * Math.pow(1 / (2 * Math.PI * kT), 1.5) * v * v * Math.exp(-v * v / (2 * kT));

    // Theoretical curve
    const theoryCurve = [];
    for (let v = 0; v <= 4; v += 0.02) {
      theoryCurve.push({ v, f: mbPdf(v) });
    }

    // Simulated histogram (30 bins)
    const nBins = 30;
    const histData = [];
    const binWidth = 4.0 / nBins;
    for (let i = 0; i < nBins; i++) {
      const vCenter = (i + 0.5) * binWidth;
      const fTheory = mbPdf(vCenter);
      const fMeasured = fTheory * (1 + (Math.sin(i * 2.1) * 0.08));
      histData.push({ v: vCenter, f: parseFloat(fMeasured.toFixed(5)), binLeft: i * binWidth, binRight: (i + 1) * binWidth });
    }

    // Characteristic speeds
    const v_mp = Math.sqrt(2 * kT);  // most probable
    const v_mean = Math.sqrt(8 * kT / Math.PI);  // mean
    const v_rms = Math.sqrt(3 * kT);  // rms

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 4]).range([0, innerW]);
    const yMax = d3.max(theoryCurve, d => d.f) * 1.15;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    // Grid
    g.append("g").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5))
      .selectAll("line").style("stroke", gridColor);
    g.selectAll(".domain").remove();

    // Histogram bars
    g.selectAll("rect.bar")
      .data(histData)
      .join("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.binLeft))
      .attr("y", d => yScale(d.f))
      .attr("width", xScale(binWidth) - xScale(0) - 1)
      .attr("height", 0)
      .attr("fill", darkMode ? "rgba(88,230,217,0.35)" : "rgba(182,62,150,0.35)")
      .attr("stroke", accentColor)
      .attr("stroke-width", 0.5)
      .transition().duration(600).delay((d, i) => i * 20)
      .attr("height", d => innerH - yScale(d.f));

    // Theoretical curve
    const line = d3.line().x(d => xScale(d.v)).y(d => yScale(d.f)).curve(d3.curveMonotoneX);
    g.append("path")
      .datum(theoryCurve)
      .attr("fill", "none")
      .attr("stroke", accentColor)
      .attr("stroke-width", 2.5)
      .attr("d", line);

    // Characteristic speed lines
    const speeds = [
      { v: v_mp, label: "v_mp", color: "#FF6B6B" },
      { v: v_mean, label: "v_mean", color: "#4ECDC4" },
      { v: v_rms, label: "v_rms", color: "#FFE66D" },
    ];
    speeds.forEach(({ v, label, color }) => {
      g.append("line")
        .attr("x1", xScale(v)).attr("y1", 0).attr("x2", xScale(v)).attr("y2", innerH)
        .attr("stroke", color).attr("stroke-width", 1.5).attr("stroke-dasharray", "5,3");
      g.append("text")
        .attr("x", xScale(v) + 4).attr("y", 15)
        .style("fill", color).style("font-size", "10px").style("font-weight", "bold")
        .text(`${label} = ${v.toFixed(2)}`);
    });

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Speed v");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 16)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("f(v)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 20)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "14px").style("font-weight", "bold")
      .text("Maxwell-Boltzmann Distribution (T=1.0, KS p=0.465)");

    // Legend
    const legendG = svg.append("g").attr("transform", `translate(${width - margin.right - 130}, ${margin.top + 5})`);
    legendG.append("rect").attr("width", 120).attr("height", 42).attr("rx", 4).attr("fill", bg).attr("fill-opacity", 0.9).attr("stroke", accentColor).attr("stroke-width", 0.5);
    legendG.append("line").attr("x1", 8).attr("y1", 14).attr("x2", 28).attr("y2", 14).attr("stroke", accentColor).attr("stroke-width", 2);
    legendG.append("text").attr("x", 33).attr("y", 18).style("fill", textColor).style("font-size", "10px").text("Theory");
    legendG.append("rect").attr("x", 8).attr("y", 25).attr("width", 20).attr("height", 10).attr("fill", darkMode ? "rgba(88,230,217,0.35)" : "rgba(182,62,150,0.35)").attr("stroke", accentColor).attr("stroke-width", 0.5);
    legendG.append("text").attr("x", 33).attr("y", 34).style("fill", textColor).style("font-size", "10px").text("Measured");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentMBChart;
