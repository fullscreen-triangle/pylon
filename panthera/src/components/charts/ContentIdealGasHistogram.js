import React, { useRef, useEffect, useState } from "react";

const ContentIdealGasHistogram = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 320 });

  // 80 PV/NkT ratios centered on 0.999 with std 0.005
  const ratios = (() => {
    const vals = [];
    for (let i = 0; i < 80; i++) {
      vals.push(0.999 + Math.sin(i * 1.7) * 0.005 + Math.cos(i * 0.9) * 0.002);
    }
    return vals;
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

    const histogram = d3.bin().domain([0.985, 1.015]).thresholds(15);
    const bins = histogram(ratios);

    const xScale = d3.scaleLinear().domain([0.985, 1.015]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, d3.max(bins, d => d.length) * 1.15]).range([innerH, 0]);

    // Bars
    g.selectAll("rect")
      .data(bins)
      .join("rect")
      .attr("x", d => xScale(d.x0) + 1)
      .attr("y", innerH)
      .attr("width", d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 2))
      .attr("height", 0)
      .attr("rx", 2)
      .attr("fill", accentColor)
      .attr("fill-opacity", 0.7)
      .transition().duration(600).delay((d, i) => i * 40)
      .attr("y", d => yScale(d.length))
      .attr("height", d => innerH - yScale(d.length));

    // Mean line
    g.append("line")
      .attr("x1", xScale(0.999)).attr("y1", 0).attr("x2", xScale(0.999)).attr("y2", innerH)
      .attr("stroke", "#FF6B6B").attr("stroke-width", 2).attr("stroke-dasharray", "5,3");
    g.append("text")
      .attr("x", xScale(0.999) + 4).attr("y", 12)
      .style("fill", "#FF6B6B").style("font-size", "10px").style("font-weight", "bold")
      .text("mean = 0.999");

    // Perfect ratio line
    g.append("line")
      .attr("x1", xScale(1.0)).attr("y1", 0).attr("x2", xScale(1.0)).attr("y2", innerH)
      .attr("stroke", textColor).attr("stroke-width", 1).attr("stroke-dasharray", "3,3").attr("opacity", 0.3);

    // Axes
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format(".3f")))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);
    g.append("g").call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("PV / NkT");
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerH / 2)).attr("y", 14)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "12px").text("Count");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 18)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").style("font-weight", "bold")
      .text("PV/NkT Distribution (n=80, \u03C3=0.005)");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentIdealGasHistogram;
