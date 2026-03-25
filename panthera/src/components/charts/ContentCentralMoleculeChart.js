import React, { useRef, useEffect, useState } from "react";

const ContentCentralMoleculeChart = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 350 });

  useEffect(() => {
    import("d3").then((d3) => { d3Ref.current = d3; renderChart(); });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) setDimensions({ width, height: Math.min(350, Math.max(250, width * 0.45)) });
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
    const margin = { top: 40, right: 30, bottom: 65, left: 100 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";

    const barData = [
      { label: "Per-Packet\nTracking", value: 69427, color: "#FF6B6B" },
      { label: "Variance\nControl", value: 1, color: darkMode ? primaryDark : primary },
    ];

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "auto")
      .style("background", bg).style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const yScale = d3.scaleBand().domain(barData.map(d => d.label)).range([0, innerH]).padding(0.4);
    const xScale = d3.scaleLog().domain([0.5, 100000]).range([0, innerW]);

    // Bars
    g.selectAll("rect.bar")
      .data(barData)
      .join("rect")
      .attr("class", "bar")
      .attr("x", 0)
      .attr("y", d => yScale(d.label))
      .attr("width", 0)
      .attr("height", yScale.bandwidth())
      .attr("rx", 6)
      .attr("fill", d => d.color)
      .attr("fill-opacity", 0.8)
      .transition().duration(1000)
      .attr("width", d => xScale(Math.max(0.5, d.value)));

    // Value labels
    g.selectAll("text.val")
      .data(barData)
      .join("text")
      .attr("class", "val")
      .attr("x", d => xScale(Math.max(0.5, d.value)) + 8)
      .attr("y", d => yScale(d.label) + yScale.bandwidth() / 2 + 5)
      .style("fill", textColor)
      .style("font-size", "13px")
      .style("font-weight", "bold")
      .text(d => d.value === 1 ? "1x (baseline)" : `${d.value.toLocaleString()}x`);

    // Y axis labels
    barData.forEach(d => {
      const lines = d.label.split("\n");
      lines.forEach((line, i) => {
        g.append("text")
          .attr("x", -10)
          .attr("y", yScale(d.label) + yScale.bandwidth() / 2 + (i - (lines.length - 1) / 2) * 14 + 4)
          .attr("text-anchor", "end")
          .style("fill", textColor)
          .style("font-size", "12px")
          .text(line);
      });
    });

    // X axis (log scale)
    g.append("g").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5, ",.0f"))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", height - 8)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "13px").text("Overhead Ratio (log scale)");
    svg.append("text").attr("x", margin.left + innerW / 2).attr("y", 22)
      .attr("text-anchor", "middle").style("fill", textColor).style("font-size", "14px").style("font-weight", "bold")
      .text("Central Molecule Impossibility: 69,427\u00D7 Overhead");

    // Ratio annotation
    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH / 2)
      .attr("text-anchor", "middle")
      .style("fill", darkMode ? primaryDark : primary)
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .style("opacity", 0.6)
      .text("69,427\u00D7 worse");
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentCentralMoleculeChart;
