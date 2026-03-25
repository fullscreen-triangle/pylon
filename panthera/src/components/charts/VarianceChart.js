import React, { useRef, useEffect, useState } from "react";

const VarianceChart = ({ rawData, restoredData, darkMode }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 280 });

  useEffect(() => {
    import("d3").then((d3) => {
      d3Ref.current = d3;
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) {
          setDimensions({ width, height: Math.min(280, width * 0.45) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current) return;
    if (!rawData || rawData.length === 0) return;

    const { width, height } = dimensions;
    const margin = { top: 25, right: 20, bottom: 40, left: 55 };
    const gap = 40;
    const panelWidth = (width - margin.left - margin.right - gap) / 2;
    const innerHeight = height - margin.top - margin.bottom;

    if (panelWidth <= 0 || innerHeight <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const bg = darkMode ? "#1b1b1b" : "#f5f5f5";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";

    svg
      .attr("width", width)
      .attr("height", height)
      .style("background", bg)
      .style("border-radius", "8px");

    const allTimes = rawData.map((d) => d.time);
    const xDomain = d3.extent(allTimes);

    const rawMax = d3.max(rawData, (d) => d.value) || 100;
    const restoredMax = d3.max(restoredData || [], (d) => d.value) || rawMax;
    const yMax = Math.max(rawMax, restoredMax) * 1.2;

    const drawPanel = (panelData, offsetX, title, color, fillColor) => {
      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left + offsetX},${margin.top})`);

      const xScale = d3.scaleLinear().domain(xDomain).range([0, panelWidth]);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerHeight, 0]);

      // Grid
      g.append("g")
        .call(d3.axisLeft(yScale).tickSize(-panelWidth).tickFormat("").ticks(4))
        .selectAll("line")
        .style("stroke", gridColor);
      g.select(".domain").remove();

      // Area
      const area = d3
        .area()
        .x((d) => xScale(d.time))
        .y0(innerHeight)
        .y1((d) => yScale(d.value))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(panelData)
        .attr("fill", fillColor)
        .attr("d", area);

      // Line
      const lineGen = d3
        .line()
        .x((d) => xScale(d.time))
        .y((d) => yScale(d.value))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(panelData)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2.5)
        .attr("d", lineGen);

      // Axes
      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(4).tickFormat((d) => `${Math.round(d)}s`))
        .selectAll("text, line, path")
        .style("color", textColor)
        .style("stroke", textColor);

      g.append("g")
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll("text, line, path")
        .style("color", textColor)
        .style("stroke", textColor);

      // Title
      g.append("text")
        .attr("x", panelWidth / 2)
        .attr("y", -8)
        .attr("text-anchor", "middle")
        .style("fill", color)
        .style("font-size", "13px")
        .style("font-weight", "bold")
        .text(title);

      // Y label
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -40)
        .attr("x", -innerHeight / 2)
        .attr("text-anchor", "middle")
        .style("fill", textColor)
        .style("font-size", "11px")
        .text("Variance (ms\u00B2)");
    };

    drawPanel(
      rawData,
      0,
      "Raw Variance",
      "#ef4444",
      darkMode ? "rgba(239,68,68,0.15)" : "rgba(239,68,68,0.1)"
    );
    drawPanel(
      restoredData || [],
      panelWidth + gap,
      "Pylon-Restored Variance",
      "#22c55e",
      darkMode ? "rgba(34,197,94,0.15)" : "rgba(34,197,94,0.1)"
    );
  }, [rawData, restoredData, dimensions, darkMode]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default VarianceChart;
