import React, { useRef, useEffect, useState } from "react";

const ComparisonBarChart = ({ values, labels, colors, darkMode }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 260 });

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
          setDimensions({ width, height: Math.min(280, width * 0.5) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current || !values || values.length === 0) return;

    const { width, height } = dimensions;
    const margin = { top: 25, right: 30, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    if (innerWidth <= 0 || innerHeight <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const bg = darkMode ? "#1b1b1b" : "#f5f5f5";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";

    svg
      .attr("width", width)
      .attr("height", height)
      .style("background", bg)
      .style("border-radius", "8px");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const barLabels = labels || values.map((_, i) => `Item ${i + 1}`);
    const barColors = colors || ["#ef4444", "#22c55e", "#3b82f6"];

    const xScale = d3
      .scaleBand()
      .domain(barLabels)
      .range([0, innerWidth])
      .padding(0.35);

    const yMax = d3.max(values) * 1.2 || 100;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerHeight, 0]);

    // Grid
    g.append("g")
      .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat("").ticks(5))
      .selectAll("line")
      .style("stroke", darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)");
    g.select(".domain").remove();

    // Bars
    g.selectAll(".bar")
      .data(values)
      .join("rect")
      .attr("class", "bar")
      .attr("x", (_, i) => xScale(barLabels[i]))
      .attr("y", (d) => yScale(d))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => innerHeight - yScale(d))
      .attr("fill", (_, i) => barColors[i % barColors.length])
      .attr("rx", 4);

    // Bar value labels
    g.selectAll(".bar-label")
      .data(values)
      .join("text")
      .attr("class", "bar-label")
      .attr("x", (_, i) => xScale(barLabels[i]) + xScale.bandwidth() / 2)
      .attr("y", (d) => yScale(d) - 6)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text((d) => d.toFixed(1));

    // X axis
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .style("fill", textColor)
      .style("font-size", "11px")
      .attr("transform", "rotate(-15)")
      .attr("text-anchor", "end");

    g.select(".domain").style("stroke", textColor);

    // Y axis
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text, line, path")
      .style("color", textColor)
      .style("stroke", textColor);

    // Y label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .text("Throughput (arb. units)");
  }, [values, labels, colors, dimensions, darkMode]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ComparisonBarChart;
