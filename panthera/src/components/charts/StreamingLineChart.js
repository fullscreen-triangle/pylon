import React, { useRef, useEffect, useState } from "react";

const StreamingLineChart = ({ data, colors, labels, darkMode }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 300 });

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
          setDimensions({ width, height: Math.min(300, width * 0.5) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current || !data || data.length === 0) return;

    const { width, height } = dimensions;
    const margin = { top: 20, right: 120, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    if (innerWidth <= 0 || innerHeight <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const bg = darkMode ? "#1b1b1b" : "#f5f5f5";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";

    svg
      .attr("width", width)
      .attr("height", height)
      .style("background", bg)
      .style("border-radius", "8px");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const timeExtent = d3.extent(data, (d) => d.time);
    const xScale = d3
      .scaleLinear()
      .domain(timeExtent)
      .range([0, innerWidth]);

    const allValues = data.flatMap((d) => Object.values(d.values));
    const yMax = d3.max(allValues) || 100;
    const yScale = d3
      .scaleLinear()
      .domain([0, yMax * 1.15])
      .range([innerHeight, 0]);

    // Grid lines
    g.append("g")
      .attr("class", "grid")
      .call(
        d3
          .axisLeft(yScale)
          .tickSize(-innerWidth)
          .tickFormat("")
          .ticks(5)
      )
      .selectAll("line")
      .style("stroke", gridColor);
    g.select(".grid .domain").remove();

    // X axis
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .ticks(6)
          .tickFormat((d) => `${Math.round(d)}s`)
      )
      .selectAll("text, line, path")
      .style("color", textColor)
      .style("stroke", textColor);

    // Y axis
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text, line, path")
      .style("color", textColor)
      .style("stroke", textColor);

    // Y axis label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .text("Latency (ms)");

    // X axis label
    g.append("text")
      .attr("y", innerHeight + 35)
      .attr("x", innerWidth / 2)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .text("Time");

    // Lines
    const keys = labels || Object.keys(data[0]?.values || {});

    keys.forEach((key, i) => {
      const color = colors?.[i] || d3.schemeCategory10[i];

      const lineGen = d3
        .line()
        .x((d) => xScale(d.time))
        .y((d) => yScale(d.values[key] || 0))
        .curve(d3.curveMonotoneX)
        .defined((d) => d.values[key] != null);

      g.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2.5)
        .attr("d", lineGen);

      // Latest point dot
      const lastPoint = [...data].reverse().find((d) => d.values[key] != null);
      if (lastPoint) {
        g.append("circle")
          .attr("cx", xScale(lastPoint.time))
          .attr("cy", yScale(lastPoint.values[key]))
          .attr("r", 4)
          .attr("fill", color)
          .attr("stroke", bg)
          .attr("stroke-width", 2);
      }
    });

    // Legend
    const legend = svg
      .append("g")
      .attr(
        "transform",
        `translate(${width - margin.right + 10}, ${margin.top})`
      );

    keys.forEach((key, i) => {
      const color = colors?.[i] || d3.schemeCategory10[i];
      const row = legend.append("g").attr("transform", `translate(0, ${i * 22})`);
      row
        .append("rect")
        .attr("width", 14)
        .attr("height", 3)
        .attr("y", 5)
        .attr("rx", 1.5)
        .attr("fill", color);
      row
        .append("text")
        .attr("x", 20)
        .attr("y", 10)
        .style("fill", textColor)
        .style("font-size", "10px")
        .text(key.length > 12 ? key.slice(0, 12) + "..." : key);
    });
  }, [data, dimensions, colors, labels, darkMode]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default StreamingLineChart;
