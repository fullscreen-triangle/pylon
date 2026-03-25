import React, { useRef, useEffect, useState } from "react";

const ScatterTimeSeries = ({ data, referenceLine, darkMode }) => {
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
    if (!d3 || !svgRef.current || !data || data.length === 0) return;

    const { width, height } = dimensions;
    const margin = { top: 20, right: 30, bottom: 40, left: 55 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    if (innerWidth <= 0 || innerHeight <= 0) return;

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

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(data, (d) => d.time))
      .range([0, innerWidth]);

    const yExtent = d3.extent(data, (d) => d.value);
    const yMin = Math.min(yExtent[0], (referenceLine || 1) * 0.5);
    const yMax = Math.max(yExtent[1], (referenceLine || 1) * 1.5);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerHeight, 0]);

    // Grid
    g.append("g")
      .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat("").ticks(5))
      .selectAll("line")
      .style("stroke", gridColor);
    g.select(".domain").remove();

    // Reference line
    if (referenceLine != null) {
      g.append("line")
        .attr("x1", 0)
        .attr("y1", yScale(referenceLine))
        .attr("x2", innerWidth)
        .attr("y2", yScale(referenceLine))
        .attr("stroke", "#B63E96")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "8,4");

      g.append("text")
        .attr("x", innerWidth - 5)
        .attr("y", yScale(referenceLine) - 6)
        .attr("text-anchor", "end")
        .style("fill", "#B63E96")
        .style("font-size", "11px")
        .style("font-weight", "bold")
        .text(`Ideal = ${referenceLine}`);
    }

    // Connecting line
    const lineGen = d3
      .line()
      .x((d) => xScale(d.time))
      .y((d) => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#58E6D9")
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.5)
      .attr("d", lineGen);

    // Scatter dots
    g.selectAll(".dot")
      .data(data)
      .join("circle")
      .attr("class", "dot")
      .attr("cx", (d) => xScale(d.time))
      .attr("cy", (d) => yScale(d.value))
      .attr("r", 4)
      .attr("fill", "#58E6D9")
      .attr("stroke", bg)
      .attr("stroke-width", 1.5);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat((d) => `${Math.round(d)}s`))
      .selectAll("text, line, path")
      .style("color", textColor)
      .style("stroke", textColor);

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text, line, path")
      .style("color", textColor)
      .style("stroke", textColor);

    // Y label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .text("PV / NkT");

    // X label
    g.append("text")
      .attr("y", innerHeight + 35)
      .attr("x", innerWidth / 2)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .text("Time");
  }, [data, referenceLine, dimensions, darkMode]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ScatterTimeSeries;
