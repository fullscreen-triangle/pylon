import React, { useRef, useEffect, useState } from "react";

const ContentBoundsChart = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const tooltipRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 420 });

  // 80 ideal gas law configurations: (N, V, T) with PV/NkT ratio
  const data = (() => {
    const points = [];
    const nodeRange = [4, 6, 8, 10, 12, 14, 16, 18, 20];
    const volRange = [100, 150, 200, 250, 300, 350, 400, 450, 500];
    let idx = 0;
    for (let i = 0; i < nodeRange.length && idx < 80; i++) {
      for (let j = 0; j < volRange.length && idx < 80; j++) {
        const N = nodeRange[i];
        const V = volRange[j];
        const T = 1.0 + (idx * 0.05);
        const ratio = 0.999 + (Math.sin(idx * 1.7) * 0.005);
        points.push({ N, V, T, ratio: parseFloat(ratio.toFixed(4)), idx });
        idx++;
      }
    }
    return points;
  })();

  useEffect(() => {
    import("d3").then((d3) => {
      d3Ref.current = d3;
      renderChart();
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) {
          setDimensions({ width, height: Math.min(420, Math.max(300, width * 0.55)) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    renderChart();
  }, [dimensions]);

  function renderChart() {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current) return;

    const darkMode = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
    const { width, height } = dimensions;
    const margin = { top: 30, right: 30, bottom: 55, left: 65 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const gridColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const pointColor = darkMode ? primaryDark : primary;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", "100%")
      .attr("height", "auto")
      .style("background", bg)
      .style("border-radius", "12px");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales: x = V (address space), y = N (nodes)
    const xScale = d3.scaleLinear().domain([50, 550]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([2, 22]).range([innerH, 0]);
    const rScale = d3.scaleLinear().domain([1.0, 5.0]).range([4, 12]);

    // Boundedness boundary rectangle
    g.append("rect")
      .attr("x", xScale(100))
      .attr("y", yScale(20))
      .attr("width", xScale(500) - xScale(100))
      .attr("height", yScale(4) - yScale(20))
      .attr("fill", "none")
      .attr("stroke", darkMode ? primaryDark : primary)
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "8,4")
      .attr("rx", 8);

    // Energy shells (concentric ellipses)
    const shells = [
      { cx: 300, cy: 12, rx: 180, ry: 7, label: "High energy shell" },
      { cx: 300, cy: 12, rx: 120, ry: 5, label: "Mid energy shell" },
      { cx: 300, cy: 12, rx: 60, ry: 3, label: "Low energy shell" },
    ];
    shells.forEach((s, i) => {
      g.append("ellipse")
        .attr("cx", xScale(s.cx))
        .attr("cy", yScale(s.cy))
        .attr("rx", (xScale(s.cx + s.rx) - xScale(s.cx)))
        .attr("ry", Math.abs(yScale(s.cy + s.ry) - yScale(s.cy)))
        .attr("fill", "none")
        .attr("stroke", darkMode ? `rgba(88,230,217,${0.15 + i * 0.1})` : `rgba(182,62,150,${0.15 + i * 0.1})`)
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,4");
    });

    // Gridlines
    g.append("g")
      .call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(6))
      .selectAll("line").style("stroke", gridColor);
    g.select(".domain").remove();

    // Data points
    const tooltip = d3.select(tooltipRef.current);

    g.selectAll("circle.point")
      .data(data)
      .join("circle")
      .attr("class", "point")
      .attr("cx", (d) => xScale(d.V))
      .attr("cy", (d) => yScale(d.N))
      .attr("r", 0)
      .attr("fill", pointColor)
      .attr("fill-opacity", 0.7)
      .attr("stroke", pointColor)
      .attr("stroke-width", 1)
      .attr("cursor", "pointer")
      .on("mouseover", function (event, d) {
        d3.select(this).attr("fill-opacity", 1).attr("r", 8);
        tooltip
          .style("display", "block")
          .style("left", `${event.offsetX + 12}px`)
          .style("top", `${event.offsetY - 40}px`)
          .html(`<strong>Config #${d.idx + 1}</strong><br/>N=${d.N}, V=${d.V}<br/>T=${d.T.toFixed(2)}<br/>PV/NkT = ${d.ratio}`);
      })
      .on("mouseout", function () {
        d3.select(this).attr("fill-opacity", 0.7).attr("r", (d) => rScale(d.T));
        tooltip.style("display", "none");
      })
      .transition()
      .duration(800)
      .delay((d, i) => i * 10)
      .attr("r", (d) => rScale(d.T));

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(6))
      .selectAll("text,line,path").style("color", textColor).style("stroke", textColor);

    // Labels
    svg.append("text")
      .attr("x", margin.left + innerW / 2)
      .attr("y", height - 8)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "13px")
      .text("Address Space V");

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -(margin.top + innerH / 2))
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "13px")
      .text("Node Count N");

    // Title
    svg.append("text")
      .attr("x", margin.left + innerW / 2)
      .attr("y", 18)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Bounded Phase Space: 80 Configurations (point size = temperature)");

    // Boundary label
    g.append("text")
      .attr("x", xScale(500) + 5)
      .attr("y", yScale(20) - 5)
      .style("fill", pointColor)
      .style("font-size", "11px")
      .text("Bounded region");
  }

  return (
    <div ref={containerRef} style={{ width: "100%", position: "relative" }}>
      <svg ref={svgRef} />
      <div
        ref={tooltipRef}
        style={{
          display: "none",
          position: "absolute",
          background: "rgba(0,0,0,0.85)",
          color: "#fff",
          padding: "8px 12px",
          borderRadius: "6px",
          fontSize: "12px",
          pointerEvents: "none",
          lineHeight: "1.5",
          zIndex: 10,
        }}
      />
    </div>
  );
};

export default ContentBoundsChart;
