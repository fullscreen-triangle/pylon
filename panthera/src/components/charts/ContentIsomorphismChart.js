import React, { useRef, useEffect, useState } from "react";

const ContentIsomorphismChart = () => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 400 });

  const mappings = [
    { network: "Nodes", gas: "Molecules", value: 0.999 },
    { network: "Address Space", gas: "Volume", value: 0.999 },
    { network: "Message Rate", gas: "Temperature", value: 0.998 },
    { network: "Routing Contention", gas: "Pressure", value: 0.999 },
    { network: "Latency Dist.", gas: "Maxwell-Boltzmann", value: 0.997 },
    { network: "Congestion", gas: "Phase Transition", value: 0.996 },
    { network: "Bandwidth", gas: "Energy Partition", value: 0.999 },
    { network: "Routing Paths", gas: "Mean Free Path", value: 0.998 },
  ];

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
          setDimensions({ width, height: Math.min(450, Math.max(350, width * 0.6)) });
        }
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
    const bg = darkMode ? "#1b1b1b" : "#ffffff";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";
    const primary = "#B63E96";
    const primaryDark = "#58E6D9";
    const accentColor = darkMode ? primaryDark : primary;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", "100%")
      .attr("height", "auto")
      .style("background", bg)
      .style("border-radius", "12px");

    const barH = 22;
    const gap = (height - 80) / mappings.length;
    const centerX = width / 2;
    const barMaxW = width * 0.25;

    // Title
    svg.append("text")
      .attr("x", centerX)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Network-Gas Isomorphism: PV/NkT = 0.999");

    // Column headers
    svg.append("text")
      .attr("x", centerX - barMaxW - 30)
      .attr("y", 52)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text("Network Domain");

    svg.append("text")
      .attr("x", centerX + barMaxW + 30)
      .attr("y", 52)
      .attr("text-anchor", "middle")
      .style("fill", accentColor)
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text("Gas Domain");

    const valScale = d3.scaleLinear().domain([0, 1]).range([0, barMaxW]);

    mappings.forEach((m, i) => {
      const y = 65 + i * gap;

      // Left bar (network)
      svg.append("rect")
        .attr("x", centerX - 40 - valScale(m.value))
        .attr("y", y)
        .attr("width", 0)
        .attr("height", barH)
        .attr("rx", 4)
        .attr("fill", darkMode ? "rgba(88,230,217,0.3)" : "rgba(182,62,150,0.3)")
        .transition()
        .duration(800)
        .delay(i * 80)
        .attr("width", valScale(m.value));

      // Right bar (gas)
      svg.append("rect")
        .attr("x", centerX + 40)
        .attr("y", y)
        .attr("width", 0)
        .attr("height", barH)
        .attr("rx", 4)
        .attr("fill", darkMode ? "rgba(88,230,217,0.3)" : "rgba(182,62,150,0.3)")
        .transition()
        .duration(800)
        .delay(i * 80)
        .attr("width", valScale(m.value));

      // Connection line
      svg.append("line")
        .attr("x1", centerX - 38)
        .attr("y1", y + barH / 2)
        .attr("x2", centerX + 38)
        .attr("y2", y + barH / 2)
        .attr("stroke", accentColor)
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,3")
        .attr("opacity", 0)
        .transition()
        .duration(400)
        .delay(i * 80 + 600)
        .attr("opacity", 0.8);

      // Center equals sign
      svg.append("text")
        .attr("x", centerX)
        .attr("y", y + barH / 2 + 5)
        .attr("text-anchor", "middle")
        .style("fill", accentColor)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text("=");

      // Left label
      svg.append("text")
        .attr("x", centerX - 40 - valScale(m.value) - 5)
        .attr("y", y + barH / 2 + 4)
        .attr("text-anchor", "end")
        .style("fill", textColor)
        .style("font-size", "11px")
        .text(m.network);

      // Right label
      svg.append("text")
        .attr("x", centerX + 40 + valScale(m.value) + 5)
        .attr("y", y + barH / 2 + 4)
        .attr("text-anchor", "start")
        .style("fill", accentColor)
        .style("font-size", "11px")
        .text(m.gas);
    });
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default ContentIsomorphismChart;
