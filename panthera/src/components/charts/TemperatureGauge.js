import React, { useRef, useEffect, useState } from "react";

const TemperatureGauge = ({ value, maxValue, label, colorScheme, darkMode }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const d3Ref = useRef(null);
  const [size, setSize] = useState(220);

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
          setSize(Math.min(260, Math.max(160, width)));
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const d3 = d3Ref.current;
    if (!d3 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const bg = darkMode ? "#1b1b1b" : "#f5f5f5";
    const textColor = darkMode ? "#f5f5f5" : "#1b1b1b";

    const width = size;
    const height = size * 0.75;
    const radius = size * 0.38;
    const cx = width / 2;
    const cy = height * 0.7;

    svg
      .attr("width", width)
      .attr("height", height)
      .style("background", bg)
      .style("border-radius", "8px");

    const startAngle = -Math.PI * 0.75;
    const endAngle = Math.PI * 0.75;
    const angleRange = endAngle - startAngle;

    // Background arc
    const bgArc = d3
      .arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(startAngle)
      .endAngle(endAngle)
      .cornerRadius(4);

    svg
      .append("path")
      .attr("d", bgArc())
      .attr("transform", `translate(${cx},${cy})`)
      .attr("fill", darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)");

    // Gradient arc
    const gradientId = `gauge-gradient-${label.replace(/\s/g, "-")}`;
    const defs = svg.append("defs");

    const colors =
      colorScheme === "hot"
        ? ["#22c55e", "#eab308", "#f97316", "#ef4444"]
        : ["#ef4444", "#f97316", "#22c55e", "#3b82f6"];

    const gradient = defs
      .append("linearGradient")
      .attr("id", gradientId)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");

    colors.forEach((c, i) => {
      gradient
        .append("stop")
        .attr("offset", `${(i / (colors.length - 1)) * 100}%`)
        .attr("stop-color", c);
    });

    // Value arc
    const clampedValue = Math.min(Math.max(value || 0, 0), maxValue || 1);
    const ratio = clampedValue / (maxValue || 1);
    const valueAngle = startAngle + ratio * angleRange;

    const valueArc = d3
      .arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(startAngle)
      .endAngle(valueAngle)
      .cornerRadius(4);

    svg
      .append("path")
      .attr("d", valueArc())
      .attr("transform", `translate(${cx},${cy})`)
      .attr("fill", `url(#${gradientId})`);

    // Needle
    const needleLength = radius * 0.85;
    const needleAngle = startAngle + ratio * angleRange - Math.PI / 2;
    const needleX = cx + needleLength * Math.cos(needleAngle);
    const needleY = cy + needleLength * Math.sin(needleAngle);

    svg
      .append("line")
      .attr("x1", cx)
      .attr("y1", cy)
      .attr("x2", needleX)
      .attr("y2", needleY)
      .attr("stroke", textColor)
      .attr("stroke-width", 2.5)
      .attr("stroke-linecap", "round");

    svg
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", 5)
      .attr("fill", textColor);

    // Value text
    svg
      .append("text")
      .attr("x", cx)
      .attr("y", cy + radius * 0.42)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", `${Math.max(18, size * 0.085)}px`)
      .style("font-weight", "bold")
      .text(
        value != null
          ? value >= 1000
            ? `${(value / 1000).toFixed(1)}K`
            : value.toFixed(1)
          : "--"
      );

    // Unit
    svg
      .append("text")
      .attr("x", cx)
      .attr("y", cy + radius * 0.58)
      .attr("text-anchor", "middle")
      .style("fill", darkMode ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)")
      .style("font-size", "11px")
      .text("(arb. units)");

    // Label
    svg
      .append("text")
      .attr("x", cx)
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .style("fill", textColor)
      .style("font-size", "13px")
      .style("font-weight", "bold")
      .text(label);

    // Min/Max labels
    svg
      .append("text")
      .attr("x", cx - radius * 0.85)
      .attr("y", cy + 16)
      .attr("text-anchor", "middle")
      .style("fill", darkMode ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)")
      .style("font-size", "10px")
      .text("0");

    svg
      .append("text")
      .attr("x", cx + radius * 0.85)
      .attr("y", cy + 16)
      .attr("text-anchor", "middle")
      .style("fill", darkMode ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)")
      .style("font-size", "10px")
      .text(maxValue >= 1000 ? `${(maxValue / 1000).toFixed(0)}K` : maxValue?.toFixed(0) || "100");
  }, [value, maxValue, label, colorScheme, darkMode, size]);

  return (
    <div ref={containerRef} style={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <svg ref={svgRef} />
    </div>
  );
};

export default TemperatureGauge;
