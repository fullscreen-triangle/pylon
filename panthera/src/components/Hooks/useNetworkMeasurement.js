import { useState, useEffect, useRef, useCallback } from "react";

const ENDPOINTS = [
  { name: "JSONPlaceholder", url: "https://jsonplaceholder.typicode.com/posts/1" },
  { name: "HTTPBin", url: "https://httpbin.org/get" },
  { name: "WorldTime", url: "https://worldtimeapi.org/api/timezone/Etc/UTC" },
];

const MEASUREMENT_INTERVAL = 500; // ms
const MAX_POINTS = 60;
const VARIANCE_WINDOW = 10;
const TAU = 0.5; // restoration time constant
const DT = 0.5; // measurement interval in seconds
const K_B = 1.0; // Boltzmann constant (normalized)
const PARTICLE_MASS = 1.0; // normalized particle mass
const VOLUME = 1000; // simulated address space volume

function computeRollingVariance(values, windowSize) {
  if (values.length < 2) return 0;
  const window = values.slice(-windowSize);
  const mean = window.reduce((a, b) => a + b, 0) / window.length;
  const variance =
    window.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (window.length - 1);
  return isFinite(variance) ? variance : 0;
}

export default function useNetworkMeasurement() {
  const [latencyData, setLatencyData] = useState([]);
  const [rawVarianceData, setRawVarianceData] = useState([]);
  const [restoredVarianceData, setRestoredVarianceData] = useState([]);
  const [gasLawData, setGasLawData] = useState([]);
  const [stats, setStats] = useState({
    totalMeasurements: 0,
    meanLatency: 0,
    meanRawVariance: 0,
    meanRestoredVariance: 0,
    varianceReduction: 0,
    meanGasLawRatio: 0,
    uptime: 0,
    currentRawVariance: 0,
    currentRestoredVariance: 0,
    rawTemperature: 0,
    pylonTemperature: 0,
    traditionalThroughput: 0,
    pylonThroughput: 0,
    improvementFactor: 0,
  });

  const latenciesRef = useRef({});
  const restoredVarianceRef = useRef(0);
  const startTimeRef = useRef(null);
  const measurementCountRef = useRef(0);
  const allRawVariancesRef = useRef([]);
  const allRestoredVariancesRef = useRef([]);
  const allLatenciesRef = useRef([]);
  const allGasLawRatiosRef = useRef([]);
  const intervalRef = useRef(null);

  const measure = useCallback(async () => {
    if (!startTimeRef.current) {
      startTimeRef.current = performance.now();
    }

    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    const values = {};

    const promises = ENDPOINTS.map(async (endpoint) => {
      try {
        const start = performance.now();
        await fetch(endpoint.url, {
          method: "GET",
          mode: "cors",
          cache: "no-store",
        });
        const rtt = performance.now() - start;
        values[endpoint.name] = rtt;

        if (!latenciesRef.current[endpoint.name]) {
          latenciesRef.current[endpoint.name] = [];
        }
        latenciesRef.current[endpoint.name].push(rtt);
        if (latenciesRef.current[endpoint.name].length > MAX_POINTS) {
          latenciesRef.current[endpoint.name].shift();
        }
      } catch {
        // Skip failed measurements
      }
    });

    await Promise.all(promises);

    if (Object.keys(values).length === 0) return;

    measurementCountRef.current += 1;

    // Compute mean latency across all endpoints for this tick
    const currentLatencies = Object.values(values);
    const meanCurrentLatency =
      currentLatencies.reduce((a, b) => a + b, 0) / currentLatencies.length;
    allLatenciesRef.current.push(meanCurrentLatency);

    // Compute raw variance from all endpoint latencies combined
    const allRecentLatencies = Object.values(latenciesRef.current).flat();
    const rawVariance = computeRollingVariance(
      allRecentLatencies,
      VARIANCE_WINDOW
    );

    // Apply Pylon exponential decay restoration
    const prevRestored = restoredVarianceRef.current;
    const decayFactor = Math.exp(-DT / TAU);
    const restoredVariance =
      prevRestored * decayFactor + rawVariance * (1 - decayFactor);
    restoredVarianceRef.current = restoredVariance;

    allRawVariancesRef.current.push(rawVariance);
    allRestoredVariancesRef.current.push(restoredVariance);

    // Temperature
    const rawTemp = (PARTICLE_MASS * rawVariance) / K_B;
    const pylonTemp = (PARTICLE_MASS * restoredVariance) / K_B;

    // Throughput
    const varianceRatio = rawVariance > 0 ? restoredVariance / rawVariance : 0;
    const baseThroughput = 100;
    const traditionalThroughput =
      baseThroughput / (1 + rawVariance / 1000);
    const pylonThroughput =
      baseThroughput / (1 + restoredVariance / 1000);
    const improvementFactor =
      traditionalThroughput > 0
        ? pylonThroughput / traditionalThroughput
        : 1;

    // Ideal Gas Law: PV/(NkT) should be close to 1
    const N = Object.keys(values).length;
    const T = rawTemp > 0 ? rawTemp : 1;
    const packetRate = measurementCountRef.current / Math.max(elapsed, 0.1);
    const P = packetRate * meanCurrentLatency;
    const gasLawRatio = (P * VOLUME) / (N * K_B * T);
    allGasLawRatiosRef.current.push(gasLawRatio);

    // Aggregate stats
    const avgLatency =
      allLatenciesRef.current.reduce((a, b) => a + b, 0) /
      allLatenciesRef.current.length;
    const avgRawVar =
      allRawVariancesRef.current.reduce((a, b) => a + b, 0) /
      allRawVariancesRef.current.length;
    const avgRestoredVar =
      allRestoredVariancesRef.current.reduce((a, b) => a + b, 0) /
      allRestoredVariancesRef.current.length;
    const avgGasLaw =
      allGasLawRatiosRef.current.reduce((a, b) => a + b, 0) /
      allGasLawRatiosRef.current.length;

    // Update state
    setLatencyData((prev) => {
      const next = [...prev, { time: elapsed, values }];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });

    setRawVarianceData((prev) => {
      const next = [...prev, { time: elapsed, value: rawVariance }];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });

    setRestoredVarianceData((prev) => {
      const next = [...prev, { time: elapsed, value: restoredVariance }];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });

    setGasLawData((prev) => {
      const next = [...prev, { time: elapsed, value: gasLawRatio }];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });

    setStats({
      totalMeasurements: measurementCountRef.current,
      meanLatency: avgLatency,
      meanRawVariance: avgRawVar,
      meanRestoredVariance: avgRestoredVar,
      varianceReduction:
        avgRawVar > 0 ? ((avgRawVar - avgRestoredVar) / avgRawVar) * 100 : 0,
      meanGasLawRatio: avgGasLaw,
      uptime: elapsed,
      currentRawVariance: rawVariance,
      currentRestoredVariance: restoredVariance,
      rawTemperature: rawTemp,
      pylonTemperature: pylonTemp,
      traditionalThroughput,
      pylonThroughput,
      improvementFactor,
    });
  }, []);

  useEffect(() => {
    intervalRef.current = setInterval(measure, MEASUREMENT_INTERVAL);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [measure]);

  return {
    latencyData,
    rawVarianceData,
    restoredVarianceData,
    gasLawData,
    stats,
    endpointNames: ENDPOINTS.map((e) => e.name),
  };
}
