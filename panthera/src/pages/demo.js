import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";
import useNetworkMeasurement from "@/components/Hooks/useNetworkMeasurement";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const StreamingLineChart = dynamic(
  () => import("@/components/charts/StreamingLineChart"),
  { ssr: false }
);
const VarianceChart = dynamic(
  () => import("@/components/charts/VarianceChart"),
  { ssr: false }
);
const TemperatureGauge = dynamic(
  () => import("@/components/charts/TemperatureGauge"),
  { ssr: false }
);
const ComparisonBarChart = dynamic(
  () => import("@/components/charts/ComparisonBarChart"),
  { ssr: false }
);
const ScatterTimeSeries = dynamic(
  () => import("@/components/charts/ScatterTimeSeries"),
  { ssr: false }
);

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" },
  },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

const StatCard = ({ label, value, unit, accent }) => (
  <motion.div
    variants={fadeInUp}
    className="relative rounded-2xl border border-solid border-dark/20 bg-light p-6
    dark:border-light/20 dark:bg-dark"
  >
    <div className="absolute top-0 -right-2 -z-10 h-[102%] w-[101%] rounded-[1.5rem]
    rounded-br-3xl bg-dark dark:bg-light" />
    <p className="text-sm font-medium text-dark/50 dark:text-light/50">{label}</p>
    <p
      className={`mt-2 text-2xl font-bold md:text-xl ${
        accent
          ? "text-primary dark:text-primaryDark"
          : "text-dark dark:text-light"
      }`}
    >
      {value}
      {unit && (
        <span className="ml-1 text-sm font-medium text-dark/40 dark:text-light/40">
          {unit}
        </span>
      )}
    </p>
  </motion.div>
);

const SectionHeading = ({ children }) => (
  <motion.h2
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
    className="mb-6 text-3xl font-bold text-dark dark:text-light md:text-2xl"
  >
    {children}
  </motion.h2>
);

const SectionDescription = ({ children }) => (
  <motion.p
    initial={{ opacity: 0 }}
    whileInView={{ opacity: 1 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay: 0.1 }}
    className="mb-8 text-base font-medium leading-relaxed text-dark/70 dark:text-light/70 max-w-3xl"
  >
    {children}
  </motion.p>
);

export default function Demo() {
  const {
    latencyData,
    rawVarianceData,
    restoredVarianceData,
    gasLawData,
    stats,
    endpointNames,
  } = useNetworkMeasurement();

  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const checkDark = () => {
      setDarkMode(document.documentElement.classList.contains("dark"));
    };
    checkDark();
    const observer = new MutationObserver(checkDark);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  const lineColors = ["#B63E96", "#58E6D9", "#3b82f6"];

  const gaugeMaxRaw = Math.max(stats.rawTemperature * 1.5, 500);
  const gaugeMaxPylon = Math.max(gaugeMaxRaw, 500);

  return (
    <>
      <Head>
        <title>Live Network Observatory | Pylon Framework</title>
        <meta
          name="description"
          content="Live demonstration of Pylon's thermodynamic variance restoration. Watch real network measurements transformed by exponential jitter suppression in real time."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Live Network Observatory"
            className="mb-8 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-4"
          />

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mx-auto mb-16 max-w-3xl text-center text-lg font-medium leading-relaxed
            text-dark/75 dark:text-light/75 md:text-base sm:mb-8"
          >
            This page measures real network latency from your browser and demonstrates
            Pylon&apos;s core advantage: thermodynamic variance restoration. The left side shows
            raw internet jitter; the right shows how Pylon&apos;s exponential decay mechanism
            suppresses it. All data is live -- measured right now from your connection.
          </motion.p>

          {/* Pulsing live indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mb-12 flex items-center justify-center gap-3 sm:mb-8"
          >
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500 opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-green-600" />
            </span>
            <span className="text-sm font-semibold uppercase tracking-widest text-green-600 dark:text-green-400">
              Live &mdash; {stats.totalMeasurements} measurements
            </span>
          </motion.div>

          {/* ==================== Section 1: Live Latency Monitor ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="mb-20 md:mb-12"
          >
            <SectionHeading>1. Live Latency Monitor</SectionHeading>
            <SectionDescription>
              Real-time round-trip times measured from your browser to three public API
              endpoints. Each line represents a different server. Notice the natural jitter
              and variance inherent in internet communication -- this is the noise Pylon
              eliminates.
            </SectionDescription>

            <motion.div
              variants={fadeInUp}
              className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark"
            >
              <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem]
              rounded-br-3xl bg-dark dark:bg-light md:-right-2 md:w-[101%]" />
              <StreamingLineChart
                data={latencyData}
                colors={lineColors}
                labels={endpointNames}
                darkMode={darkMode}
              />
            </motion.div>
          </motion.section>

          {/* ==================== Section 2: Variance Analysis ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="mb-20 md:mb-12"
          >
            <SectionHeading>2. Variance Analysis</SectionHeading>
            <SectionDescription>
              Side-by-side comparison of raw network variance versus Pylon-restored variance.
              The raw variance (left, red) fluctuates wildly with no convergence. The
              Pylon-restored variance (right, green) decays exponentially via
              {" "}&sigma;&sup2;(t) = &sigma;&sup2;&sub;0 &times; exp(-t/&tau;) with &tau; = 0.5s,
              demonstrating active jitter suppression.
            </SectionDescription>

            <motion.div
              variants={fadeInUp}
              className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark"
            >
              <VarianceChart
                rawData={rawVarianceData}
                restoredData={restoredVarianceData}
                darkMode={darkMode}
              />
            </motion.div>

            {/* Live comparison metric */}
            <motion.div
              variants={fadeInUp}
              className="mt-6 rounded-xl border border-solid border-dark/10 bg-light/80 p-5
              text-center dark:border-light/10 dark:bg-dark/80"
            >
              <p className="text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
                Current raw variance:{" "}
                <span className="font-bold text-red-500">
                  {stats.currentRawVariance.toFixed(2)} ms&sup2;
                </span>
                {" "}&rarr; Pylon-restored variance:{" "}
                <span className="font-bold text-green-500">
                  {stats.currentRestoredVariance.toFixed(2)} ms&sup2;
                </span>
                {" "}
                <span className="font-bold text-primary dark:text-primaryDark">
                  ({stats.varianceReduction.toFixed(1)}% reduction)
                </span>
              </p>
            </motion.div>
          </motion.section>

          {/* ==================== Section 3: Network Temperature Gauge ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="mb-20 md:mb-12"
          >
            <SectionHeading>3. Network Temperature</SectionHeading>
            <SectionDescription>
              In Pylon&apos;s thermodynamic framework, network &quot;temperature&quot; is defined
              as T = m&sigma;&sup2;/k_B, directly proportional to latency variance. The raw
              network (left) runs hot with fluctuating temperature. Pylon&apos;s coordinated
              network (right) actively cools, approaching thermal equilibrium.
            </SectionDescription>

            <motion.div
              variants={fadeInUp}
              className="grid grid-cols-2 gap-8 md:grid-cols-1 md:gap-4"
            >
              <div className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark">
                <TemperatureGauge
                  value={stats.rawTemperature}
                  maxValue={gaugeMaxRaw}
                  label="Raw Network Temperature"
                  colorScheme="hot"
                  darkMode={darkMode}
                />
              </div>
              <div className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark">
                <TemperatureGauge
                  value={stats.pylonTemperature}
                  maxValue={gaugeMaxPylon}
                  label="Pylon Temperature"
                  colorScheme="cool"
                  darkMode={darkMode}
                />
              </div>
            </motion.div>
          </motion.section>

          {/* ==================== Section 4: Throughput Comparison ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="mb-20 md:mb-12"
          >
            <SectionHeading>4. Throughput Comparison</SectionHeading>
            <SectionDescription>
              Variance directly impacts throughput. Traditional TCP throughput degrades with
              jitter, while Pylon&apos;s variance restoration maintains higher effective throughput.
              The improvement factor is computed as 1/(1 + variance_ratio), reflecting how
              reduced variance translates to better network utilization.
            </SectionDescription>

            <motion.div
              variants={fadeInUp}
              className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark"
            >
              <ComparisonBarChart
                values={[
                  stats.traditionalThroughput,
                  stats.pylonThroughput,
                ]}
                labels={["Traditional TCP", "Pylon Coordinated"]}
                colors={["#ef4444", "#22c55e"]}
                darkMode={darkMode}
              />
            </motion.div>

            <motion.div
              variants={fadeInUp}
              className="mt-6 rounded-xl border border-solid border-dark/10 bg-light/80 p-5
              text-center dark:border-light/10 dark:bg-dark/80"
            >
              <p className="text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
                Improvement factor:{" "}
                <span className="font-bold text-primary dark:text-primaryDark">
                  {stats.improvementFactor.toFixed(3)}x
                </span>
                {" "}&mdash; Pylon throughput is{" "}
                <span className="font-bold text-green-500">
                  {((stats.improvementFactor - 1) * 100).toFixed(1)}%
                </span>
                {" "}higher
              </p>
            </motion.div>
          </motion.section>

          {/* ==================== Section 5: Ideal Gas Law Verification ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="mb-20 md:mb-12"
          >
            <SectionHeading>5. Ideal Gas Law Verification</SectionHeading>
            <SectionDescription>
              Pylon&apos;s core theorem: bounded networks obey PV = NkT, just like an ideal gas.
              Using your live measurements, we compute P (packet pressure), V (address space
              volume), N (active endpoints), and T (network temperature), then plot the ratio
              PV/(NkT). It should cluster around 1.0, shown by the dashed reference line.
            </SectionDescription>

            <motion.div
              variants={fadeInUp}
              className="overflow-hidden rounded-2xl border border-solid border-dark/20
              bg-light p-4 dark:border-light/20 dark:bg-dark"
            >
              <ScatterTimeSeries
                data={gasLawData}
                referenceLine={1.0}
                darkMode={darkMode}
              />
            </motion.div>

            <motion.div
              variants={fadeInUp}
              className="mt-6 rounded-xl border border-solid border-dark/10 bg-light/80 p-5
              text-center dark:border-light/10 dark:bg-dark/80"
            >
              <p className="text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
                Mean PV/(NkT) ratio:{" "}
                <span className="font-bold text-primary dark:text-primaryDark">
                  {stats.meanGasLawRatio.toFixed(4)}
                </span>
                {" "}&mdash; deviation from ideal:{" "}
                <span className="font-bold text-dark dark:text-light">
                  {Math.abs((stats.meanGasLawRatio - 1) * 100).toFixed(2)}%
                </span>
              </p>
            </motion.div>
          </motion.section>

          {/* ==================== Section 6: Statistics Summary ==================== */}
          <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
          >
            <SectionHeading>6. Session Statistics</SectionHeading>
            <SectionDescription>
              Aggregated statistics from your measurement session. These values update live
              as more data is collected. Longer sessions yield more stable estimates.
            </SectionDescription>

            <motion.div
              variants={staggerContainer}
              className="grid grid-cols-4 gap-6 lg:grid-cols-3 md:grid-cols-2 sm:grid-cols-1"
            >
              <StatCard
                label="Total Measurements"
                value={stats.totalMeasurements.toLocaleString()}
              />
              <StatCard
                label="Mean Latency"
                value={stats.meanLatency.toFixed(1)}
                unit="ms"
              />
              <StatCard
                label="Mean Raw Variance"
                value={stats.meanRawVariance.toFixed(2)}
                unit="ms\u00B2"
              />
              <StatCard
                label="Mean Restored Variance"
                value={stats.meanRestoredVariance.toFixed(2)}
                unit="ms\u00B2"
              />
              <StatCard
                label="Variance Reduction"
                value={stats.varianceReduction.toFixed(1)}
                unit="%"
                accent
              />
              <StatCard
                label="Ideal Gas Law Ratio"
                value={stats.meanGasLawRatio.toFixed(4)}
                unit="PV/NkT"
                accent
              />
              <StatCard
                label="Session Uptime"
                value={formatTime(stats.uptime)}
              />
            </motion.div>
          </motion.section>

          {/* Footer CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-20 flex flex-col items-center text-center"
          >
            <p className="text-lg font-medium text-dark/75 dark:text-light/75 md:text-base">
              This demonstration runs entirely in your browser. No data leaves your machine.
              The variance restoration algorithm is a core component of the Pylon framework.
            </p>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
