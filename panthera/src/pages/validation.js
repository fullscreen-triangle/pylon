import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
};

const TestResult = ({ name, status, metric }) => {
  const isPass = status === "PASS";
  const isPartial = status === "PARTIAL";

  return (
    <motion.div
      variants={fadeInUp}
      className="flex items-center justify-between rounded-lg border border-solid
      border-dark/20 bg-light p-4 dark:border-light/20 dark:bg-dark sm:flex-col sm:items-start sm:gap-2"
    >
      <div className="flex items-center gap-3">
        <span
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold text-light
          ${isPass ? "bg-green-600" : isPartial ? "bg-yellow-500" : "bg-red-500"}`}
        >
          {isPass ? "\u2713" : isPartial ? "~" : "\u2717"}
        </span>
        <span className="text-base font-semibold text-dark dark:text-light sm:text-sm">
          {name}
        </span>
      </div>
      <div className="flex items-center gap-4 sm:ml-11 sm:gap-2">
        <span className="text-sm font-medium text-dark/60 dark:text-light/60">
          {metric}
        </span>
        <span
          className={`rounded-md px-3 py-1 text-xs font-bold uppercase tracking-wider
          ${isPass
            ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
            : isPartial
            ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400"
            : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
          }`}
        >
          {status}
        </span>
      </div>
    </motion.div>
  );
};

const paper1Tests = [
  { name: "Ideal Gas Law", status: "PASS", metric: "PV/(NkT) = 0.999" },
  { name: "Maxwell-Boltzmann Distribution", status: "PASS", metric: "KS test p > 0.05" },
  { name: "Phase Transitions", status: "PASS", metric: "T_c = 3.42 detected" },
  { name: "Van der Waals Corrections", status: "PASS", metric: "B\u2082 error = 0.0%" },
  { name: "Variance Restoration", status: "PASS", metric: "\u03C4 = 0.499 ms" },
  { name: "Uncertainty Relations", status: "PASS", metric: "\u0394E\u0394t \u2265 \u0127/2" },
  { name: "Thermodynamic Potentials", status: "PASS", metric: "All identities satisfied" },
  { name: "Central Molecule Theorem", status: "PASS", metric: "Equipartition verified" },
  { name: "Phonon Dispersion", status: "PASS", metric: "Dispersion relation matched" },
  { name: "Heat Capacity", status: "PASS", metric: "Cv/Cp ratio correct" },
];

const paper2Tests = [
  { name: "Gauge Invariance", status: "PASS", metric: "Max change 2.2\u00D710\u207B\u00B9\u2076" },
  { name: "Fiber Bundle Structure", status: "PASS", metric: "Error 2.7\u00D710\u207B\u00B9\u2076" },
  { name: "Topological Protection", status: "PASS", metric: "Winding number preserved" },
  { name: "G\u00F6delian Residue", status: "PASS", metric: "Bounded and quantified" },
  { name: "Thermodynamic Security", status: "PASS", metric: "Attack cost \u2192 \u221E" },
  { name: "Byzantine Tolerance", status: "PASS", metric: "0.51 (vs PBFT 0.34)" },
  { name: "Operational Trichotomy", status: "PASS", metric: "Three classes verified" },
  { name: "Information Geometry", status: "PASS", metric: "Fisher metric consistent" },
  { name: "Entropy Computation", status: "PASS", metric: "S-coordinates validated" },
  { name: "S-Space Clustering", status: "PASS", metric: "Clusters match topology" },
  { name: "Backward Navigation", status: "PARTIAL", metric: "O(log M) at tested scales" },
  { name: "Program Synthesis", status: "PARTIAL", metric: "Simple programs only" },
  { name: "Renormalization Group", status: "PARTIAL", metric: "First-order flow verified" },
];

export default function Validation() {
  const totalTests = paper1Tests.length + paper2Tests.length;
  const passTests = [...paper1Tests, ...paper2Tests].filter((t) => t.status === "PASS").length;
  const partialTests = [...paper1Tests, ...paper2Tests].filter((t) => t.status === "PARTIAL").length;

  return (
    <>
      <Head>
        <title>Experimental Validation | Pylon Framework</title>
        <meta
          name="description"
          content="23 independent computational experiments validate the Pylon framework. 20 tests fully pass, 3 partially validated. Results span ideal gas law verification, phase transitions, gauge invariance, and thermodynamic security."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Experimental Validation"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="mx-auto max-w-3xl text-center"
          >
            <p className="text-lg font-medium text-dark/80 dark:text-light/80 md:text-base">
              {totalTests} independent computational experiments validate predictions from both papers.
              Each test compares theoretical predictions against measured network behavior.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.5 }}
            className="mx-auto mt-10 grid max-w-2xl grid-cols-3 gap-6 md:gap-4"
          >
            <div className="flex flex-col items-center rounded-xl border-2 border-solid border-green-600 p-6
            dark:border-green-400 sm:p-4">
              <span className="text-4xl font-bold text-green-600 dark:text-green-400 md:text-3xl">
                {passTests}
              </span>
              <span className="mt-1 text-sm font-medium text-dark/70 dark:text-light/70">
                Fully Validated
              </span>
            </div>
            <div className="flex flex-col items-center rounded-xl border-2 border-solid border-yellow-500 p-6
            dark:border-yellow-400 sm:p-4">
              <span className="text-4xl font-bold text-yellow-500 dark:text-yellow-400 md:text-3xl">
                {partialTests}
              </span>
              <span className="mt-1 text-sm font-medium text-dark/70 dark:text-light/70">
                Partially Validated
              </span>
            </div>
            <div className="flex flex-col items-center rounded-xl border-2 border-solid border-dark/30 p-6
            dark:border-light/30 sm:p-4">
              <span className="text-4xl font-bold text-dark dark:text-light md:text-3xl">
                {totalTests}
              </span>
              <span className="mt-1 text-sm font-medium text-dark/70 dark:text-light/70">
                Total Tests
              </span>
            </div>
          </motion.div>

          {/* Paper 1 Results */}
          <section className="mt-20 md:mt-14">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="mb-2 text-3xl font-bold text-dark dark:text-light md:text-2xl">
                Paper 1: Equations of State
              </h2>
              <p className="mb-8 text-base font-medium text-dark/60 dark:text-light/60">
                10 tests &middot; 10 passed &middot; 0 partial
              </p>
            </motion.div>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.05 } },
              }}
              className="flex flex-col gap-3"
            >
              {paper1Tests.map((test) => (
                <TestResult
                  key={test.name}
                  name={test.name}
                  status={test.status}
                  metric={test.metric}
                />
              ))}
            </motion.div>
          </section>

          {/* Paper 2 Results */}
          <section className="mt-20 md:mt-14">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="mb-2 text-3xl font-bold text-dark dark:text-light md:text-2xl">
                Paper 2: Trajectory Completion
              </h2>
              <p className="mb-8 text-base font-medium text-dark/60 dark:text-light/60">
                13 tests &middot; 10 passed &middot; 3 partial
              </p>
            </motion.div>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.05 } },
              }}
              className="flex flex-col gap-3"
            >
              {paper2Tests.map((test) => (
                <TestResult
                  key={test.name}
                  name={test.name}
                  status={test.status}
                  metric={test.metric}
                />
              ))}
            </motion.div>
          </section>

          {/* Note on Partial Results */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-16 rounded-xl border border-solid border-yellow-500/40 bg-yellow-50/50 p-8
            dark:border-yellow-400/30 dark:bg-yellow-900/10 md:p-6"
          >
            <h3 className="text-lg font-bold text-dark dark:text-light">
              Note on Partial Validations
            </h3>
            <p className="mt-3 text-base font-medium leading-relaxed text-dark/75 dark:text-light/75">
              Three tests achieve partial validation. <strong>Backward Navigation</strong> confirms
              O(log M) scaling at tested network sizes but has not yet been validated at extreme
              scales (M &gt; 10&#8313;). <strong>Program Synthesis</strong> succeeds for simple
              programs but requires further work on compositional synthesis. <strong>Renormalization
              Group</strong> confirms first-order flow but higher-order corrections await larger-scale
              experiments. These represent active research directions rather than failures.
            </p>
          </motion.section>

          {/* Navigation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-16 flex flex-wrap items-center justify-center gap-6 sm:flex-col sm:gap-3"
          >
            <Link
              href="/state"
              className="rounded-lg border-2 border-solid bg-dark p-2.5 px-8
              text-lg font-semibold text-light hover:border-dark hover:bg-transparent
              hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
              dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
            >
              Paper 1: Equations of State
            </Link>
            <Link
              href="/trajectory"
              className="rounded-lg border-2 border-solid bg-dark p-2.5 px-8
              text-lg font-semibold text-light hover:border-dark hover:bg-transparent
              hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
              dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
            >
              Paper 2: Trajectory Completion
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
