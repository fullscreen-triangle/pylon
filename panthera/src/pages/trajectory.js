import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
};

const FigurePanel = ({ src, alt, caption, description }) => {
  return (
    <motion.figure
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="mt-16 first:mt-0 md:mt-10"
    >
      <div className="overflow-hidden rounded-2xl border border-solid border-dark dark:border-light">
        <Image
          src={src}
          alt={alt}
          width={1200}
          height={800}
          className="h-auto w-full"
          sizes="100vw"
        />
      </div>
      <figcaption className="mt-4">
        <p className="text-lg font-bold text-dark dark:text-light md:text-base">
          {caption}
        </p>
        <p className="mt-1 text-sm font-medium text-dark/70 dark:text-light/70">
          {description}
        </p>
      </figcaption>
    </motion.figure>
  );
};

const MetricCard = ({ label, value, detail }) => {
  return (
    <motion.div
      variants={fadeInUp}
      className="flex flex-col rounded-xl border border-solid border-dark/30 bg-light p-6
      dark:border-light/30 dark:bg-dark"
    >
      <span className="text-sm font-medium uppercase tracking-wider text-dark/60 dark:text-light/60">
        {label}
      </span>
      <span className="mt-2 text-2xl font-bold text-primary dark:text-primaryDark md:text-xl">
        {value}
      </span>
      <span className="mt-1 text-sm font-medium text-dark/70 dark:text-light/70">
        {detail}
      </span>
    </motion.div>
  );
};

export default function Trajectory() {
  return (
    <>
      <Head>
        <title>Backward Trajectory Completion on Gear Ratio Manifolds | Pylon</title>
        <meta
          name="description"
          content="Paper 2: Backward Trajectory Completion on Gear Ratio Manifolds. O(log M) navigation replacing O(M) search, thermodynamic security immune to quantum computing, and gauge-invariant fiber bundle architecture."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Trajectory Completion"
            className="!text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl"
          />

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="mt-4 text-center"
          >
            <h2 className="text-xl font-medium text-dark/80 dark:text-light/80 md:text-lg sm:text-base">
              Backward Trajectory Completion on Gear Ratio Manifolds
            </h2>
            <p className="mt-2 text-base font-medium text-primary dark:text-primaryDark">
              K.F. Sachikonye &middot; Technical University of Munich
            </p>
          </motion.div>

          {/* Abstract */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.5 }}
            className="mx-auto mt-12 max-w-3xl"
          >
            <h3 className="mb-4 text-lg font-bold text-dark dark:text-light">Abstract</h3>
            <p className="text-base font-medium leading-relaxed text-dark/80 dark:text-light/80">
              Building on the thermodynamic equations of state for communication networks, we
              develop backward trajectory completion on gear ratio manifolds. Transcendent observers
              at each scale maintain gear ratios that enable O(log M) message identification,
              replacing O(M) forward search with speedups exceeding 10&#8312; at scale. We derive
              thermodynamic security guarantees from the Second Law, prove gauge invariance of
              the fiber bundle architecture, and demonstrate Byzantine tolerance of 0.51 &mdash;
              exceeding the PBFT threshold of 0.34.
            </p>
          </motion.section>

          {/* Figure Panels */}
          <section className="mt-20 md:mt-12">
            <FigurePanel
              src="/images/figures/panel_5_backward_navigation.png"
              alt="Backward navigation scaling comparison showing O(log M) vs O(M) performance"
              caption="Panel 5: Backward Navigation Scaling"
              description="Comparison of forward search O(M) versus backward trajectory completion O(log M).
              At M = 10&#8313; messages, backward completion achieves a speedup factor exceeding 10&#8312;.
              The logarithmic scaling is maintained across all tested network sizes and configurations."
            />

            <FigurePanel
              src="/images/figures/panel_6_security_byzantine.png"
              alt="Thermodynamic security analysis and Byzantine fault tolerance measurements"
              caption="Panel 6: Thermodynamic Security and Byzantine Tolerance"
              description="Security derived from the Second Law of Thermodynamics. Attack cost grows exponentially
              with network size, becoming effectively infinite at operational scales. Byzantine tolerance
              measured at 0.51, exceeding the PBFT threshold of 0.34. Detection rate: 100%, false positive rate: 0%."
            />

            <FigurePanel
              src="/images/figures/panel_7_geometry_topology.png"
              alt="Geometric and topological properties of the gear ratio manifold"
              caption="Panel 7: Geometry and Topology"
              description="The gear ratio manifold exhibits fiber bundle structure with gauge-invariant connections.
              Topological protection ensures that navigation paths are robust against local perturbations.
              Gauge invariance: maximum change 2.2 &times; 10&#8315;&sup1;&#8310; under coordinate transformation."
            />

            <FigurePanel
              src="/images/figures/panel_8_residue_entropy.png"
              alt="Godelian residue structure and entropy computation in S-space"
              caption="Panel 8: G&ouml;delian Residue and Entropy"
              description="S-entropy coordinates partition the network state space into computable and incomputable
              regions. The G&ouml;delian residue &mdash; the irreducibly incomputable fraction &mdash; is bounded
              and quantified. Entropy computation validates the thermodynamic consistency of the framework."
            />
          </section>

          {/* Key Results Grid */}
          <section className="mt-20 md:mt-12">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-10 text-3xl font-bold text-dark dark:text-light md:text-2xl"
            >
              Quantitative Results
            </motion.h2>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
              }}
              className="grid grid-cols-2 gap-6 md:grid-cols-1"
            >
              <MetricCard
                label="Gauge Invariance"
                value="Max change: 2.2 &times; 10&#8315;&sup1;&#8310;"
                detail="Under arbitrary coordinate transformation"
              />
              <MetricCard
                label="Fiber Bundle Transitivity"
                value="Error: 2.7 &times; 10&#8315;&sup1;&#8310;"
                detail="Machine-precision structural consistency"
              />
              <MetricCard
                label="Byzantine Tolerance"
                value="0.51 (vs PBFT 0.34)"
                detail="50% improvement over state of the art"
              />
              <MetricCard
                label="Intrusion Detection"
                value="100% detection, 0% false positives"
                detail="Thermodynamic anomaly detection"
              />
            </motion.div>
          </section>

          {/* Navigation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-20 flex flex-col items-center gap-4 sm:gap-3"
          >
            <Link
              href="/validation"
              className="rounded-lg border-2 border-solid bg-dark p-2.5 px-8
              text-lg font-semibold text-light hover:border-dark hover:bg-transparent
              hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
              dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
            >
              View All Experimental Results
            </Link>
            <Link
              href="/state"
              className="text-base font-medium text-dark underline underline-offset-4
              dark:text-light"
            >
              Back to Paper 1: Equations of State
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
