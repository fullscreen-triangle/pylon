import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const LaptopModel = dynamic(() => import("@/components/LaptopModel"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[500px] w-full items-center justify-center lg:h-[400px] md:h-[350px] sm:h-[300px]">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-primary border-t-transparent dark:border-primaryDark dark:border-t-transparent" />
    </div>
  ),
});

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.3 },
  },
};

const ResultCard = ({ metric, title, description }) => {
  return (
    <motion.div
      variants={fadeInUp}
      className="flex flex-col items-center rounded-2xl border border-solid border-dark
      bg-light p-8 shadow-lg dark:border-light dark:bg-dark"
    >
      <span className="text-4xl font-bold text-primary dark:text-primaryDark md:text-3xl sm:text-2xl">
        {metric}
      </span>
      <h3 className="mt-3 text-xl font-bold text-dark dark:text-light md:text-lg">
        {title}
      </h3>
      <p className="mt-3 text-center text-sm font-medium text-dark/75 dark:text-light/75">
        {description}
      </p>
    </motion.div>
  );
};

const StepCard = ({ number, title, description }) => {
  return (
    <motion.div
      variants={fadeInUp}
      className="flex flex-col items-center text-center"
    >
      <span
        className="flex h-14 w-14 items-center justify-center rounded-full border-2 border-solid
        border-primary text-xl font-bold text-primary dark:border-primaryDark dark:text-primaryDark"
      >
        {number}
      </span>
      <h3 className="mt-4 text-lg font-bold text-dark dark:text-light">
        {title}
      </h3>
      <p className="mt-2 text-sm font-medium text-dark/75 dark:text-light/75">
        {description}
      </p>
    </motion.div>
  );
};

export default function Home() {
  return (
    <>
      <Head>
        <title>Pylon | Thermodynamic Network Coordination Framework</title>
        <meta
          name="description"
          content="Pylon proves distributed communication networks are mathematically identical to ideal gases, deriving equations of state, phase transitions, and thermodynamic security immune to quantum computing."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="!pt-16">
          {/* Hero Section */}
          <section className="flex w-full items-center justify-between gap-8 lg:flex-col">
            <div className="w-1/2 lg:w-full lg:text-center">
              <AnimatedText
                text="SRS Panthera"
                className="!text-left !text-8xl xl:!text-7xl lg:!text-center lg:!text-6xl md:!text-5xl"
              />
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.6 }}
                className="mt-4 text-2xl font-medium text-dark/80 dark:text-light/80 md:text-xl sm:text-lg"
              >
                Thermodynamic Network Coordination Through Gear Ratio Manifolds
              </motion.h2>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.6 }}
                className="mt-8 max-w-xl text-lg font-medium text-dark/75 dark:text-light/75 lg:mx-auto md:text-base sm:text-sm"
              >
                We prove that distributed communication networks are mathematically
                identical to ideal gases. From this single insight, we derive equations
                of state, phase transitions, and a new security model immune to quantum
                computing.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2, duration: 0.6 }}
                className="mt-10 flex items-center gap-6 lg:justify-center sm:flex-col sm:gap-4"
              >
                <Link
                  href="/publications"
                  className="flex items-center rounded-lg border-2 border-solid bg-dark p-2.5 px-6
                  text-lg font-semibold text-light hover:border-dark hover:bg-transparent
                  hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
                  dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-4 md:text-base"
                >
                  Read the Papers
                </Link>
                <Link
                  href="mailto:kundai.sachikonye@wzw.tum.de"
                  className="text-lg font-medium text-dark underline underline-offset-4
                  dark:text-light md:text-base"
                >
                  Collaborate
                </Link>
              </motion.div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }}
              className="w-1/2 lg:w-full"
            >
              <LaptopModel />
            </motion.div>
          </section>

          {/* Key Results Section */}
          <section className="mt-32 md:mt-20">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-16 text-center text-5xl font-bold text-dark dark:text-light lg:text-4xl md:text-3xl sm:mb-8"
            >
              Key Results
            </motion.h2>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid grid-cols-3 gap-10 lg:gap-6 md:grid-cols-1 md:gap-8"
            >
              <ResultCard
                metric="PV = NkT"
                title="Ideal Gas Law"
                description="Network pressure, volume, and temperature obey the same law as physical gases. Validated to 0.1% accuracy across 80 configurations."
              />
              <ResultCard
                metric="O(log M)"
                title="Backward Navigation"
                description="Message identification through backward trajectory completion, replacing O(M) forward search. Speedup exceeds 10&#8312; at scale."
              />
              <ResultCard
                metric="&#8734; Attack Cost"
                title="Thermodynamic Security"
                description="Security derived from the Second Law of Thermodynamics. Immune to P=NP and quantum computing. Zero computational overhead."
              />
            </motion.div>
          </section>

          {/* How It Works Section */}
          <section className="mt-32 md:mt-20">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-16 text-center text-5xl font-bold text-dark dark:text-light lg:text-4xl md:text-3xl sm:mb-8"
            >
              How It Works
            </motion.h2>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid grid-cols-3 gap-16 lg:gap-8 md:grid-cols-1 md:gap-12"
            >
              <StepCard
                number="1"
                title="Networks are Gases"
                description="Bounded communication networks with finite addresses, time, and nodes satisfy the same axioms as confined molecular systems. The ideal gas law emerges directly from network topology."
              />
              <StepCard
                number="2"
                title="Navigation via Gear Ratios"
                description="Transcendent observers at each scale maintain gear ratios between network parameters. Backward trajectory completion replaces exhaustive forward search with O(log M) lookups."
              />
              <StepCard
                number="3"
                title="Observation IS Computation"
                description="The triple identity collapses observation, computing, and processing into a single operation. S-entropy coordinates enable navigation without traditional computational overhead."
              />
            </motion.div>
          </section>

          {/* Preview Figures Section */}
          <section className="mt-32 md:mt-20">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-16 text-center text-5xl font-bold text-dark dark:text-light lg:text-4xl md:text-3xl sm:mb-8"
            >
              Experimental Evidence
            </motion.h2>
            <div className="grid grid-cols-2 gap-12 lg:gap-8 md:grid-cols-1">
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
                className="flex flex-col"
              >
                <Link href="/state" className="group">
                  <div className="overflow-hidden rounded-2xl border border-solid border-dark dark:border-light">
                    <Image
                      src="/images/figures/panel_1_ideal_gas_law.png"
                      alt="Ideal Gas Law Validation showing PV/(NkT) ratio across 80 network configurations"
                      width={800}
                      height={600}
                      className="h-auto w-full transition-transform duration-300 group-hover:scale-105"
                      sizes="(max-width: 768px) 100vw, 50vw"
                    />
                  </div>
                  <p className="mt-4 text-center text-base font-medium text-dark/75 dark:text-light/75">
                    <span className="font-bold text-dark dark:text-light">Ideal Gas Law Validation</span>
                    {" "}&mdash; PV/(NkT) = 0.999 across 80 configurations
                  </p>
                </Link>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
                className="flex flex-col"
              >
                <Link href="/trajectory" className="group">
                  <div className="overflow-hidden rounded-2xl border border-solid border-dark dark:border-light">
                    <Image
                      src="/images/figures/panel_5_backward_navigation.png"
                      alt="Backward Navigation Scaling showing O(log M) vs O(M) performance"
                      width={800}
                      height={600}
                      className="h-auto w-full transition-transform duration-300 group-hover:scale-105"
                      sizes="(max-width: 768px) 100vw, 50vw"
                    />
                  </div>
                  <p className="mt-4 text-center text-base font-medium text-dark/75 dark:text-light/75">
                    <span className="font-bold text-dark dark:text-light">Backward Navigation Scaling</span>
                    {" "}&mdash; O(log M) replaces O(M) with 10&#8312; speedup
                  </p>
                </Link>
              </motion.div>
            </div>
          </section>

          {/* Bottom CTA Section */}
          <section className="mt-32 md:mt-20">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="flex flex-col items-center rounded-3xl border-2 border-solid border-dark
              bg-light p-16 text-center shadow-lg dark:border-light dark:bg-dark md:p-10 sm:p-6"
            >
              <h2 className="text-4xl font-bold text-dark dark:text-light md:text-3xl sm:text-2xl">
                This research is seeking funding and collaborators
              </h2>
              <p className="mx-auto mt-6 max-w-2xl text-lg font-medium text-dark/75 dark:text-light/75 md:text-base">
                Pylon introduces a paradigm shift in distributed systems: physics-based
                coordination that replaces consensus algorithms with thermodynamic law.
                We are actively looking for research partners and funding support.
              </p>
              <Link
                href="mailto:kundai.sachikonye@wzw.tum.de"
                className="mt-8 rounded-lg border-2 border-solid bg-dark p-2.5 px-8
                text-lg font-semibold text-light hover:border-dark hover:bg-transparent
                hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
                dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
              >
                Get in Touch
              </Link>
            </motion.div>
          </section>
        </Layout>
      </main>
    </>
  );
}
