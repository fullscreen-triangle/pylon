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

export default function State() {
  return (
    <>
      <Head>
        <title>Equations of State for Transcendent Observer Networks | Pylon</title>
        <meta
          name="description"
          content="Paper 1: Equations of State for Transcendent Observer Networks. Proving PV=NkT for communication networks with 0.1% accuracy across 80 configurations. Phase transitions, Maxwell-Boltzmann distributions, and thermodynamic potentials."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Equations of State"
            className="!text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl"
          />

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="mt-4 text-center"
          >
            <h2 className="text-xl font-medium text-dark/80 dark:text-light/80 md:text-lg sm:text-base">
              For Transcendent Observer Networks
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
              We establish that bounded communication networks satisfy the axioms of confined
              molecular systems and derive complete equations of state governing their macroscopic
              behavior. The ideal gas law PV = NkT emerges directly from network boundedness and
              is validated to 0.1% accuracy. We identify network phase transitions at critical
              temperatures, derive Maxwell-Boltzmann latency distributions, and compute all
              thermodynamic potentials for network systems.
            </p>
          </motion.section>

          {/* Figure Panels */}
          <section className="mt-20 md:mt-12">
            <FigurePanel
              src="/images/figures/panel_1_ideal_gas_law.png"
              alt="Ideal Gas Law validation showing PV/(NkT) ratio across 80 network configurations"
              caption="Panel 1: Ideal Gas Law Validation"
              description="PV/(NkT) measured across 80 independent network configurations spanning 4-20 nodes,
              100-500 address space, and varied message rates. Mean ratio: 0.999. Standard deviation: 0.005.
              The ideal gas law holds for communication networks with the same precision as for physical gases."
            />

            <FigurePanel
              src="/images/figures/panel_2_phase_transitions.png"
              alt="Network phase transitions showing critical temperature and melting point detection"
              caption="Panel 2: Network Phase Transitions"
              description="Phase transition detection reveals critical temperature T_c = 3.42 and melting temperature
              T_m = 2.65. Networks undergo qualitative state changes at these thresholds, directly analogous to
              gas-liquid and liquid-solid transitions in physical systems."
            />

            <FigurePanel
              src="/images/figures/panel_3_maxwell_boltzmann_variance.png"
              alt="Maxwell-Boltzmann distribution fit and variance restoration dynamics"
              caption="Panel 3: Maxwell-Boltzmann Distribution and Variance Restoration"
              description="Network latency distributions follow the Maxwell-Boltzmann form with high fidelity.
              Variance restoration time: 0.499 ms (theoretical prediction: 0.500 ms). The system autonomously
              restores statistical equilibrium after perturbation."
            />

            <FigurePanel
              src="/images/figures/panel_4_equations_of_state.png"
              alt="Complete equations of state including van der Waals corrections and thermodynamic potentials"
              caption="Panel 4: Complete Equations of State"
              description="Van der Waals corrections, second virial coefficients, thermodynamic potentials
              (Helmholtz free energy, Gibbs free energy, enthalpy), and heat capacities computed for the
              network gas. All quantities validated against independent measurements."
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
                label="Ideal Gas Law"
                value="PV/(NkT) = 0.999 &plusmn; 0.005"
                detail="Validated across 80 configurations"
              />
              <MetricCard
                label="Variance Restoration"
                value="&tau; = 0.499 ms"
                detail="Theory: 0.500 ms (0.2% error)"
              />
              <MetricCard
                label="Second Virial Coefficient"
                value="B&#8322; error = 0.0%"
                detail="Van der Waals correction exact"
              />
              <MetricCard
                label="Phase Transitions"
                value="T_c = 3.42, T_m = 2.65"
                detail="Critical and melting temperatures detected"
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
              href="/trajectory"
              className="rounded-lg border-2 border-solid bg-dark p-2.5 px-8
              text-lg font-semibold text-light hover:border-dark hover:bg-transparent
              hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
              dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
            >
              Continue to Paper 2: Trajectory Completion
            </Link>
            <Link
              href="/validation"
              className="text-base font-medium text-dark underline underline-offset-4
              dark:text-light"
            >
              View All Experimental Results
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
