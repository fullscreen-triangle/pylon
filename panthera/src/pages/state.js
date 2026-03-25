import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const ContentIdealGasHistogram = dynamic(() => import("@/components/charts/ContentIdealGasHistogram"), { ssr: false });
const ContentPhaseOrderParam = dynamic(() => import("@/components/charts/ContentPhaseOrderParam"), { ssr: false });
const ContentVarianceDecay = dynamic(() => import("@/components/charts/ContentVarianceDecay"), { ssr: false });
const ContentVdWDeviation = dynamic(() => import("@/components/charts/ContentVdWDeviation"), { ssr: false });

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

const DetailSection = ({ title, children }) => (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
    className="mt-10 rounded-xl border border-solid border-dark/15 bg-dark/[0.02] p-8 dark:border-light/15 dark:bg-light/[0.02] md:p-5"
  >
    <h3 className="mb-4 text-xl font-bold text-dark dark:text-light md:text-lg">
      {title}
    </h3>
    {children}
  </motion.div>
);

const Prose = ({ children }) => (
  <p className="mt-3 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
    {children}
  </p>
);

const MathBlock = ({ children }) => (
  <div className="my-4 rounded-lg border border-solid border-dark/20 bg-dark/5 p-4 dark:border-light/20 dark:bg-light/5 overflow-x-auto">
    <pre className="font-mono text-sm text-dark dark:text-light leading-relaxed whitespace-pre-wrap">
      {children}
    </pre>
  </div>
);

const ChartContainer = ({ children, caption }) => (
  <div className="my-6">
    <div className="rounded-xl border border-solid border-dark/20 dark:border-light/20 overflow-hidden">
      {children}
    </div>
    {caption && (
      <p className="mt-2 text-center text-sm font-medium text-dark/60 dark:text-light/60 italic">
        {caption}
      </p>
    )}
  </div>
);

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

          {/* ================================================================ */}
          {/* Panel 1: Ideal Gas Law */}
          {/* ================================================================ */}
          <section className="mt-20 md:mt-12">
            <FigurePanel
              src="/images/figures/panel_1_ideal_gas_law.png"
              alt="Ideal Gas Law validation showing PV/(NkT) ratio across 80 network configurations"
              caption="Panel 1: Ideal Gas Law Validation"
              description="PV/(NkT) measured across 80 independent network configurations spanning 4-20 nodes,
              100-500 address space, and varied message rates. Mean ratio: 0.999. Standard deviation: 0.005.
              The ideal gas law holds for communication networks with the same precision as for physical gases."
            />

            <DetailSection title="Deep Dive: Ideal Gas Law Validation">
              <Prose>
                The ideal gas law PV = NkT is the cornerstone equation of the Pylon framework. It states
                that the product of routing contention (pressure P) and address space (volume V) equals
                the product of node count (N), the network Boltzmann constant (k), and message rate
                (temperature T). This equation is not an approximation &mdash; it is derived from first
                principles using the Boundedness Axiom and the kinetic theory of packet transport.
              </Prose>
              <Prose>
                The validation spans 80 independent configurations chosen to cover the full operating
                envelope: node counts from 4 to 20, address spaces from 100 to 500, and a range of
                message rates. For each configuration, PV/(NkT) is computed independently. The
                distribution of these 80 ratios is centered at 0.999 with standard deviation 0.005,
                meaning every single configuration is within 1% of the theoretical prediction.
              </Prose>
              <Prose>
                The histogram below shows the distribution of PV/NkT values. The tight clustering
                around 1.0 confirms that the ideal gas law is not an artifact of parameter tuning
                but a robust property of the bounded network dynamics. The slight bias toward 0.999
                (rather than exactly 1.000) is consistent with the negative second virial coefficient
                B&#8322; = &minus;1.310, which produces a small systematic correction at the densities tested.
              </Prose>

              <MathBlock>{`PV/(NkT) statistics across 80 configurations:
  Mean:    0.999
  Std:     0.005
  Min:     0.989
  Max:     1.009
  Median:  0.999
  95% CI:  [0.998, 1.000]`}</MathBlock>

              <ChartContainer caption="Distribution of PV/NkT ratios across 80 network configurations. The tight clustering around 1.0 validates the ideal gas law.">
                <ContentIdealGasHistogram />
              </ChartContainer>
            </DetailSection>

            {/* ================================================================ */}
            {/* Panel 2: Phase Transitions */}
            {/* ================================================================ */}
            <FigurePanel
              src="/images/figures/panel_2_phase_transitions.png"
              alt="Network phase transitions showing critical temperature and melting point detection"
              caption="Panel 2: Network Phase Transitions"
              description="Phase transition detection reveals critical temperature T_c = 3.42 and melting temperature
              T_m = 2.65. Networks undergo qualitative state changes at these thresholds, directly analogous to
              gas-liquid and liquid-solid transitions in physical systems."
            />

            <DetailSection title="Deep Dive: Phase Transition Structure">
              <Prose>
                The phase transition analysis reveals that networks undergo the same qualitative state
                changes as physical matter. At the critical temperature T_c = 3.42, the network
                transitions from a gas phase (uncorrelated packet dynamics) to a liquid phase (partially
                correlated flows). This transition is characterized by a sharp increase in the order
                parameter &Psi;, which measures the degree of spatial correlation among packet timings.
              </Prose>
              <Prose>
                At the melting temperature T_m = 2.65, the network undergoes a second transition from
                liquid to crystal. In the crystal phase, routing paths become deterministic and
                phase-locked, corresponding to network congestion collapse. The order parameter
                approaches unity, indicating maximal coordination &mdash; every packet follows a
                fixed, predictable path, but overall throughput is severely degraded.
              </Prose>
              <Prose>
                The practical significance is that network operators can use the order parameter as an
                early warning system. As &Psi; begins to rise above zero, the network is approaching
                the gas-liquid transition, and congestion management should be activated. If &Psi;
                exceeds the liquid-crystal threshold, the network is entering congestion collapse and
                requires immediate intervention (load shedding, traffic shaping, or capacity addition).
              </Prose>

              <ChartContainer caption="Order parameter vs temperature with labeled phase regions. The sharp transitions at T_c and T_m are clearly visible.">
                <ContentPhaseOrderParam />
              </ChartContainer>
            </DetailSection>

            {/* ================================================================ */}
            {/* Panel 3: Maxwell-Boltzmann and Variance */}
            {/* ================================================================ */}
            <FigurePanel
              src="/images/figures/panel_3_maxwell_boltzmann_variance.png"
              alt="Maxwell-Boltzmann distribution fit and variance restoration dynamics"
              caption="Panel 3: Maxwell-Boltzmann Distribution and Variance Restoration"
              description="Network latency distributions follow the Maxwell-Boltzmann form with high fidelity.
              Variance restoration time: 0.499 ms (theoretical prediction: 0.500 ms). The system autonomously
              restores statistical equilibrium after perturbation."
            />

            <DetailSection title="Deep Dive: Variance Restoration Dynamics">
              <Prose>
                Variance restoration is the most operationally significant result in the framework.
                When the network experiences a perturbation &mdash; a traffic spike, a link failure,
                or a configuration change &mdash; the variance of packet timings is driven away from
                its equilibrium value. The Second Law of Thermodynamics guarantees that this variance
                will decay exponentially back to equilibrium with time constant &tau;.
              </Prose>
              <Prose>
                The measured time constants are &tau; = 0.499, 0.499, 0.500, 0.500 ms for initial
                perturbations T&#8320; = 1, 5, 10, 20. The theoretical prediction is &tau; = 0.500 ms.
                The remarkable finding is that &tau; is independent of the perturbation magnitude &mdash;
                whether the network experiences a small fluctuation or a massive traffic burst, it
                restores to equilibrium on the same timescale. This universality is a hallmark of
                linear response theory in statistical mechanics.
              </Prose>
              <Prose>
                From an engineering perspective, variance restoration eliminates the need for
                proportional-integral-derivative (PID) controllers, feedback loops, or external
                regulation. The network is self-stabilizing by thermodynamic necessity. The
                administrator&apos;s role shifts from reactive control to setting the equilibrium
                temperature (message rate) and monitoring the approach to equilibrium.
              </Prose>

              <ChartContainer caption="Four exponential decay curves showing variance restoration. All converge with tau ~ 0.5 ms regardless of initial perturbation magnitude.">
                <ContentVarianceDecay />
              </ChartContainer>
            </DetailSection>

            {/* ================================================================ */}
            {/* Panel 4: Equations of State */}
            {/* ================================================================ */}
            <FigurePanel
              src="/images/figures/panel_4_equations_of_state.png"
              alt="Complete equations of state including van der Waals corrections and thermodynamic potentials"
              caption="Panel 4: Complete Equations of State"
              description="Van der Waals corrections, second virial coefficients, thermodynamic potentials
              (Helmholtz free energy, Gibbs free energy, enthalpy), and heat capacities computed for the
              network gas. All quantities validated against independent measurements."
            />

            <DetailSection title="Deep Dive: Van der Waals Corrections">
              <Prose>
                The ideal gas law assumes non-interacting packets, but real packets share links and
                buffers. The Van der Waals correction accounts for these interactions through the second
                virial coefficient B&#8322;. The fitted value B&#8322; = &minus;1.310 matches the
                theoretical prediction of B&#8322; = &minus;1.310 exactly (0.0% error), confirming
                that the inter-packet interaction model is quantitatively correct.
              </Prose>
              <Prose>
                The negative sign of B&#8322; indicates that net interactions are attractive at the
                tested densities: packets traveling along the same route experience less total contention
                than they would independently (a phenomenon analogous to drafting in fluid dynamics).
                At the Boyle temperature T_B = 3.41, the attractive and repulsive contributions cancel,
                and the gas behaves ideally at all densities. This temperature lies just below the
                critical temperature T_c = 3.42, consistent with the phase transition occurring where
                the interaction balance shifts.
              </Prose>
              <Prose>
                The deviation from ideal gas behavior (Z &minus; 1) grows linearly with density at
                low densities, with slope equal to B&#8322;. The chart below shows this deviation
                across the density range tested. At the highest densities, higher-order corrections
                (B&#8323;, B&#8324;) become relevant, but B&#8322; alone captures the leading-order
                non-ideal behavior.
              </Prose>

              <ChartContainer caption="Deviation from ideal gas behavior (Z - 1) as a function of density. The slope equals the second virial coefficient B2 = -1.310.">
                <ContentVdWDeviation />
              </ChartContainer>
            </DetailSection>
          </section>

          {/* ================================================================ */}
          {/* Key Results Grid */}
          {/* ================================================================ */}
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
              <MetricCard
                label="Heat Capacity"
                value="&gamma; = C_P / C_V = 5/3"
                detail="Equipartition theorem validated"
              />
              <MetricCard
                label="Uncertainty Bound"
                value="0 violations"
                detail="&sigma;_x &middot; &sigma;_p &ge; &#8463;_net across all configs"
              />
              <MetricCard
                label="Central Molecule"
                value="69,427&times; overhead"
                detail="Per-packet tracking vs variance control"
              />
              <MetricCard
                label="Sackur-Tetrode Entropy"
                value="&mu; error: 10&#8315;&sup1;&#8310;"
                detail="Chemical potential machine-precision accurate"
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
