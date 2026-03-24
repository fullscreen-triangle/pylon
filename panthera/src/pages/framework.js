import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
};

const Section = ({ title, children, delay = 0 }) => {
  return (
    <motion.section
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      className="mt-20 first:mt-0 md:mt-14"
    >
      <h2 className="mb-8 text-3xl font-bold text-dark dark:text-light md:text-2xl sm:text-xl">
        {title}
      </h2>
      {children}
    </motion.section>
  );
};

const MappingRow = ({ network, gas }) => {
  return (
    <tr className="border-b border-dark/20 dark:border-light/20">
      <td className="py-3 pr-8 text-base font-medium text-dark dark:text-light sm:text-sm sm:pr-4">
        {network}
      </td>
      <td className="py-3 text-base font-medium text-primary dark:text-primaryDark sm:text-sm">
        {gas}
      </td>
    </tr>
  );
};

export default function Framework() {
  return (
    <>
      <Head>
        <title>Theoretical Foundations | Pylon Framework</title>
        <meta
          name="description"
          content="The theoretical foundations of the Pylon framework: the Boundedness Axiom, network-gas isomorphism, transcendent observer architecture, and the triple identity of observation, computing, and processing."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Theoretical Foundations"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Section 1: The Boundedness Axiom */}
          <Section title="1. The Boundedness Axiom">
            <p className="text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              Every physical communication network is bounded. Address space V is finite.
              Observation time T is finite. Node count N is finite. These are not approximations
              or simplifying assumptions &mdash; they are physical facts about any real network.
            </p>
            <p className="mt-4 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              From boundedness, Poincar&eacute; recurrence follows immediately: any state-space
              trajectory in a bounded system must return arbitrarily close to its initial condition.
              This guarantees oscillatory dynamics and makes the network amenable to the full
              apparatus of statistical mechanics.
            </p>
            <div className="mt-6 rounded-xl border border-solid border-dark/30 bg-dark/5 p-6
            dark:border-light/30 dark:bg-light/5">
              <p className="font-mono text-sm text-dark dark:text-light">
                Axiom: For every network N, there exist finite bounds V, T, N such that
                all addresses a &isin; [0, V], all times t &isin; [0, T], and all nodes
                n &isin; [1, N].
              </p>
            </div>
          </Section>

          {/* Section 2: The Network-Gas Isomorphism */}
          <Section title="2. The Network-Gas Isomorphism">
            <p className="text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              This is not a metaphor. It is not an analogy. It is a mathematical identity. Bounded
              communication networks satisfy exactly the same axioms as confined ideal gas systems.
              Every theorem that holds for ideal gases holds, with quantitative precision, for
              communication networks.
            </p>
            <div className="mt-8 overflow-x-auto">
              <table className="w-full max-w-2xl">
                <thead>
                  <tr className="border-b-2 border-dark/40 dark:border-light/40">
                    <th className="pb-3 pr-8 text-left text-lg font-bold text-dark dark:text-light sm:text-base sm:pr-4">
                      Network Domain
                    </th>
                    <th className="pb-3 text-left text-lg font-bold text-primary dark:text-primaryDark sm:text-base">
                      Gas Domain
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <MappingRow network="Nodes" gas="Molecules" />
                  <MappingRow network="Address space (V)" gas="Volume" />
                  <MappingRow network="Message rate" gas="Temperature" />
                  <MappingRow network="Routing contention" gas="Pressure" />
                  <MappingRow network="Latency distribution" gas="Maxwell-Boltzmann distribution" />
                  <MappingRow network="Network congestion" gas="Phase transition" />
                  <MappingRow network="Bandwidth allocation" gas="Energy partition" />
                  <MappingRow network="Routing paths" gas="Mean free path" />
                </tbody>
              </table>
            </div>
            <p className="mt-6 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              Validated to 0.1% accuracy. PV/(NkT) = 0.999 &plusmn; 0.005 across 80 independent
              network configurations. Second virial coefficient error: 0.0%.
            </p>
          </Section>

          {/* Section 3: Transcendent Observer Architecture */}
          <Section title="3. Transcendent Observer Architecture">
            <p className="text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              A transcendent observer operates at scale s and monitors scale s&minus;1. It maintains
              a gear ratio &gamma;(s) = V(s)/V(s&minus;1) between adjacent scales. Navigation across
              scales requires only O(1) operations per level, giving O(log M) total navigation cost
              for M messages.
            </p>
            <p className="mt-4 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              The architecture uses eight scales, from individual message bytes to global network
              topology. Each scale preserves gauge invariance: physical observables remain unchanged
              under coordinate transformations within the fiber bundle structure.
            </p>
            <div className="mt-6 rounded-xl border border-solid border-dark/30 bg-dark/5 p-6
            dark:border-light/30 dark:bg-light/5">
              <pre className="overflow-x-auto font-mono text-sm text-dark dark:text-light leading-relaxed">
{`Scale 8: Global topology     (network-wide)
Scale 7: Cluster dynamics    (subnet groups)
Scale 6: Subnet structure    (local regions)
Scale 5: Flow patterns       (traffic classes)
Scale 4: Session state       (connections)
Scale 3: Message sequences   (streams)
Scale 2: Packet structure    (datagrams)
Scale 1: Byte-level content  (payload)`}
              </pre>
            </div>
            <p className="mt-4 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              Gauge invariance validated: maximum change under coordinate transformation
              = 2.2 &times; 10&#8315;&sup1;&#8310;. Fiber bundle transitivity error: 2.7 &times; 10&#8315;&sup1;&#8310;.
            </p>
          </Section>

          {/* Section 4: The Triple Identity */}
          <Section title="4. The Triple Identity">
            <p className="text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              Pylon collapses three traditionally separate operations into one:
            </p>
            <div className="mt-6 flex flex-col gap-4">
              <div className="rounded-lg border border-solid border-primary/30 p-5 dark:border-primaryDark/30">
                <h3 className="font-bold text-primary dark:text-primaryDark">Observation</h3>
                <p className="mt-1 text-sm font-medium text-dark/75 dark:text-light/75">
                  Reading the state of a network element at its current coordinates.
                </p>
              </div>
              <div className="rounded-lg border border-solid border-primary/30 p-5 dark:border-primaryDark/30">
                <h3 className="font-bold text-primary dark:text-primaryDark">Computing</h3>
                <p className="mt-1 text-sm font-medium text-dark/75 dark:text-light/75">
                  Deriving the next state from thermodynamic equations of motion.
                </p>
              </div>
              <div className="rounded-lg border border-solid border-primary/30 p-5 dark:border-primaryDark/30">
                <h3 className="font-bold text-primary dark:text-primaryDark">Processing</h3>
                <p className="mt-1 text-sm font-medium text-dark/75 dark:text-light/75">
                  Transforming input to output through backward trajectory completion.
                </p>
              </div>
            </div>
            <p className="mt-6 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
              Backward trajectory completion works by starting from a target state and computing
              the unique history that produced it. In S-entropy coordinates, this completion is
              deterministic and requires no search. The observation of a network element simultaneously
              computes and processes it.
            </p>
          </Section>

          {/* Section 5: System Architecture */}
          <Section title="5. System Architecture">
            <p className="text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed mb-6">
              The Pylon framework is organized as a hierarchical component system. Each layer
              builds on the thermodynamic foundations established by the layers below.
            </p>
            <div className="rounded-xl border border-solid border-dark/30 bg-dark/5 p-6
            dark:border-light/30 dark:bg-light/5">
              <pre className="overflow-x-auto font-mono text-sm text-dark dark:text-light leading-relaxed">
{`pylon/
  core/
    thermodynamics/          # Equations of state, phase transitions
      ideal_gas.py           # PV = NkT for networks
      van_der_waals.py       # Non-ideal corrections (B2, B3)
      maxwell_boltzmann.py   # Latency distributions
      phase_transitions.py   # Critical temperature detection
    observers/               # Transcendent observer hierarchy
      gear_ratios.py         # Scale coupling coefficients
      fiber_bundle.py        # Gauge-invariant coordinates
      s_entropy.py           # Entropy-based navigation
    trajectory/              # Backward trajectory completion
      completion.py          # O(log M) backward navigation
      trichotomy.py          # Operational classification
      synthesis.py           # Program synthesis from traces
  security/
    thermodynamic.py         # Second Law security guarantees
    byzantine.py             # Enhanced fault tolerance (0.51)
  validation/
    experimental.py          # 23 independent tests
    figures.py               # Automated figure generation`}
              </pre>
            </div>

            <div className="mt-12 flex flex-col items-center">
              <Link
                href="/publications"
                className="rounded-lg border-2 border-solid bg-dark p-2.5 px-8
                text-lg font-semibold text-light hover:border-dark hover:bg-transparent
                hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
                dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
              >
                Read the Full Papers
              </Link>
            </div>
          </Section>
        </Layout>
      </main>
    </>
  );
}
