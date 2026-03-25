import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const ContentBoundsChart = dynamic(() => import("@/components/charts/ContentBoundsChart"), { ssr: false });
const ContentIsomorphismChart = dynamic(() => import("@/components/charts/ContentIsomorphismChart"), { ssr: false });
const ContentIdealGasScatter = dynamic(() => import("@/components/charts/ContentIdealGasScatter"), { ssr: false });
const ContentPhaseChart = dynamic(() => import("@/components/charts/ContentPhaseChart"), { ssr: false });
const ContentMBChart = dynamic(() => import("@/components/charts/ContentMBChart"), { ssr: false });
const ContentVdWChart = dynamic(() => import("@/components/charts/ContentVdWChart"), { ssr: false });
const ContentVarianceChart = dynamic(() => import("@/components/charts/ContentVarianceChart"), { ssr: false });
const ContentCentralMoleculeChart = dynamic(() => import("@/components/charts/ContentCentralMoleculeChart"), { ssr: false });

const Section = ({ title, children, delay = 0 }) => {
  return (
    <motion.section
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      className="mt-24 first:mt-0 md:mt-16"
    >
      <h2 className="mb-8 text-3xl font-bold text-dark dark:text-light md:text-2xl sm:text-xl">
        {title}
      </h2>
      {children}
    </motion.section>
  );
};

const MathBlock = ({ children }) => (
  <div className="my-6 rounded-xl border border-solid border-dark/30 bg-dark/5 p-6 dark:border-light/30 dark:bg-light/5 overflow-x-auto">
    <pre className="font-mono text-sm text-dark dark:text-light leading-relaxed whitespace-pre-wrap">
      {children}
    </pre>
  </div>
);

const Prose = ({ children }) => (
  <p className="mt-4 text-base font-medium text-dark/85 dark:text-light/85 leading-relaxed">
    {children}
  </p>
);

const ChartContainer = ({ children, caption }) => (
  <div className="my-8">
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
          content="Complete theoretical foundations of the Pylon framework: the Boundedness Axiom, network-gas isomorphism, ideal gas law derivation, phase transitions, Maxwell-Boltzmann distributions, van der Waals corrections, variance restoration, and thermodynamic potentials."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Theoretical Foundations"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="mx-auto max-w-3xl text-center text-lg font-medium text-dark/80 dark:text-light/80 mb-8"
          >
            This page presents the complete theoretical framework of Pylon in sufficient depth
            that a reader can understand every result without consulting the original papers.
            Each section includes derivations, explanations, and interactive charts built from
            actual experimental validation data.
          </motion.p>

          {/* ================================================================ */}
          {/* Section 1: The Boundedness Axiom */}
          {/* ================================================================ */}
          <Section title="1. The Boundedness Axiom">
            <Prose>
              Every physical communication network is bounded. The address space V is finite &mdash;
              IPv4 provides 2&#179;&#178; addresses, IPv6 provides 2&#185;&#178;&#8312;, and every
              real deployment uses some strict subset. Observation time T is finite &mdash; no
              measurement can run for infinite duration. Node count N is finite &mdash; no network
              contains infinitely many devices. These are not approximations, simplifying assumptions,
              or engineering trade-offs. They are hard physical constraints imposed by the finiteness
              of real-world resources.
            </Prose>
            <Prose>
              The Boundedness Axiom is the single foundational assumption of the entire Pylon framework.
              From it, the full machinery of equilibrium statistical mechanics becomes applicable to
              network systems. The key mathematical consequence is Poincar&eacute; recurrence: in any
              bounded, measure-preserving dynamical system, almost every state returns arbitrarily close
              to its initial condition. For networks, this means that traffic patterns, congestion states,
              and routing configurations all exhibit recurrent, oscillatory behavior over sufficiently long
              time horizons.
            </Prose>
            <Prose>
              This recurrence is not a model or heuristic &mdash; it is a theorem. It guarantees that
              ensemble averages converge, that equilibrium distributions exist, and that the entire
              formalism of thermodynamics (partition functions, free energies, equations of state) can
              be rigorously applied. The bounded phase space for a network with N nodes, address space V,
              and observation window T is the set of all (position, momentum) pairs that satisfy the
              boundary constraints simultaneously.
            </Prose>

            <MathBlock>{`Axiom (Boundedness): For every physical network N, there exist finite bounds such that:
  - All addresses a in [0, V]     (finite address space)
  - All times t in [0, T]         (finite observation window)
  - All nodes n in {1, 2, ..., N} (finite node count)

Consequence (Poincare Recurrence): For any epsilon > 0, there exists a recurrence time
  t_rec such that ||x(t_rec) - x(0)|| < epsilon for almost all initial states x(0).

Phase space dimension: d = 6N (3 position + 3 momentum coordinates per node)
Phase space volume: Omega = V^N * (2*pi*m*kT)^(3N/2) / (N! * h^(3N))`}</MathBlock>

            <ChartContainer caption="Figure 1: Bounded phase space visualization showing 80 validated network configurations. Point size encodes temperature. Hover for PV/NkT ratio at each configuration.">
              <ContentBoundsChart />
            </ChartContainer>

            <Prose>
              The chart above visualizes the bounded phase space occupied by the 80 network configurations
              used in experimental validation. Each point represents a distinct (N, V, T) configuration.
              The dashed boundary shows the finite extent of the allowed parameter space. Concentric
              ellipses represent constant-energy shells in the phase space &mdash; the statistical mechanical
              analog of energy conservation surfaces. Every validated configuration falls strictly within
              the bounded region, confirming the axiom empirically.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 2: The Network-Gas Isomorphism */}
          {/* ================================================================ */}
          <Section title="2. The Network-Gas Isomorphism">
            <Prose>
              This is not a metaphor. It is not an analogy. It is not a &quot;model inspired by physics.&quot;
              It is a rigorous mathematical identity. We prove that bounded communication networks satisfy
              exactly the same axioms as confined ideal gas systems. The proof proceeds by establishing a
              bijection between the state spaces and showing that the dynamical evolution equations are
              identical under the mapping. Every theorem that holds for ideal gases holds, with
              quantitative precision, for communication networks.
            </Prose>
            <Prose>
              The isomorphism maps network quantities to gas quantities one-to-one. Nodes become molecules.
              The address space becomes the volume. Message rate becomes temperature. Routing contention
              becomes pressure. The key insight is that packet timings in a bounded network satisfy the
              same statistics as molecular velocities in a confined gas: the Maxwell-Boltzmann distribution
              emerges naturally from the maximum entropy principle applied to the bounded system.
            </Prose>
            <Prose>
              The isomorphism is validated to 0.1% accuracy across 80 independent network configurations.
              The ratio PV/(NkT) = 0.999 &plusmn; 0.005, meaning the network ideal gas law holds with the
              same precision routinely achieved in undergraduate physics laboratories. The second virial
              coefficient B&#8322; matches the theoretical prediction to within machine precision (0.0% error),
              confirming that even the non-ideal corrections transfer exactly.
            </Prose>

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
                  <MappingRow network="Nodes (routers, hosts)" gas="Molecules (particles)" />
                  <MappingRow network="Address space V (IPs)" gas="Volume V (container)" />
                  <MappingRow network="Message rate (packets/s)" gas="Temperature T (kinetic energy)" />
                  <MappingRow network="Routing contention (queue depth)" gas="Pressure P (wall collisions)" />
                  <MappingRow network="Latency distribution" gas="Maxwell-Boltzmann velocity distribution" />
                  <MappingRow network="Network congestion (saturation)" gas="Phase transition (condensation)" />
                  <MappingRow network="Bandwidth allocation" gas="Energy partition (equipartition)" />
                  <MappingRow network="Routing paths (hop count)" gas="Mean free path (collision distance)" />
                  <MappingRow network="Packet retransmission" gas="Elastic collision" />
                  <MappingRow network="QoS priority levels" gas="Energy levels (quantum states)" />
                  <MappingRow network="Flow control (TCP window)" gas="Equation of state constraint" />
                  <MappingRow network="Network entropy (Shannon)" gas="Thermodynamic entropy S" />
                </tbody>
              </table>
            </div>

            <ChartContainer caption="Figure 2: Network-gas isomorphism visualization. Each mapping preserves quantitative accuracy (PV/NkT = 0.999).">
              <ContentIsomorphismChart />
            </ChartContainer>

            <Prose>
              The isomorphism is structure-preserving in the mathematical sense: it respects the algebraic
              relations between quantities. For example, the network equipartition theorem states that
              each degree of freedom carries exactly (1/2)kT of average &quot;kinetic energy&quot; (measured
              as message rate variance per node). This matches the classical equipartition theorem for
              ideal gases to within measurement precision.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 3: The Ideal Gas Law: PV = NkT */}
          {/* ================================================================ */}
          <Section title="3. The Ideal Gas Law: PV = NkT">
            <Prose>
              The ideal gas law for networks is not assumed &mdash; it is derived from the Boundedness Axiom
              through a sequence of rigorous steps. The derivation follows the classical kinetic theory
              approach, adapted to the network setting. We begin with the microscopic dynamics of
              individual packets and derive the macroscopic equation of state that relates aggregate
              network quantities.
            </Prose>
            <Prose>
              Step 1: Start from the Boundedness Axiom. N nodes occupy address space V, exchanging messages
              at rate proportional to temperature T. Each node&apos;s message rate constitutes a degree of
              freedom with associated kinetic energy. Step 2: Apply the equipartition theorem. In
              equilibrium, each degree of freedom carries average energy (1/2)kT, where k is the
              network Boltzmann constant (determined by the gear ratio structure). Step 3: Compute
              pressure. Routing contention P arises from &quot;collisions&quot; of packets at shared
              resources (switches, links). The rate of these collisions is proportional to N * v_rms / V,
              where v_rms is the root-mean-square message rate.
            </Prose>
            <Prose>
              Step 4: Combine. Each collision transfers momentum 2mv to the &quot;walls&quot; (network
              boundaries). The collision rate per unit area is N * v_rms / (6V). The resulting pressure is
              P = N * m * v_rms&#178; / (3V). Using equipartition ((1/2)m * v_rms&#178; = (3/2)kT), we
              obtain P = NkT / V. Rearranging: PV = NkT. This is not a fit or an approximation &mdash; it
              is an exact consequence of the axioms.
            </Prose>

            <MathBlock>{`Derivation of PV = NkT from network axioms:

1. Equipartition:    <(1/2) m v_i^2> = (1/2) kT    for each degree of freedom i
2. RMS velocity:     <v^2> = v_x^2 + v_y^2 + v_z^2 = 3kT/m
3. Wall collision rate:  Gamma = N * <|v_x|> / (2V^(1/3))  per unit area
4. Momentum transfer:   Delta_p = 2m * v_x  per collision
5. Pressure:         P = Gamma * Delta_p = N * m * <v_x^2> / V = NkT / (3V) * 3 = NkT/V
6. Equation of state:   PV = NkT

Validation:  PV/(NkT) = 0.999 +/- 0.005  across 80 configurations
             R^2 = 0.999 for the PV vs NkT regression
             Mean absolute error: 0.1%`}</MathBlock>

            <ChartContainer caption="Figure 3: PV vs NkT for 80 network configurations. Perfect correlation along y=x line confirms the ideal gas law.">
              <ContentIdealGasScatter />
            </ChartContainer>

            <Prose>
              The scatter plot shows PV plotted against NkT for all 80 validated configurations. The
              data points fall precisely on the y = x line (shown dashed), confirming PV = NkT. The
              configurations span node counts from 4 to 20, address spaces from 100 to 500, and a
              range of message rates. The law holds uniformly across this entire parameter space,
              not just for a narrow slice of configurations.
            </Prose>
            <Prose>
              The R&#178; value of 0.999 means that 99.9% of the variance in PV is explained by NkT.
              The remaining 0.1% is attributable to finite-size effects and non-ideal corrections
              (quantified by the second virial coefficient B&#8322;, discussed in Section 6). This
              level of accuracy is comparable to standard laboratory measurements of the ideal gas
              law in physical systems.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 4: Phase Transitions */}
          {/* ================================================================ */}
          <Section title="4. Phase Transitions">
            <Prose>
              Networks undergo genuine phase transitions &mdash; qualitative changes in collective behavior
              at specific critical temperatures. These are not gradual performance degradations but
              sharp, discontinuous changes in the order parameter. The Pylon framework identifies three
              distinct phases, each with a clear physical interpretation in network terms.
            </Prose>
            <Prose>
              The <strong>gas phase</strong> (T &gt; T_c = 3.42) corresponds to low-utilization networks
              where packets travel independently with minimal interaction. Routing is chaotic in the
              statistical mechanical sense: packet trajectories are effectively uncorrelated, and the
              system explores its full phase space ergodically. Latency distributions are broad and
              follow the Maxwell-Boltzmann form. This is the normal operating regime for most networks.
            </Prose>
            <Prose>
              The <strong>liquid phase</strong> (T_m &lt; T &lt; T_c, where T_m = 2.65) represents
              moderate congestion where packets begin to interact strongly. Partial coordination
              emerges: routing paths become correlated as shared bottlenecks create effective
              &quot;intermolecular forces.&quot; The latency distribution narrows and shifts. Flow
              control mechanisms (TCP windows, buffer management) begin to constrain the dynamics.
              This is the phase where traditional congestion control mechanisms operate.
            </Prose>
            <Prose>
              The <strong>crystal phase</strong> (T &lt; T_m = 2.65) corresponds to heavy congestion
              with phase-locked routing. Packets are trapped in persistent queues, and the system
              loses ergodicity &mdash; it can no longer explore its full state space. Routing paths
              become deterministic and frozen. The order parameter &Psi; approaches 1.0, indicating
              maximal spatial correlation. In practice, this corresponds to network gridlock or
              congestion collapse.
            </Prose>

            <MathBlock>{`Phase structure:

Gas phase      (T > T_c = 3.42):  Psi ~ 0, ergodic, Maxwell-Boltzmann statistics
Liquid phase   (T_m < T < T_c):   0 < Psi < 1, partial coordination, correlated flows
Crystal phase  (T < T_m = 2.65):  Psi ~ 1, phase-locked, deterministic routing

Order parameter:  Psi = |<exp(i*theta_j)>|  (phase coherence of packet timings)
                  Psi = 0: fully disordered (gas)
                  Psi = 1: fully ordered (crystal)

Critical exponents:
  - Correlation length:  xi ~ |T - T_c|^(-nu)
  - Order parameter:     Psi ~ (T_c - T)^(beta)   for T < T_c
  - Susceptibility:      chi ~ |T - T_c|^(-gamma)`}</MathBlock>

            <ChartContainer caption="Figure 4: Order parameter vs temperature showing gas-liquid (T_c = 3.42) and liquid-crystal (T_m = 2.65) phase transitions.">
              <ContentPhaseChart />
            </ChartContainer>

            <Prose>
              The phase diagram reveals the characteristic sigmoidal shape of a continuous phase
              transition at T_c = 3.42. Below this temperature, the order parameter rises sharply,
              indicating the onset of spatial correlation in packet timings. The second transition at
              T_m = 2.65 marks the liquid-crystal boundary, where the system transitions to a fully
              ordered state. The shaded regions in the chart correspond to the three phases, with
              critical temperatures marked by dashed lines.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 5: Maxwell-Boltzmann Distribution */}
          {/* ================================================================ */}
          <Section title="5. Maxwell-Boltzmann Distribution">
            <Prose>
              In the gas phase, packet timing intervals follow the Maxwell-Boltzmann distribution.
              This is not an empirical observation requiring a fitting procedure &mdash; it is a
              mathematical consequence of the Boundedness Axiom combined with the principle of
              maximum entropy. The Maxwell-Boltzmann distribution is the unique probability
              distribution that maximizes entropy subject to fixed mean energy, which is exactly
              the constraint imposed by a bounded system in thermal equilibrium.
            </Prose>
            <Prose>
              The distribution has the form f(v) = 4&pi;(m/2&pi;kT)&#179;&#8260;&#178; v&#178;
              exp(&minus;mv&#178;/2kT), where v represents the inter-packet timing interval (the
              network analog of molecular speed). Three characteristic speeds emerge naturally:
              the most probable speed v_mp = &radic;(2kT/m), the mean speed v_mean = &radic;(8kT/(&pi;m)),
              and the root-mean-square speed v_rms = &radic;(3kT/m). These correspond to the mode,
              mean, and RMS of the latency distribution respectively.
            </Prose>
            <Prose>
              Validation uses the Kolmogorov-Smirnov (KS) test, which quantifies the maximum deviation
              between the empirical and theoretical cumulative distribution functions. At T = 1.0, the
              KS test yields a p-value of 0.465, meaning there is a 46.5% probability of seeing a
              deviation this large or larger even if the data truly follows the Maxwell-Boltzmann
              distribution. All tested temperatures pass the KS test, confirming the distributional
              identity across the full operating range.
            </Prose>

            <MathBlock>{`Maxwell-Boltzmann speed distribution:

f(v) = 4*pi * (m / (2*pi*kT))^(3/2) * v^2 * exp(-m*v^2 / (2*kT))

Characteristic speeds (for T = 1.0, m = 1.0):
  v_mp   = sqrt(2*kT/m)        = 1.414  (most probable)
  v_mean = sqrt(8*kT/(pi*m))   = 1.596  (mean)
  v_rms  = sqrt(3*kT/m)        = 1.732  (root-mean-square)

Validation:
  KS test p-value at T=1.0:  0.465  (fail to reject H0: data ~ MB)
  All temperatures pass KS test with p > 0.05
  Energy distribution: f(E) = 2*pi * (1/(pi*kT))^(3/2) * sqrt(E) * exp(-E/kT)`}</MathBlock>

            <ChartContainer caption="Figure 5: Maxwell-Boltzmann distribution at T=1.0. Histogram shows measured latency intervals; solid curve is the theoretical MB distribution. Vertical lines mark characteristic speeds.">
              <ContentMBChart />
            </ChartContainer>

            <Prose>
              The histogram shows the measured distribution of inter-packet timing intervals for a
              network operating at T = 1.0. The solid curve is the theoretical Maxwell-Boltzmann
              prediction with no free parameters (temperature is measured independently from the
              mean kinetic energy). The agreement is excellent, with the KS p-value of 0.465
              indicating no statistically significant deviation from the theoretical form.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 6: Van der Waals Corrections */}
          {/* ================================================================ */}
          <Section title="6. Van der Waals Corrections">
            <Prose>
              The ideal gas law PV = NkT assumes non-interacting particles (packets). In real networks,
              packets do interact: they share links, contend for buffers, and experience queueing delays
              that create effective inter-packet forces. These interactions produce systematic deviations
              from ideal behavior, especially at high network density (utilization). The Van der Waals
              equation captures these corrections.
            </Prose>
            <Prose>
              The virial expansion Z = PV/(NkT) = 1 + B&#8322;&rho; + B&#8323;&rho;&#178; + ... provides
              a systematic series of corrections to the ideal gas law. The second virial coefficient
              B&#8322; encodes the leading two-body interaction: B&#8322; &lt; 0 indicates net attractive
              interactions (packets that travel together experience less contention than independent packets),
              while B&#8322; &gt; 0 indicates net repulsive interactions (shared congestion). For the validated
              network configurations, B&#8322; = &minus;1.310, matching the theoretical prediction exactly.
            </Prose>
            <Prose>
              The Boyle temperature T_B = 3.41 is the temperature at which B&#8322;(T_B) = 0 and the gas
              behaves ideally regardless of density. Below the Boyle temperature, attractive interactions
              dominate (B&#8322; &lt; 0), and the network compresses more easily than the ideal prediction.
              Above it, repulsive interactions dominate (B&#8322; &gt; 0), and the network is harder to
              compress. The Boyle temperature lies just below the critical temperature T_c = 3.42,
              which is physically consistent &mdash; near the critical point, the balance between
              attractive and repulsive interactions shifts.
            </Prose>

            <MathBlock>{`Van der Waals equation of state:

(P + a*N^2/V^2)(V - N*b) = NkT

Virial expansion:  Z = PV/(NkT) = 1 + B2*rho + B3*rho^2 + ...
  where rho = N/V is the number density

Second virial coefficient:
  B2_fitted       = -1.310
  B2_theoretical  = -1.310
  Error           = 0.0%

Boyle temperature:  T_B = 3.41  (where B2(T_B) = 0)

Physical interpretation:
  B2 < 0: Attractive interactions dominate (packets cluster)
  B2 > 0: Repulsive interactions dominate (congestion excludes)
  B2 = 0: Ideal behavior recovered (at T = T_B)`}</MathBlock>

            <ChartContainer caption="Figure 6: Compressibility Z vs density. Ideal gas (dashed), virial expansion (solid), and measured data points. Deviation from Z=1 quantifies non-ideal behavior.">
              <ContentVdWChart />
            </ChartContainer>
          </Section>

          {/* ================================================================ */}
          {/* Section 7: Variance Restoration */}
          {/* ================================================================ */}
          <Section title="7. Variance Restoration">
            <Prose>
              Variance restoration is the most practically important result of the framework. It
              provides a mechanism for autonomous network self-regulation that requires no external
              controller, no feedback loop, and no centralized decision-making. When the network is
              perturbed from equilibrium (by a traffic spike, a link failure, or a routing change),
              the variance of packet timings decays exponentially back to its equilibrium value with
              a characteristic time constant &tau;.
            </Prose>
            <Prose>
              The mechanism is thermodynamic: the Second Law guarantees that entropy increases toward
              its maximum (the equilibrium value), and this entropy increase drives the variance
              restoration. The decay follows &Delta;T(t) = T&#8320; exp(&minus;t/&tau;), where T&#8320;
              is the initial perturbation magnitude and &tau; is the restoration time constant. The
              fitted time constants are &tau; = 0.499, 0.499, 0.500, 0.500 ms for initial perturbations
              T&#8320; = 1, 5, 10, 20 respectively. The theoretical prediction is &tau; = 0.500 ms,
              giving an error of only 0.2%.
            </Prose>
            <Prose>
              Crucially, the restoration time &tau; is independent of the perturbation magnitude T&#8320;.
              A small perturbation and a large perturbation both restore on the same timescale. This is
              a hallmark of linear response theory: the system&apos;s response to small perturbations is
              proportional to the perturbation itself, with a universal proportionality constant. This
              universality is what makes variance restoration a practical tool &mdash; the network
              administrator does not need to know the perturbation magnitude in advance.
            </Prose>

            <MathBlock>{`Variance restoration dynamics:

DeltaT(t) = T0 * exp(-t / tau)

Fitted time constants:
  T0 =  1:  tau = 0.499 ms
  T0 =  5:  tau = 0.499 ms
  T0 = 10:  tau = 0.500 ms
  T0 = 20:  tau = 0.500 ms

Theoretical prediction:  tau = 0.500 ms
Maximum error:           0.2%

Key property:  tau is INDEPENDENT of T0  (linear response universality)

Physical mechanism:
  dS/dt >= 0  (Second Law)  =>  system driven toward equilibrium
  d(DeltaT)/dt = -DeltaT/tau  (Newton's law of cooling for networks)
  Solution: exponential decay with universal time constant`}</MathBlock>

            <ChartContainer caption="Figure 7: Variance restoration for four perturbation magnitudes (T0 = 1, 5, 10, 20). All decay with tau ~ 0.5 ms regardless of initial amplitude.">
              <ContentVarianceChart />
            </ChartContainer>

            <Prose>
              The four decay curves in the chart demonstrate the universality of the restoration time
              constant. Despite initial perturbations spanning a factor of 20 (from T&#8320; = 1 to
              T&#8320; = 20), all four curves exhibit the same exponential decay rate. The sampled
              points along each curve confirm the fitted values, and the vertical dashed line at
              t = &tau; &asymp; 0.5 ms marks one time constant, where each curve has decayed to
              1/e &asymp; 37% of its initial value.
            </Prose>
          </Section>

          {/* ================================================================ */}
          {/* Section 8: Thermodynamic Potentials */}
          {/* ================================================================ */}
          <Section title="8. Thermodynamic Potentials">
            <Prose>
              The complete thermodynamic description of a network system requires four fundamental
              potentials, each providing a different &quot;natural&quot; perspective on the system&apos;s
              state. These potentials are not independent &mdash; they are related by Legendre
              transforms, ensuring thermodynamic consistency. All four are derived from the partition
              function Z = &Sigma; exp(&minus;E_i / kT), which encodes the full statistical mechanics
              of the bounded network.
            </Prose>
            <Prose>
              The internal energy U = (3/2)NkT gives the total kinetic energy of all packets. The
              Helmholtz free energy F = U &minus; TS = &minus;NkT ln(V/N) &minus; NkT is the maximum
              work extractable at constant temperature and volume. The Gibbs free energy
              G = F + PV = NkT[ln(P/kT) + const] governs equilibria at constant temperature and
              pressure. The chemical potential &mu; = (&part;G/&part;N) at constant T and P determines whether
              adding another node to the network is thermodynamically favorable.
            </Prose>

            <MathBlock>{`Thermodynamic potentials for the network gas:

Internal energy:      U = (3/2) NkT
Entropy:              S = Nk [ln(V/(N*lambda^3)) + 5/2]   (Sackur-Tetrode)
Helmholtz free energy: F = U - TS = -NkT ln(V/N) - NkT + (3/2)NkT ln(2*pi*mkT/h^2)
Gibbs free energy:     G = F + PV = NkT [ln(N*kT/(V*P_0)) + const]
Enthalpy:             H = U + PV = (5/2) NkT
Chemical potential:   mu = (dG/dN)_{T,P} = kT ln(N/(V-Nb)) + corrections

Maxwell relations (consistency checks):
  (dT/dV)_S = -(dP/dS)_V
  (dS/dV)_T =  (dP/dT)_V
  (dS/dP)_T = -(dV/dT)_P
  (dT/dP)_S =  (dV/dS)_P

Validated: mu error = 10^(-16) (Sackur-Tetrode entropy)`}</MathBlock>
          </Section>

          {/* ================================================================ */}
          {/* Section 9: Transport Coefficients */}
          {/* ================================================================ */}
          <Section title="9. Transport Coefficients">
            <Prose>
              Transport coefficients quantify how quickly the network equilibrates with respect to
              different conserved quantities. Three transport coefficients are primary: viscosity &eta;
              (momentum transport), diffusion coefficient D (particle transport), and thermal conductivity
              &kappa; (energy transport). Each has a network interpretation and is computed from the
              kinetic theory of the network gas.
            </Prose>
            <Prose>
              Network viscosity &eta; = (1/3) n m &lambda; v_mean measures the resistance of the network
              to shear flow &mdash; the tendency for adjacent regions of the network to maintain different
              message rates. The diffusion coefficient D = (1/3) &lambda; v_mean governs how quickly
              traffic perturbations spread spatially. Thermal conductivity &kappa; = (1/3) n c_v &lambda;
              v_mean describes how quickly energy (message rate) imbalances equilibrate across the network.
            </Prose>
            <Prose>
              These three coefficients satisfy the Wiedemann-Franz law: &kappa; / (&sigma; T) = L, where
              &sigma; is the electrical (network) conductivity and L is the Lorenz number. This law,
              discovered empirically for metals in 1853, emerges as a theorem in the network context.
              It states that a network&apos;s ability to transport energy is proportional to its ability
              to transport messages, with a universal proportionality constant determined by the network
              Boltzmann constant.
            </Prose>

            <MathBlock>{`Transport coefficients:

Viscosity:             eta = (1/3) * n * m * lambda * v_mean
Diffusion:             D   = (1/3) * lambda * v_mean
Thermal conductivity:  kappa = (1/3) * n * c_v * lambda * v_mean

where:
  n = N/V (number density)
  lambda = 1/(n * sigma_cross) (mean free path)
  v_mean = sqrt(8*kT / (pi*m))

Wiedemann-Franz law:   kappa / (sigma * T) = L = (pi^2 / 3) * (k/e)^2
  => Energy transport efficiency is proportional to message transport efficiency

Green-Kubo relations:  D = (1/3) * integral_0^inf <v(0) * v(t)> dt
  => Transport coefficients from velocity autocorrelation functions`}</MathBlock>
          </Section>

          {/* ================================================================ */}
          {/* Section 10: Central Molecule Impossibility */}
          {/* ================================================================ */}
          <Section title="10. Central Molecule Impossibility">
            <Prose>
              The Central Molecule approach to network monitoring attempts to track every individual
              packet &mdash; recording its source, destination, timestamps, and path through the network.
              This is the analog of tracking every molecule in a gas, which is both impractical and
              unnecessary. The Pylon framework proves that this approach is not merely inefficient but
              fundamentally impossible at scale, and provides a quantitative bound on the overhead.
            </Prose>
            <Prose>
              The measured overhead ratio is 69,427&times;. That is, per-packet tracking requires nearly
              70,000 times more resources than the statistical mechanical (variance-based) approach.
              This ratio is not a tunable parameter &mdash; it is a consequence of the information-theoretic
              structure of the problem. Tracking N particles individually requires O(N) state, while the
              thermodynamic approach requires O(1) state (temperature, pressure, and a few other macroscopic
              variables). As N grows, the ratio grows proportionally.
            </Prose>
            <Prose>
              This result has profound practical implications. It means that any monitoring system based on
              deep packet inspection, per-flow state, or complete traffic capture is operating in the
              wrong complexity class. The thermodynamic approach achieves the same monitoring fidelity
              (0.1% accuracy) with five orders of magnitude less overhead. This is not an engineering
              improvement but a fundamental change in the computational complexity of the problem.
            </Prose>

            <ChartContainer caption="Figure 8: Central Molecule overhead comparison. Per-packet tracking requires 69,427x more resources than variance-based control.">
              <ContentCentralMoleculeChart />
            </ChartContainer>
          </Section>

          {/* ================================================================ */}
          {/* Section 11: Heat Capacity and Uncertainty */}
          {/* ================================================================ */}
          <Section title="11. Heat Capacity and Uncertainty">
            <Prose>
              The heat capacity of the network gas provides a direct test of the equipartition theorem.
              For an ideal gas with f degrees of freedom, C_V = (f/2)Nk. For a monatomic ideal gas
              (f = 3), C_V = (3/2)Nk. The ratio &gamma; = C_P / C_V = (f+2)/f = 5/3 for f = 3 is
              a dimensionless quantity that characterizes the compressibility of the network gas.
              The measured value matches the theoretical prediction exactly.
            </Prose>
            <Prose>
              The network uncertainty relation &sigma;_x &middot; &sigma;_p &ge; &#8463;_net establishes
              a fundamental lower bound on the product of position and momentum uncertainties in the
              network phase space. Here &#8463;_net is the network Planck constant, determined by the
              minimum resolvable address-time cell. This is not an analogy to quantum mechanics &mdash;
              it is a direct consequence of the discrete, bounded nature of the network state space.
              The validation confirms zero violations across all tested configurations.
            </Prose>

            <MathBlock>{`Heat capacity:

C_V = (3/2) Nk        (constant volume, monatomic network gas)
C_P = (5/2) Nk        (constant pressure)
gamma = C_P / C_V = 5/3 = 1.667

Validation: gamma measured = 5/3 (exact match)

Network uncertainty relation:

sigma_x * sigma_p >= hbar_net / 2

where:
  sigma_x = position uncertainty (address resolution)
  sigma_p = momentum uncertainty (timing resolution)
  hbar_net = minimum address-time cell area

Violations detected: 0 out of 80 configurations
Minimum product: sigma_x * sigma_p = 1.002 * hbar_net/2  (just above bound)`}</MathBlock>
          </Section>

          {/* ================================================================ */}
          {/* Section 12: Phonon Dispersion */}
          {/* ================================================================ */}
          <Section title="12. Phonon Dispersion">
            <Prose>
              In the crystal phase (T &lt; T_m = 2.65), the network supports collective excitations
              analogous to phonons in solid-state physics. These &quot;network phonons&quot; are
              propagating waves of packet timing perturbation that travel through the lattice of
              phase-locked routing paths. The dispersion relation
              &omega;&#178;(q) = &omega;&#8320;&#178;[1 &minus; cos(qa)] describes how the frequency
              of these waves depends on their wavevector q.
            </Prose>
            <Prose>
              At long wavelengths (small q), the dispersion is linear: &omega; &asymp; (&omega;&#8320; a / &radic;2) q.
              This linear regime corresponds to sound waves &mdash; coherent disturbances that propagate at
              a well-defined speed (the network &quot;speed of sound&quot;). At short wavelengths (large q,
              near the Brillouin zone boundary), the dispersion flattens, and group velocity goes to zero.
              This means that short-wavelength perturbations cannot propagate and remain localized.
            </Prose>
            <Prose>
              The phonon dispersion relation has practical significance for network resilience. In the
              crystal phase, perturbations propagate as waves rather than diffusing randomly. This means
              that a localized failure can produce a coherent disturbance that travels across the network
              at the speed of sound, potentially triggering cascading failures. Understanding the
              dispersion relation allows prediction and mitigation of these propagating failure modes.
            </Prose>

            <MathBlock>{`Phonon dispersion relation:

omega^2(q) = omega_0^2 * [1 - cos(q*a)]

where:
  omega_0 = natural frequency (set by network coupling strength)
  a = lattice spacing (inter-node distance in address space)
  q = wavevector (spatial frequency of the perturbation)

Limiting cases:
  Small q (long wavelength):  omega ~ (omega_0 * a / sqrt(2)) * q  (linear, acoustic)
  q = pi/a (zone boundary):   omega = sqrt(2) * omega_0  (maximum frequency)

Group velocity:  v_g = d(omega)/dq = (omega_0^2 * a * sin(qa)) / (2*omega)
Speed of sound:  c_s = lim_{q->0} v_g = omega_0 * a / sqrt(2)

Band structure:
  Acoustic band:  0 <= omega <= sqrt(2) * omega_0
  Band gap:       0.586 (between acoustic and optical branches)
  Optical band:   above the gap
  Topological invariant: winding number = 1, Berry phase = pi`}</MathBlock>
          </Section>

          {/* ================================================================ */}
          {/* Navigation */}
          {/* ================================================================ */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-24 flex flex-col items-center gap-4 sm:gap-3"
          >
            <div className="flex gap-4 flex-wrap justify-center">
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
            </div>
            <Link
              href="/publications"
              className="text-base font-medium text-dark underline underline-offset-4
              dark:text-light"
            >
              Read the Full Papers
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
