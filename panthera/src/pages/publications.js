import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

const PaperCard = ({ title, authors, year, description, link, linkLabel }) => {
  return (
    <motion.div
      variants={fadeInUp}
      className="relative rounded-2xl border border-solid border-dark bg-light p-8
      dark:border-light dark:bg-dark md:p-6 sm:p-5"
    >
      <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem]
      rounded-br-3xl bg-dark dark:bg-light md:-right-2 md:w-[101%]" />
      <div className="flex flex-col">
        <span className="text-sm font-medium text-primary dark:text-primaryDark">
          {year}
        </span>
        <h3 className="mt-2 text-xl font-bold text-dark dark:text-light md:text-lg">
          {title}
        </h3>
        <p className="mt-1 text-sm font-medium text-dark/60 dark:text-light/60">
          {authors}
        </p>
        <p className="mt-4 text-base font-medium leading-relaxed text-dark/80 dark:text-light/80 sm:text-sm">
          {description}
        </p>
        {link && (
          <Link
            href={link}
            className="mt-4 inline-block text-base font-semibold text-primary underline
            underline-offset-4 hover:text-primary/80 dark:text-primaryDark dark:hover:text-primaryDark/80"
          >
            {linkLabel || "Read more"}
          </Link>
        )}
      </div>
    </motion.div>
  );
};

export default function Publications() {
  return (
    <>
      <Head>
        <title>Publications and References | Pylon Framework</title>
        <meta
          name="description"
          content="Publications and references for the Pylon thermodynamic network coordination framework. Two current papers on equations of state and trajectory completion, plus foundation and source theory references."
        />
      </Head>

      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Publications & References"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Section 1: Current Work */}
          <section>
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-10 text-3xl font-bold text-dark dark:text-light md:text-2xl"
            >
              Current Work
            </motion.h2>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.15 } },
              }}
              className="grid grid-cols-1 gap-12"
            >
              <PaperCard
                title="Equations of State for Transcendent Observer Networks"
                authors="K.F. Sachikonye"
                year="2025"
                description="Establishes the mathematical identity between bounded communication networks and
                confined ideal gas systems. Derives PV = NkT for networks, validates to 0.1% accuracy across
                80 configurations. Identifies phase transitions, derives Maxwell-Boltzmann latency distributions,
                and computes all thermodynamic potentials. 10 independent tests, all fully validated."
                link="/state"
                linkLabel="View paper details and figures"
              />
              <PaperCard
                title="Backward Trajectory Completion on Gear Ratio Manifolds"
                authors="K.F. Sachikonye"
                year="2025"
                description="Develops backward trajectory completion using transcendent observers with gear ratios
                across eight scales. Achieves O(log M) message identification replacing O(M) forward search.
                Derives thermodynamic security immune to quantum computing. Proves gauge invariance and fiber
                bundle transitivity to machine precision. 13 independent tests, 10 fully validated."
                link="/trajectory"
                linkLabel="View paper details and figures"
              />
            </motion.div>
          </section>

          {/* Section 2: Foundation Papers */}
          <section className="mt-24 md:mt-16">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-10 text-3xl font-bold text-dark dark:text-light md:text-2xl"
            >
              Foundation Papers
            </motion.h2>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
              }}
              className="grid grid-cols-2 gap-10 lg:gap-8 md:grid-cols-1"
            >
              <PaperCard
                title="The Transcendent Observer: A Unified Framework for Network Coordination"
                authors="K.F. Sachikonye"
                year="2024"
                description="Introduces the transcendent observer concept: finite observers that monitor adjacent
                scales through gear ratios, enabling hierarchical network coordination without global consensus."
              />
              <PaperCard
                title="Bounded Network Dynamics and Poincar&eacute; Recurrence"
                authors="K.F. Sachikonye"
                year="2024"
                description="Proves that bounded communication networks exhibit Poincar&eacute; recurrence and derives
                the oscillatory dynamics that make statistical mechanical treatment possible."
              />
              <PaperCard
                title="S-Entropy Coordinates for Distributed Systems"
                authors="K.F. Sachikonye"
                year="2024"
                description="Develops the S-entropy coordinate system that enables deterministic backward trajectory
                completion. Maps network state space into thermodynamic coordinates where navigation is direct."
              />
              <PaperCard
                title="Thermodynamic Security: Beyond Computational Hardness"
                authors="K.F. Sachikonye"
                year="2024"
                description="Derives security guarantees from the Second Law of Thermodynamics rather than
                computational hardness assumptions. Proves immunity to P=NP resolution and quantum computing
                attacks."
              />
            </motion.div>
          </section>

          {/* Section 3: Source Theory */}
          <section className="mt-24 md:mt-16">
            <motion.h2
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="mb-10 text-3xl font-bold text-dark dark:text-light md:text-2xl"
            >
              Source Theory
            </motion.h2>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={{
                hidden: { opacity: 0 },
                visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
              }}
              className="grid grid-cols-2 gap-10 lg:gap-8 md:grid-cols-1"
            >
              <PaperCard
                title="Gas Computing: A Novel Framework for Distributed Computation"
                authors="K.F. Sachikonye"
                year="2023"
                description="Introduces the gas computing paradigm: treating distributed computation as molecular
                dynamics in a confined gas. Establishes the foundational mapping between computational processes
                and thermodynamic state variables."
              />
              <PaperCard
                title="Single Particle Dynamics in Communication Networks"
                authors="K.F. Sachikonye"
                year="2023"
                description="Analyzes the behavior of individual messages as particles in a network gas. Derives
                single-particle statistics, collision dynamics, and mean free path for network messages."
              />
              <PaperCard
                title="Trajectory Computing: State Evolution in Bounded Systems"
                authors="K.F. Sachikonye"
                year="2023"
                description="Develops the trajectory computing framework where computation is defined as trajectory
                evolution in a bounded state space. Connects computational processes to physical trajectories."
              />
              <PaperCard
                title="Backward Trajectory Completion: Theory and Algorithms"
                authors="K.F. Sachikonye"
                year="2023"
                description="Original theoretical development of backward trajectory completion. Proves existence
                and uniqueness of backward completions under boundedness, and derives the logarithmic scaling
                advantage over forward search."
              />
            </motion.div>
          </section>

          {/* Contact CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mt-20 flex flex-col items-center text-center"
          >
            <p className="text-lg font-medium text-dark/75 dark:text-light/75 md:text-base">
              Interested in this research? We welcome collaborators and funding partners.
            </p>
            <Link
              href="mailto:kundai.sachikonye@wzw.tum.de"
              className="mt-6 rounded-lg border-2 border-solid bg-dark p-2.5 px-8
              text-lg font-semibold text-light hover:border-dark hover:bg-transparent
              hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
              dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-6 md:text-base"
            >
              Contact Us
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
