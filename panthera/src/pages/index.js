import Head from "next/head";
import dynamic from "next/dynamic";

const LaptopModel = dynamic(() => import("@/components/LaptopModel"), {
  ssr: false,
  loading: () => null,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Pylon</title>
        <meta name="description" content="Thermodynamic network coordination framework." />
      </Head>
      <main className="fixed inset-0 flex items-center justify-center bg-dark">
        <div className="w-full max-w-3xl">
          <LaptopModel />
        </div>
      </main>
    </>
  );
}
