import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

const LaptopModel = dynamic(() => import("@/components/LaptopModel"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[500px] w-full items-center justify-center">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-primary border-t-transparent dark:border-primaryDark dark:border-t-transparent" />
    </div>
  ),
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Pylon | Thermodynamic Network Coordination</title>
        <meta
          name="description"
          content="Pylon proves distributed communication networks are mathematically identical to ideal gases."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full items-center justify-center dark:text-light" style={{ minHeight: "calc(100vh - 5.5rem)" }}>
        <div className="w-full max-w-2xl px-8">
          <LaptopModel />
        </div>
      </main>
    </>
  );
}
