import { useEffect } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";

export default function About() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/framework");
  }, [router]);

  return (
    <>
      <Head>
        <title>Redirecting to Framework | Pylon</title>
        <meta name="description" content="Redirecting to the Pylon framework overview page." />
      </Head>
      <TransitionEffect />
    </>
  );
}
