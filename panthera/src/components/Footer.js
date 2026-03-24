import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer
      className="w-full border-t-2 border-solid border-dark
    font-medium text-lg dark:text-light dark:border-light sm:text-base
    "
    >
      <Layout className="py-8 flex items-center justify-between lg:flex-col lg:py-6">
        <span>&copy; 2025 Pylon Framework. Technical University of Munich.</span>

        <div className="flex flex-col items-center lg:py-2">
          <div className="flex items-center">
            <Link
              href="mailto:kundai.sachikonye@wzw.tum.de"
              className="underline underline-offset-2"
            >
              Collaborate
            </Link>
            <span className="text-primary text-2xl px-1 dark:text-primaryDark">&#9825;</span>
            <Link
              href="https://github.com/fullscreen-triangle/pylon"
              target="_blank"
              className="underline underline-offset-2"
            >
              GitHub
            </Link>
          </div>
          <span className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Kundai Farai Sachikonye &middot; School of Life Sciences
          </span>
        </div>

        <Link
          href="mailto:kundai.sachikonye@wzw.tum.de"
          className="underline underline-offset-2"
        >
          Contact
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
