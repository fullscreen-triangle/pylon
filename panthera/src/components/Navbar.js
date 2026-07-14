import Link from "next/link";
import React, { useState } from "react";
import Logo from "./Logo";
import { useRouter } from "next/router";
import { GithubIcon } from "./Icons";
import { motion, AnimatePresence } from "framer-motion";

const NavLink = ({ href, title, onClick }) => {
  const router = useRouter();
  const active = router.asPath === href;

  return (
    <Link
      href={href}
      onClick={onClick}
      className={`relative text-sm tracking-widest uppercase text-zinc-400 hover:text-white transition-colors duration-200 ${
        active ? "text-white" : ""
      }`}
    >
      {title}
      {active && (
        <span className="absolute -bottom-0.5 left-0 w-full h-px bg-white/40" />
      )}
    </Link>
  );
};

const Navbar = () => {
  const [open, setOpen] = useState(false);

  const close = () => setOpen(false);

  return (
    <>
      {/* invisible hover strip at top of screen */}
      <div
        className="fixed top-0 left-0 w-full h-6 z-50"
        onMouseEnter={() => setOpen(true)}
      />

      <AnimatePresence>
        {open && (
          <motion.header
            initial={{ y: "-100%" }}
            animate={{ y: 0 }}
            exit={{ y: "-100%" }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            onMouseLeave={() => setOpen(false)}
            className="fixed top-0 left-0 w-full z-50 flex items-center justify-between px-12 py-5
              bg-black/80 backdrop-blur-md border-b border-white/5"
          >
            {/* logo */}
            <Logo />

            {/* nav links */}
            <nav className="flex items-center gap-8">
              <NavLink href="/" title="Home" onClick={close} />
              <NavLink href="/framework" title="Framework" onClick={close} />
              <NavLink href="/state" title="State" onClick={close} />
              <NavLink href="/trajectory" title="Trajectory" onClick={close} />
              <NavLink href="/demo" title="Demo" onClick={close} />
              <NavLink href="/mesh" title="Mesh" onClick={close} />
              <NavLink href="/publications" title="Papers" onClick={close} />
            </nav>

            {/* github */}
            <motion.a
              href="https://github.com/fullscreen-triangle/pylon"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.9 }}
              aria-label="GitHub"
              className="w-5 h-5 text-zinc-400 hover:text-white transition-colors"
            >
              <GithubIcon />
            </motion.a>
          </motion.header>
        )}
      </AnimatePresence>
    </>
  );
};

export default Navbar;
