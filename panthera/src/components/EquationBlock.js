import React from "react";
import { motion } from "framer-motion";

const EquationBlock = ({ equation, label }) => {
  return (
    <motion.div
      className="w-full my-6 bg-gray-50 dark:bg-gray-800 rounded-2xl px-8 py-6 flex items-center justify-between"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      viewport={{ once: true }}
    >
      <div className="flex-1 text-center">
        <span className="font-mono text-lg md:text-base text-dark dark:text-light">
          {equation}
        </span>
      </div>
      {label && (
        <span className="text-sm text-gray-500 dark:text-gray-400 ml-4 whitespace-nowrap">
          ({label})
        </span>
      )}
    </motion.div>
  );
};

export default EquationBlock;
