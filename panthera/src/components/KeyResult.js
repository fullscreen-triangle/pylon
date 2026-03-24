import React from "react";
import { motion } from "framer-motion";

const KeyResult = ({ value, label, description }) => {
  return (
    <motion.div
      className="rounded-2xl border border-solid border-dark dark:border-light shadow-lg p-6
        bg-light dark:bg-dark"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      viewport={{ once: true }}
    >
      <span className="text-4xl font-bold text-dark dark:text-light block">
        {value}
      </span>
      <span className="text-lg font-medium text-dark dark:text-light block mt-2">
        {label}
      </span>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
        {description}
      </p>
    </motion.div>
  );
};

export default KeyResult;
