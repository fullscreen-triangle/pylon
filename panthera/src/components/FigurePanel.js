import React from "react";
import Image from "next/image";
import { motion } from "framer-motion";

const FigurePanel = ({ src, caption, label }) => {
  return (
    <motion.figure
      className="w-full my-8"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      viewport={{ once: true }}
    >
      <div className="w-full rounded-2xl shadow-lg overflow-hidden">
        <Image
          src={src}
          alt={caption || label || "Figure"}
          className="w-full h-auto"
          width={1200}
          height={800}
          priority={false}
        />
      </div>
      <figcaption className="mt-3 text-center">
        {label && (
          <span className="font-semibold text-dark dark:text-light mr-2">
            {label}.
          </span>
        )}
        <span className="text-sm italic text-gray-600 dark:text-gray-400">
          {caption}
        </span>
      </figcaption>
    </motion.figure>
  );
};

export default FigurePanel;
