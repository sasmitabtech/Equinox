"use client";

/** Adapted from the official Magic UI `@magicui/magic-card` registry block. */
import { useState, type ComponentPropsWithoutRef } from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

interface MagicCardProps extends ComponentPropsWithoutRef<typeof motion.div> {
  disabled?: boolean;
}

/**
 * A restrained, pointer-only focus treatment. The original registry component
 * uses a gradient spotlight; Equinox replaces it with a charcoal hairline and
 * diffuse elevation so the editorial surface remains gradient-free.
 */
function MagicCard({ className, disabled = false, children, ...props }: MagicCardProps) {
  const [active, setActive] = useState(false);
  return (
    <motion.div
      className={cn("relative rounded-md border border-stone-300 bg-[#fbfaf7]", className)}
      onPointerEnter={() => !disabled && setActive(true)}
      onPointerLeave={() => setActive(false)}
      animate={active ? { borderColor: "#5d594f", boxShadow: "0 3px 12px rgba(36, 35, 31, 0.05)" } : { borderColor: "#d6d3cc", boxShadow: "0 0 0 rgba(36, 35, 31, 0)" }}
      transition={{ type: "spring", stiffness: 420, damping: 34 }}
      {...props}
    >
      {children}
    </motion.div>
  );
}

export { MagicCard };
