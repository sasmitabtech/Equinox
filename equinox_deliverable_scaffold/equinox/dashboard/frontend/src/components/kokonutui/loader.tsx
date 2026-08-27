"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

interface LoaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  size?: "sm" | "md" | "lg";
}

/**
 * Adapted from Kokonut UI's installed loader registry block for Equinox's
 * low-motion, no-gradient operations context.
 */
export default function Loader({
  title = "Loading evidence",
  subtitle,
  size = "md",
  className,
  ...props
}: LoaderProps) {
  const dimensions = { sm: "size-6", md: "size-8", lg: "size-10" }[size];

  return (
    <div className={cn("flex flex-col items-center gap-3 text-center", className)} {...props}>
      <motion.span
        aria-hidden="true"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.15, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
        className={cn("block rounded-full border-2 border-stone-300 border-t-stone-800", dimensions)}
      />
      <div>
        <p className="text-sm font-medium text-stone-800">{title}</p>
        {subtitle ? <p className="mt-1 text-xs text-stone-500">{subtitle}</p> : null}
      </div>
    </div>
  );
}
