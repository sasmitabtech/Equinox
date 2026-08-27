"use client";

/**
 * Adapted from Kokonut UI's official `@kokonutui/carousel-cards` registry
 * component for Vite and Equinox's restrained evidence-review use case.
 */
import { useRef } from "react";
import { ArrowLeftIcon, ArrowRightIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DetectorEvidenceItem {
  src: string;
  alt: string;
  label: string;
  detail: string;
}

interface DetectorEvidenceCarouselProps {
  items: DetectorEvidenceItem[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

export default function DetectorEvidenceCarousel({ items, selectedIndex, onSelect }: DetectorEvidenceCarouselProps) {
  const strip = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();
  const move = (direction: -1 | 1) => {
    const next = (selectedIndex + direction + items.length) % items.length;
    onSelect(next);
    strip.current?.children[next]?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest", inline: "center" });
  };

  return (
    <section aria-label="Detector evidence carousel" className="mt-4">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <p className="text-sm text-stone-600">Choose a rendered prediction sample</p>
        <div className="flex gap-1">
          <Button type="button" variant="outline" size="icon-sm" className="border-stone-300 bg-white" onClick={() => move(-1)} aria-label="Previous detector sample"><ArrowLeftIcon size={16} weight="bold" /></Button>
          <Button type="button" variant="outline" size="icon-sm" className="border-stone-300 bg-white" onClick={() => move(1)} aria-label="Next detector sample"><ArrowRightIcon size={16} weight="bold" /></Button>
        </div>
      </div>
      <div ref={strip} className="mt-3 flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2" aria-label="Detector evidence samples">
        {items.map((item, index) => (
          <button
            type="button"
            key={item.src}
            aria-pressed={selectedIndex === index}
            onClick={() => onSelect(index)}
            className={cn("relative w-44 shrink-0 snap-start overflow-hidden rounded-md border bg-white text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-700", selectedIndex === index ? "border-stone-900" : "border-stone-300 hover:bg-stone-50")}
          >
            {selectedIndex === index ? <motion.span layoutId="detector-carousel-selection" transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 32 }} className="absolute inset-0 border-2 border-stone-900" aria-hidden="true" /> : null}
            <img src={item.src} alt="" className="aspect-video w-full object-cover" />
            <span className="block px-2 py-2 text-xs font-medium text-stone-800">{item.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
