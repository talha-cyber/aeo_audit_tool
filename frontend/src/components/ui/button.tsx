'use client';

import clsx from 'clsx';
import Link from 'next/link';
import type { Route } from 'next';
import { ButtonHTMLAttributes } from 'react';

type ButtonVariant = 'primary' | 'ghost';
type ButtonSize = 'sm' | 'md';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  href?: string;
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'h-9 px-3 text-sm',
  md: 'h-11 px-4 text-sm'
};

const variantStyles: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-background shadow-sm transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40',
  ghost: 'border border-border bg-surface text-text transition-colors hover:bg-elevated focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border'
};

function buildClassName(variant: ButtonVariant, size: ButtonSize, className?: string) {
  return clsx(
    'inline-flex items-center justify-center rounded-lg font-medium tracking-tight',
    sizeStyles[size],
    variantStyles[variant],
    className
  );
}

export function Button({ variant = 'primary', size = 'md', className, href, type, ...props }: ButtonProps) {
  if (href) {
    return (
      <Link href={href as Route} className={buildClassName(variant, size, className)}>
        {props.children}
      </Link>
    );
  }

  return <button type={type ?? 'button'} className={buildClassName(variant, size, className)} {...props} />;
}
