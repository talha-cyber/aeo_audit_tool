import clsx from 'clsx';
import { HTMLAttributes, ReactNode } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  corner?: boolean;
}

interface CardHeaderProps {
  title: string;
  description?: string;
  action?: ReactNode;
}

type CardSectionProps = HTMLAttributes<HTMLDivElement>;

export function Card({ corner = false, className, ...props }: CardProps) {
  return (
    <div
      className={clsx(
        'relative rounded-xl border border-border bg-surface p-6 shadow-sm',
        { 'before:absolute before:left-0 before:top-0 before:h-12 before:w-12 before:border-b before:border-r before:border-border before:content-[""]': corner },
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({ title, description, action }: CardHeaderProps) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div>
        <h3 className="text-base font-semibold tracking-tight text-text">{title}</h3>
        {description ? <p className="mt-1 text-sm text-muted">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function CardContent({ className, ...props }: CardSectionProps) {
  return <div className={clsx('space-y-4', className)} {...props} />;
}

export function CardFooter({ className, ...props }: CardSectionProps) {
  return <div className={clsx('mt-6 flex items-center justify-between text-sm text-muted', className)} {...props} />;
}
