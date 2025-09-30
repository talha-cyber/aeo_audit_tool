interface KpiProps {
  label: string;
  value: string;
  delta?: {
    value: string;
    trend: 'up' | 'down' | 'flat';
  };
}

const trendSymbol: Record<'up' | 'down' | 'flat', string> = {
  up: '▲',
  down: '▼',
  flat: '■'
};

export function Kpi({ label, value, delta }: KpiProps) {
  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-text">{value}</p>
      {delta ? (
        <p className="mt-1 text-xs text-muted">
          {trendSymbol[delta.trend]} {delta.value} vs. last period
        </p>
      ) : null}
    </div>
  );
}
