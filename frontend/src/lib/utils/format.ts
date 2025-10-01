export function formatRelative(isoDate?: string | null) {
  if (!isoDate) {
    return '—';
  }

  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) {
    return '—';
  }
  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.round(diffMs / 60000);

  if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  }
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours}h ago`;
  }
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays}d ago`;
}

export function formatPercent(value: number, fractionDigits = 0) {
  return `${(value * 100).toFixed(fractionDigits)}%`;
}

export function formatScore(value?: number | null) {
  if (value === null || value === undefined) {
    return '—';
  }
  return `${value.toFixed(0)}`;
}
