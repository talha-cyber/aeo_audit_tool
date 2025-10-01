import { describe, expect, it, vi, beforeAll, afterAll } from 'vitest';
import { formatPercent, formatRelative, formatScore } from './format';

const mockNow = new Date('2024-08-12T12:00:00.000Z').getTime();

describe('format utilities', () => {
  let dateSpy: ReturnType<typeof vi.spyOn>;

  beforeAll(() => {
    dateSpy = vi.spyOn(Date, 'now').mockReturnValue(mockNow);
  });

  afterAll(() => {
    dateSpy.mockRestore();
  });

  it('formats relative time under an hour in minutes', () => {
    const thirtyMinutesAgo = new Date(mockNow - 30 * 60 * 1000).toISOString();
    expect(formatRelative(thirtyMinutesAgo)).toBe('30m ago');
  });

  it('formats relative time over a day in days', () => {
    const twoDaysAgo = new Date(mockNow - 48 * 60 * 60 * 1000).toISOString();
    expect(formatRelative(twoDaysAgo)).toBe('2d ago');
  });

  it('returns em dash for missing relative date', () => {
    expect(formatRelative(undefined)).toBe('—');
    expect(formatRelative('not-a-date')).toBe('—');
  });

  it('formats percentage with default precision', () => {
    expect(formatPercent(0.482)).toBe('48%');
  });

  it('formats score to whole number', () => {
    expect(formatScore(74.6)).toBe('75');
  });

  it('returns em dash for missing score', () => {
    expect(formatScore(undefined)).toBe('—');
  });
});
