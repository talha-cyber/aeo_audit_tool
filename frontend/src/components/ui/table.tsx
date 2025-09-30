import clsx from 'clsx';
import { ReactNode } from 'react';

export interface TableColumn<T> {
  key: keyof T;
  header: string;
  render?: (row: T) => ReactNode;
  width?: string;
}

interface TableProps<T> {
  data: T[];
  columns: TableColumn<T>[];
  caption?: string;
  emptyState?: ReactNode;
}

export function Table<T extends Record<string, unknown>>({
  data,
  columns,
  caption,
  emptyState
}: TableProps<T>) {
  if (!data.length && emptyState) {
    return <div>{emptyState}</div>;
  }

  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <table className="min-w-full divide-y divide-border bg-surface">
        {caption ? <caption className="px-4 py-3 text-left text-sm text-muted">{caption}</caption> : null}
        <thead className="bg-elevated text-xs uppercase tracking-[0.18em] text-muted">
          <tr>
            {columns.map((column) => (
              <th
                key={String(column.key)}
                scope="col"
                className={clsx('px-4 py-3 text-left font-medium', column.width)}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border text-sm text-text">
          {data.map((row, index) => (
            <tr key={index} className="hover:bg-elevated/60">
              {columns.map((column) => (
                <td key={String(column.key)} className="px-4 py-3 align-top">
                  {column.render ? column.render(row) : String(row[column.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
