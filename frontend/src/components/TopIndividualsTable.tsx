import type { OptIndividual } from '../lib/btApi';

/** Format a fitness/number to a fixed number of decimals, with an em-dash fallback. */
function fmt(v?: number | null, n = 3): string {
  return typeof v === 'number' && isFinite(v) ? v.toFixed(n) : '–';
}

/** Render a single gene value compactly (round floats; pass through ints/strings/bools). */
function fmtVal(v: unknown): string {
  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return String(v);
}

/**
 * Compact view of the genes that distinguish individuals: the strategy-level tp/sl plus a
 * couple of expert `model:*` genes (the RM/decision settings). Other namespaces (cond:*,
 * exit:*) are skipped to keep the cell readable. Returns '' when there's nothing to show.
 */
function paramsPreview(params?: Record<string, unknown>): string {
  if (!params) return '';
  const bits: string[] = [];
  if ('tp' in params) bits.push(`tp ${fmtVal(params.tp)}`);
  if ('sl' in params) bits.push(`sl ${fmtVal(params.sl)}`);
  const modelGenes = Object.keys(params)
    .filter(k => k.startsWith('model:'))
    .slice(0, 2);
  for (const k of modelGenes) {
    bits.push(`${k.slice('model:'.length)} ${fmtVal(params[k])}`);
  }
  return bits.join(' · ');
}

/**
 * Shared top-individuals mini-table used by both the running-jobs panel (live) and the
 * Opt-History tab (completed jobs). Rows are already ranked best-first by the backend
 * (`_top_individuals`); we just render rank / fitness / trades / a compact params preview.
 */
export function TopIndividualsTable({
  individuals,
  fitnessMetric,
  note,
}: {
  individuals?: OptIndividual[];
  fitnessMetric?: string;
  note?: string;
}) {
  const top = individuals ?? [];
  if (top.length === 0) {
    return (
      <div className="text-xs text-gray-400 dark:text-gray-500">
        No individuals evaluated yet.
      </div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-400 dark:text-gray-500 text-left">
            <th className="font-medium py-1 pr-3">#</th>
            <th className="font-medium py-1 pr-3 text-right">{fitnessMetric ?? 'fitness'}</th>
            <th className="font-medium py-1 pr-3 text-right">trades</th>
            <th className="font-medium py-1 pr-3">params</th>
          </tr>
        </thead>
        <tbody>
          {top.map(ind => {
            const preview = paramsPreview(ind.params);
            return (
              <tr key={ind.rank} className="border-t border-gray-50 dark:border-gray-700/50">
                <td className="py-1 pr-3 text-gray-500 dark:text-gray-400">{ind.rank}</td>
                <td className="py-1 pr-3 text-right font-medium text-gray-800 dark:text-gray-200">{fmt(ind.fitness)}</td>
                <td className="py-1 pr-3 text-right text-gray-600 dark:text-gray-400">{ind.nTrades ?? '–'}</td>
                <td className="py-1 pr-3 font-mono text-gray-500 dark:text-gray-400">
                  {preview || <span className="text-gray-300 dark:text-gray-600">–</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {note && (
        <div className="text-[11px] text-gray-400 dark:text-gray-500 mt-1.5">{note}</div>
      )}
    </div>
  );
}
