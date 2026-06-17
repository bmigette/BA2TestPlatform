import { useEffect, useMemo, useState } from 'react';
import { listOptimizationJobs } from '../lib/btApi';
import type { OptimizationJob, OptJobSettings } from '../lib/btApi';

const inputClass = "px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

/** Human-readable elapsed time between two ISO timestamps (e.g. "3h 12m", "45s"). */
function humanDuration(startIso?: string | null, endIso?: string | null): string {
  if (!startIso || !endIso) return '—';
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (!isFinite(start) || !isFinite(end) || end < start) return '—';
  let secs = Math.round((end - start) / 1000);
  if (secs < 1) return '<1s';
  const d = Math.floor(secs / 86400); secs -= d * 86400;
  const h = Math.floor(secs / 3600); secs -= h * 3600;
  const m = Math.floor(secs / 60); secs -= m * 60;
  const parts: string[] = [];
  if (d) parts.push(`${d}d`);
  if (h) parts.push(`${h}h`);
  if (m) parts.push(`${m}m`);
  if (!d && !h && (secs || !m)) parts.push(`${secs}s`);
  return parts.slice(0, 2).join(' ');
}

/** Local-time, compact date for the "Date" column (run date = created_at). */
function fmtDate(iso?: string | null): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!isFinite(t)) return '—';
  return new Date(t).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function fmtFitness(v?: number | null): string {
  return typeof v === 'number' && isFinite(v) ? v.toFixed(4) : '—';
}

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  running: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  pending: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  failed: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
};

function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300';
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded-full ${cls}`}>
      {status}
    </span>
  );
}

/** Condensed one-line preview of the optimization settings for the collapsed cell. */
function settingsPreview(s: OptJobSettings): string {
  const bits: string[] = [];
  if (s.ga.populationSize != null && s.ga.generations != null) {
    bits.push(`pop ${s.ga.populationSize} × ${s.ga.generations} gen`);
  }
  const nRanges = Object.keys(s.expertRanges || {}).length;
  if (nRanges) bits.push(`${nRanges} param${nRanges === 1 ? '' : 's'}`);
  if (s.universeMode) bits.push(s.universeMode === 'screener' ? 'screener' : s.universeMode);
  return bits.length ? bits.join(' · ') : 'no settings';
}

/** Expanded settings: GA config, optimized expert/RM param ranges, and screener settings. */
function SettingsDetail({ s }: { s: OptJobSettings }) {
  const gaEntries = Object.entries(s.ga ?? {});
  const rangeEntries = Object.entries(s.expertRanges ?? {});
  const screenerEntries = Object.entries(s.screener?.screener_settings ?? {});
  return (
    <div className="mt-2 space-y-3 text-xs">
      {/* Genetic config */}
      {gaEntries.length > 0 && (
        <div>
          <div className="font-medium text-gray-600 dark:text-gray-300 mb-1">Genetic config</div>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-gray-600 dark:text-gray-400">
            {gaEntries.map(([k, v]) => (
              <span key={k}><span className="text-gray-400 dark:text-gray-500">{k}:</span> {String(v)}</span>
            ))}
          </div>
        </div>
      )}

      {/* Backtest window / engine */}
      {(s.engine || s.startDate || s.endDate) && (
        <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-gray-600 dark:text-gray-400">
          {s.engine && <span><span className="text-gray-400 dark:text-gray-500">engine:</span> {s.engine}</span>}
          {(s.startDate || s.endDate) && (
            <span><span className="text-gray-400 dark:text-gray-500">window:</span> {s.startDate ?? '?'} → {s.endDate ?? '?'}</span>
          )}
        </div>
      )}

      {/* Optimized expert / RM param ranges */}
      <div>
        <div className="font-medium text-gray-600 dark:text-gray-300 mb-1">
          Optimized params {rangeEntries.length ? `(${rangeEntries.length})` : ''}
        </div>
        {rangeEntries.length === 0 ? (
          <div className="text-gray-400 dark:text-gray-500">None (expert frozen)</div>
        ) : (
          <table className="text-xs">
            <tbody>
              {rangeEntries.map(([name, r]) => (
                <tr key={name}>
                  <td className="pr-3 py-0.5 font-mono text-gray-700 dark:text-gray-300">{name}</td>
                  <td className="py-0.5 text-gray-500 dark:text-gray-400">
                    [{r.min ?? '?'} … {r.max ?? '?'}]
                    {r.step != null ? ` step ${r.step}` : ''}
                    {r.type ? ` · ${r.type}` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Screener settings (only when universe is screener-mode) */}
      {s.universeMode === 'screener' && (
        <div>
          <div className="font-medium text-gray-600 dark:text-gray-300 mb-1">Screener</div>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-gray-600 dark:text-gray-400">
            {s.screener?.group && <span><span className="text-gray-400 dark:text-gray-500">group:</span> {s.screener.group}</span>}
            {s.screener?.cache_db && <span className="truncate max-w-xs"><span className="text-gray-400 dark:text-gray-500">cache:</span> {s.screener.cache_db}</span>}
          </div>
          {screenerEntries.length > 0 && (
            <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-0.5 text-gray-600 dark:text-gray-400">
              {screenerEntries.map(([k, v]) => (
                <span key={k}><span className="text-gray-400 dark:text-gray-500">{k}:</span> {String(v)}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Optimization-Jobs tab: lists genetic StrategyOptimization runs (distinct from individual
 * backtests). For each job: name, status, best fitness, run date, duration (completed − created),
 * and an expandable Settings cell (GA config + optimized expert/RM ranges + screener settings).
 */
export function OptimizationJobsTable() {
  const [rows, setRows] = useState<OptimizationJob[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState('');
  const [q, setQ] = useState('');
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  useEffect(() => {
    listOptimizationJobs()
      .then(r => setRows(r))
      .catch(() => setRows([]))
      .finally(() => setLoaded(true));
  }, []);

  const statuses = useMemo(
    () => Array.from(new Set(rows.map(r => r.status).filter(Boolean))),
    [rows],
  );
  const filtered = rows.filter(r =>
    (!status || r.status === status) &&
    (!q || (r.name || '').toLowerCase().includes(q.toLowerCase())),
  );

  const toggle = (id: number) =>
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });

  if (loaded && rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-16 text-gray-400 dark:text-gray-500">
        <p className="text-sm">No optimization jobs yet.</p>
        <p className="text-xs mt-1">Launch a genetic optimization from the New Backtest tab and it will appear here.</p>
      </div>
    );
  }

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <div className="flex gap-2 p-3 bg-gray-50 dark:bg-gray-700/50 border-b border-gray-200 dark:border-gray-700">
        <select value={status} onChange={(e) => setStatus(e.target.value)} className={inputClass}>
          <option value="">All statuses</option>
          {statuses.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <input placeholder="search name" value={q} onChange={(e) => setQ(e.target.value)} className={inputClass} />
      </div>
      <table className="w-full text-sm">
        <thead className="bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">id</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">name</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">status</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">best fitness</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">date</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">duration</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">settings</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(r => {
            const isOpen = expanded.has(r.id);
            return (
              <tr key={r.id}
                className="border-b border-gray-200 dark:border-gray-600 align-top hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{r.id}</td>
                <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100 max-w-xs break-words">
                  {r.name || <span className="text-gray-400">unnamed</span>}
                  {r.fitnessMetric && (
                    <span className="block text-xs text-gray-400 dark:text-gray-500">{r.fitnessMetric}</span>
                  )}
                </td>
                <td className="px-3 py-2 text-sm">
                  <StatusBadge status={r.status} />
                  {r.status === 'failed' && r.errorMessage && (
                    <span className="block text-xs text-red-500 dark:text-red-400 mt-0.5 max-w-xs truncate" title={r.errorMessage}>
                      {r.errorMessage}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-sm font-medium text-emerald-600 dark:text-emerald-400">{fmtFitness(r.bestFitness)}</td>
                <td className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">{fmtDate(r.createdAt)}</td>
                <td className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">{humanDuration(r.createdAt, r.completedAt)}</td>
                <td className="px-3 py-2 text-sm text-gray-700 dark:text-gray-300">
                  <button
                    type="button"
                    onClick={() => toggle(r.id)}
                    className="inline-flex items-center gap-1 text-left hover:text-blue-600 dark:hover:text-blue-400"
                    title={isOpen ? 'Hide settings' : 'Show settings'}
                  >
                    <span className="text-xs">{isOpen ? '▾' : '▸'}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{settingsPreview(r.settings)}</span>
                  </button>
                  {isOpen && <SettingsDetail s={r.settings} />}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
