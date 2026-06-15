import { useEffect, useMemo, useState } from 'react';
import { listBacktests } from '../lib/btApi';

const inputClass = "px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

export function RunHistoryTable({ savedOnly, onSelect }:
  { savedOnly: boolean; onSelect: (id: number) => void; }) {
  const [rows, setRows] = useState<any[]>([]);
  const [expert, setExpert] = useState('');
  const [optId, setOptId] = useState('');
  const [q, setQ] = useState('');

  useEffect(() => {
    listBacktests({
      saved: savedOnly ? true : undefined,
      expert: expert || undefined,
      optimization_id: optId ? Number(optId) : undefined,
    })
      .then(setRows)
      .catch(() => setRows([]));
  }, [savedOnly, expert, optId]);

  // Field names confirmed against backend/app/api/backtests.py list endpoint
  // (camelCase: expertName / optimizationId / totalReturn / sharpeRatio / isSaved).
  // snake_case fallbacks kept for robustness per the plan.
  const experts = useMemo(
    () => Array.from(new Set(rows.map(r => r.expertName ?? r.expert_name).filter(Boolean))),
    [rows],
  );
  const optIds = useMemo(
    () => Array.from(new Set(rows.map(r => r.optimizationId ?? r.optimization_id).filter((x: any) => x != null))),
    [rows],
  );
  const filtered = rows.filter(r => !q || (r.name || '').toLowerCase().includes(q.toLowerCase()));

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <div className="flex gap-2 p-3 bg-gray-50 dark:bg-gray-700/50 border-b border-gray-200 dark:border-gray-700">
        <select value={expert} onChange={(e) => setExpert(e.target.value)} className={inputClass}>
          <option value="">All experts</option>
          {experts.map(x => <option key={x}>{x}</option>)}
        </select>
        <select value={optId} onChange={(e) => setOptId(e.target.value)} className={inputClass}>
          <option value="">All opt jobs</option>
          {optIds.map(x => <option key={x} value={x}>#{x}</option>)}
        </select>
        <input placeholder="search name" value={q} onChange={(e) => setQ(e.target.value)} className={inputClass} />
      </div>
      <table className="w-full text-sm">
        <thead className="bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">id</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">expert</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">opt#</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">ret%</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">sharpe</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">saved</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">name</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(r => (
            <tr key={r.id} onClick={() => onSelect(r.id)}
              className="border-b border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors">
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{r.id}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{(r.expertName ?? r.expert_name) ?? '—'}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{(r.optimizationId ?? r.optimization_id) ?? '—'}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{(r.totalReturn ?? r.total_return) ?? '—'}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{(r.sharpeRatio ?? r.sharpe_ratio) ?? '—'}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{(r.isSaved ?? r.is_saved) ? '★' : ''}</td>
              <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{r.name}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
