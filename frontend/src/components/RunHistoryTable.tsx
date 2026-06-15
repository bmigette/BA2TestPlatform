import { useEffect, useMemo, useState } from 'react';
import { listBacktests } from '../lib/btApi';

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
    <div>
      <div className="flex gap-2 text-sm mb-2">
        <select value={expert} onChange={(e) => setExpert(e.target.value)}>
          <option value="">All experts</option>
          {experts.map(x => <option key={x}>{x}</option>)}
        </select>
        <select value={optId} onChange={(e) => setOptId(e.target.value)}>
          <option value="">All opt jobs</option>
          {optIds.map(x => <option key={x} value={x}>#{x}</option>)}
        </select>
        <input placeholder="search name" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th>id</th><th>expert</th><th>opt#</th><th>ret%</th><th>sharpe</th><th>saved</th><th>name</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(r => (
            <tr key={r.id} onClick={() => onSelect(r.id)} style={{ cursor: 'pointer' }}>
              <td>{r.id}</td>
              <td>{(r.expertName ?? r.expert_name) ?? '—'}</td>
              <td>{(r.optimizationId ?? r.optimization_id) ?? '—'}</td>
              <td>{(r.totalReturn ?? r.total_return) ?? '—'}</td>
              <td>{(r.sharpeRatio ?? r.sharpe_ratio) ?? '—'}</td>
              <td>{(r.isSaved ?? r.is_saved) ? '★' : ''}</td>
              <td>{r.name}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
