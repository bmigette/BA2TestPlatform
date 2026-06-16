import { useEffect, useState } from 'react';
import { Activity, XCircle } from 'lucide-react';
import { listTasks, cancelTask } from '../lib/btApi';
import type { TaskInfo } from '../lib/btApi';

const BT_TASK_TYPES = new Set(['daily_backtest', 'backtest', 'strategy_optimization']);

/** Parse "Gen 2/3 best=2.8400" (the optimizer's progress_message) into structured bits. */
function parseProgress(msg?: string): { gen?: number; total?: number; best?: string } {
  if (!msg) return {};
  const g = msg.match(/Gen\s+(\d+)\s*\/\s*(\d+)/i);
  const b = msg.match(/best\s*=\s*([-\d.]+)/i);
  return {
    gen: g ? Number(g[1]) : undefined,
    total: g ? Number(g[2]) : undefined,
    best: b ? b[1] : undefined,
  };
}

/**
 * Rich running-jobs view (its own Backtesting tab). For each in-flight backtest /
 * optimization shows the full (untruncated) name, total-generation progress bar, the
 * current generation (Gen X/N), best fitness so far, and a Cancel action. Self-polls
 * every 2s; runs independently of the rest of the page so it never blocks running jobs.
 */
export function RunningJobsPanel() {
  const [jobs, setJobs] = useState<TaskInfo[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const all = await listTasks('running');
        if (alive) setJobs(all.filter(t => !t.task_type || BT_TASK_TYPES.has(t.task_type)));
      } catch {
        if (alive) setJobs([]);
      } finally {
        if (alive) setLoaded(true);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const onCancel = (taskId: string) => {
    cancelTask(taskId)
      .then(() => setJobs(prev => prev.filter(x => x.task_id !== taskId)))
      .catch(() => { /* keep row; next poll reconciles */ });
  };

  if (loaded && !jobs.length) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-16 text-gray-400 dark:text-gray-500">
        <Activity className="w-10 h-10 mb-3 opacity-50" />
        <p className="text-sm">No running jobs.</p>
        <p className="text-xs mt-1">Submitted backtests &amp; optimizations appear here with live progress.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
        Running jobs ({jobs.length})
      </div>
      {jobs.map(j => {
        const { gen, total, best } = parseProgress(j.progress_message);
        const pct = Math.max(0, Math.min(100, Math.round(j.progress ?? 0)));
        return (
          <div key={j.task_id}
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-start justify-between gap-3 mb-3">
              <div className="min-w-0">
                <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 break-words">
                  {j.name ?? j.task_id}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  {j.task_type ?? 'job'} · {j.status}
                </div>
              </div>
              <button type="button" onClick={() => onCancel(j.task_id)}
                className="shrink-0 inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-white bg-red-600 hover:bg-red-700 rounded border border-red-700">
                <XCircle className="w-3.5 h-3.5" /> Cancel
              </button>
            </div>

            {/* Total-generation progress */}
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div className="bg-blue-500 h-2.5 rounded-full transition-all" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-xs font-medium text-gray-700 dark:text-gray-300 w-10 text-right">{pct}%</span>
            </div>

            {/* Per-generation detail */}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1 mt-2 text-xs text-gray-600 dark:text-gray-400">
              {gen != null && total != null && (
                <span>Generation <span className="font-semibold text-gray-800 dark:text-gray-200">{gen} / {total}</span></span>
              )}
              {best != null && (
                <span>Best fitness <span className="font-semibold text-emerald-600 dark:text-emerald-400">{best}</span></span>
              )}
              {gen == null && j.progress_message && (
                <span className="truncate">{j.progress_message}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
