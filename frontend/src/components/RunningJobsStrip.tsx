import { useEffect, useState } from 'react';
import { listTasks, cancelTask } from '../lib/btApi';
import type { TaskInfo } from '../lib/btApi';

const BT_TASK_TYPES = new Set(['daily_backtest', 'backtest', 'strategy_optimization']);

export function RunningJobsStrip() {
  const [jobs, setJobs] = useState<TaskInfo[]>([]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const all = await listTasks('running');
        if (alive) setJobs(all.filter(t => !t.task_type || BT_TASK_TYPES.has(t.task_type)));
      } catch {
        if (alive) setJobs([]);
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!jobs.length) return null;

  const onCancel = (taskId: string) => {
    cancelTask(taskId)
      .then(() => setJobs(prev => prev.filter(x => x.task_id !== taskId)))
      .catch(() => { /* keep row; next poll reconciles */ });
  };

  return (
    <div style={{ border: '1px solid #cbd5e1', borderRadius: 6, padding: 8, marginBottom: 12 }}>
      <b>Running jobs ({jobs.length})</b>
      {jobs.map(j => (
        <div key={j.task_id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
          <span style={{ width: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {j.task_type ?? 'job'} · {j.name ?? j.task_id}
          </span>
          <progress max={100} value={j.progress ?? 0} style={{ flex: 1 }} />
          <span style={{ width: 40, textAlign: 'right' }}>{Math.round(j.progress ?? 0)}%</span>
          <button type="button" onClick={() => onCancel(j.task_id)}>Cancel</button>
        </div>
      ))}
    </div>
  );
}
