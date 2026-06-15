const API_BASE = 'http://localhost:8000/api';

export interface ExpertInfo { class: string; label: string; bypasses_classic_rm: boolean; uses_risk_manager: boolean; }
export interface SettingDef { type: string; default?: unknown; choices?: unknown[]; valid_values?: unknown[]; description?: string; tooltip?: string; }

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function jpost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const listExperts = () => jget<{ experts: ExpertInfo[] }>('/experts').then(r => r.experts);
export const getExpertSettings = (cls: string) => jget<{ definitions: Record<string, SettingDef> }>(`/experts/${cls}/settings-definitions`).then(r => r.definitions);
export const importRules = (json: unknown, which: 'enter' | 'exit') => jpost<{ tree: unknown }>('/strategies/import-rules', { json, which }).then(r => r.tree);
export const exportRulesUrl = (strategyId: number, which: 'enter' | 'exit') => `${API_BASE}/strategies/${strategyId}/export-rules?which=${which}`;
export interface TaskInfo {
  id: number;
  task_id: string;
  task_type?: string;
  name?: string;
  status: string;
  progress?: number;
  progress_message?: string;
}
export const listTasks = (status = 'running') =>
  jget<{ tasks: TaskInfo[] } | TaskInfo[]>(`/tasks?status=${status}&limit=100`)
    .then(r => (Array.isArray(r) ? r : r.tasks ?? []));
export const cancelTask = (id: string) => jpost<unknown>(`/tasks/${id}/cancel`, {});

export const listBacktests = (q: { expert?: string; optimization_id?: number; saved?: boolean } = {}) => {
  const p = new URLSearchParams();
  if (q.expert) p.set('expert', q.expert);
  if (q.optimization_id != null) p.set('optimization_id', String(q.optimization_id));
  if (q.saved != null) p.set('saved', String(q.saved));
  return jget<{ backtests: any[] }>(`/backtests?${p.toString()}`).then(r => r.backtests);
};
