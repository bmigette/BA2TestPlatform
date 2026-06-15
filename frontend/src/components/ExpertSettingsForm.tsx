// frontend/src/components/ExpertSettingsForm.tsx
import { useEffect, useState } from 'react';
import { getExpertSettings } from '../lib/btApi';
import type { SettingDef } from '../lib/btApi';

export interface OptRange { min: number; max: number; step: number; }
export interface ExpertSettingsValue {
  settings: Record<string, unknown>;            // chosen values (fixed)
  expert_params: Record<string, OptRange & { type: string }>; // Opt-on numeric settings
}
const isNumeric = (t: string) => t === 'float' || t === 'int';

const inputClass = "px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500";
const rangeInputClass = "w-16 px-1.5 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100";

export function ExpertSettingsForm({ expertClass, value, onChange }:
  { expertClass: string; value: ExpertSettingsValue; onChange: (v: ExpertSettingsValue) => void; }) {
  const [defs, setDefs] = useState<Record<string, SettingDef>>({});
  useEffect(() => {
    if (!expertClass) return;
    getExpertSettings(expertClass).then((d) => {
      setDefs(d);
      // seed defaults for any setting not yet set
      const settings = { ...value.settings };
      for (const [k, def] of Object.entries(d)) if (!(k in settings) && def.default !== undefined) settings[k] = def.default;
      onChange({ ...value, settings });
    }).catch(() => setDefs({}));
  }, [expertClass]);

  const setVal = (k: string, v: unknown) => onChange({ ...value, settings: { ...value.settings, [k]: v } });
  const setOpt = (k: string, type: string, on: boolean, range?: Partial<OptRange>) => {
    const ep = { ...value.expert_params };
    if (on) ep[k] = { type, min: range?.min ?? 0, max: range?.max ?? 0, step: range?.step ?? 0 };
    else delete ep[k];
    onChange({ ...value, expert_params: ep });
  };

  return (
    <div className="space-y-2">
      {Object.entries(defs).map(([k, def]) => {
        const choices = (def.choices ?? def.valid_values) as unknown[] | undefined;
        const opt = value.expert_params[k];
        return (
          <div
            key={k}
            title={def.tooltip || def.description || ''}
            className="flex items-center justify-between gap-3 p-2 bg-gray-50 dark:bg-gray-700/50 rounded border border-gray-200 dark:border-gray-600"
          >
            <span className="flex-1 text-sm text-gray-700 dark:text-gray-300">{k}</span>
            <div>
              {def.type === 'bool' ? (
                <input type="checkbox" className="rounded" checked={!!value.settings[k]} onChange={(e) => setVal(k, e.target.checked)} />
              ) : choices ? (
                <select className={inputClass} value={String(value.settings[k] ?? '')} onChange={(e) => setVal(k, e.target.value)}>
                  {choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
                </select>
              ) : (
                <input className={inputClass} type={isNumeric(def.type) ? 'number' : 'text'}
                  value={String(value.settings[k] ?? '')}
                  onChange={(e) => setVal(k, isNumeric(def.type) ? Number(e.target.value) : e.target.value)} />
              )}
            </div>
            <div>
              {isNumeric(def.type) && (
                <label className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
                  <input type="checkbox" className="rounded" checked={!!opt}
                    onChange={(e) => setOpt(k, def.type, e.target.checked)} /> Opt
                  {opt && (<>
                    <input type="number" placeholder="min" value={opt.min} className={rangeInputClass}
                      onChange={(e) => setOpt(k, def.type, true, { ...opt, min: Number(e.target.value) })} />
                    <input type="number" placeholder="max" value={opt.max} className={rangeInputClass}
                      onChange={(e) => setOpt(k, def.type, true, { ...opt, max: Number(e.target.value) })} />
                    <input type="number" placeholder="step" value={opt.step} className={rangeInputClass}
                      onChange={(e) => setOpt(k, def.type, true, { ...opt, step: Number(e.target.value) })} />
                  </>)}
                </label>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
