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
    <table className="w-full text-sm">
      <tbody>
        {Object.entries(defs).map(([k, def]) => {
          const choices = (def.choices ?? def.valid_values) as unknown[] | undefined;
          const opt = value.expert_params[k];
          return (
            <tr key={k} title={def.tooltip || def.description || ''}>
              <td className="pr-2">{k}</td>
              <td>
                {def.type === 'bool' ? (
                  <input type="checkbox" checked={!!value.settings[k]} onChange={(e) => setVal(k, e.target.checked)} />
                ) : choices ? (
                  <select value={String(value.settings[k] ?? '')} onChange={(e) => setVal(k, e.target.value)}>
                    {choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
                  </select>
                ) : (
                  <input type={isNumeric(def.type) ? 'number' : 'text'}
                    value={String(value.settings[k] ?? '')}
                    onChange={(e) => setVal(k, isNumeric(def.type) ? Number(e.target.value) : e.target.value)} />
                )}
              </td>
              <td>
                {isNumeric(def.type) && (
                  <label className="text-xs flex gap-1 items-center">
                    <input type="checkbox" checked={!!opt}
                      onChange={(e) => setOpt(k, def.type, e.target.checked)} /> Opt
                    {opt && (<>
                      <input type="number" placeholder="min" value={opt.min}
                        onChange={(e) => setOpt(k, def.type, true, { ...opt, min: Number(e.target.value) })} style={{ width: 60 }} />
                      <input type="number" placeholder="max" value={opt.max}
                        onChange={(e) => setOpt(k, def.type, true, { ...opt, max: Number(e.target.value) })} style={{ width: 60 }} />
                      <input type="number" placeholder="step" value={opt.step}
                        onChange={(e) => setOpt(k, def.type, true, { ...opt, step: Number(e.target.value) })} style={{ width: 60 }} />
                    </>)}
                  </label>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
