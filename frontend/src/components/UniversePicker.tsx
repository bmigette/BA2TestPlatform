// frontend/src/components/UniversePicker.tsx
import { useRef } from 'react';
import { parseSymbols } from '../lib/symbols';

export type UniverseValue =
  | { mode: 'static'; symbols: string[] }
  | { mode: 'screener'; screener_settings: Record<string, number | string> };

const SCREENER_FIELDS: [string, string][] = [
  ['screener_market_cap_min', 'Market cap min'], ['screener_volume_min', 'Volume min'],
  ['screener_price_min', 'Price min'], ['screener_relative_volume_min', 'RVOL min'],
  ['screener_max_stocks', 'Max stocks'],
];

const inputClass = "px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

export function UniversePicker({ value, onChange }: { value: UniverseValue; onChange: (v: UniverseValue) => void; }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const onFile = (f: File | undefined) => {
    if (!f) return;
    f.text().then((t) => onChange({ mode: 'static', symbols: parseSymbols(t) }));
  };
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-1 text-sm text-gray-700 dark:text-gray-300">
          <input type="radio" checked={value.mode === 'static'} onChange={() => onChange({ mode: 'static', symbols: value.mode === 'static' ? value.symbols : [] })} /> Static list
        </label>
        <label className="flex items-center gap-1 text-sm text-gray-700 dark:text-gray-300">
          <input type="radio" checked={value.mode === 'screener'} onChange={() => onChange({ mode: 'screener', screener_settings: value.mode === 'screener' ? value.screener_settings : {} })} /> Screener
        </label>
      </div>

      {value.mode === 'static' ? (
        <div className="space-y-2">
          <textarea className={`${inputClass} w-full`} rows={3} placeholder="AAPL, MSFT, NVDA …"
            value={value.symbols.join(', ')}
            onChange={(e) => onChange({ mode: 'static', symbols: parseSymbols(e.target.value) })} />
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => fileRef.current?.click()}
              className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-1">
              ⬆ Import from .txt
            </button>
            <input ref={fileRef} type="file" accept=".txt,text/plain" className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])} />
            <span className="text-xs text-gray-500 dark:text-gray-400">{value.symbols.length} symbols</span>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {SCREENER_FIELDS.map(([k, label]) => (
            <div key={k} className="flex items-center justify-between gap-3 p-2 bg-gray-50 dark:bg-gray-700/50 rounded border border-gray-200 dark:border-gray-600">
              <span className="flex-1 text-sm text-gray-700 dark:text-gray-300">{label}</span>
              <input type="number" className={`${inputClass} w-24`} value={Number(value.screener_settings[k] ?? 0)}
                onChange={(e) => onChange({ mode: 'screener', screener_settings: { ...value.screener_settings, [k]: Number(e.target.value) } })} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
