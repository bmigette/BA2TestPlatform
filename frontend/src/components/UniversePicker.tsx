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

export function UniversePicker({ value, onChange }: { value: UniverseValue; onChange: (v: UniverseValue) => void; }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const onFile = (f: File | undefined) => {
    if (!f) return;
    f.text().then((t) => onChange({ mode: 'static', symbols: parseSymbols(t) }));
  };
  return (
    <div>
      <label><input type="radio" checked={value.mode === 'static'} onChange={() => onChange({ mode: 'static', symbols: value.mode === 'static' ? value.symbols : [] })} /> Static list</label>
      <label className="ml-3"><input type="radio" checked={value.mode === 'screener'} onChange={() => onChange({ mode: 'screener', screener_settings: value.mode === 'screener' ? value.screener_settings : {} })} /> Screener</label>

      {value.mode === 'static' ? (
        <div>
          <textarea style={{ width: '100%', height: 56 }} placeholder="AAPL, MSFT, NVDA …"
            value={value.symbols.join(', ')}
            onChange={(e) => onChange({ mode: 'static', symbols: parseSymbols(e.target.value) })} />
          <button type="button" onClick={() => fileRef.current?.click()}>⬆ Import from .txt</button>
          <input ref={fileRef} type="file" accept=".txt,text/plain" style={{ display: 'none' }}
            onChange={(e) => onFile(e.target.files?.[0])} />
          <span className="text-xs text-gray-500"> {value.symbols.length} symbols</span>
        </div>
      ) : (
        <table className="text-sm">
          <tbody>
            {SCREENER_FIELDS.map(([k, label]) => (
              <tr key={k}><td>{label}</td><td>
                <input type="number" value={Number(value.screener_settings[k] ?? 0)}
                  onChange={(e) => onChange({ mode: 'screener', screener_settings: { ...value.screener_settings, [k]: Number(e.target.value) } })} />
              </td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
