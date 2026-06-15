// frontend/src/components/ExpertPicker.tsx
import { useEffect, useState } from 'react';
import { listExperts } from '../lib/btApi';
import type { ExpertInfo } from '../lib/btApi';

export function ExpertPicker({ value, onChange }: { value: string; onChange: (cls: string, info: ExpertInfo) => void; }) {
  const [experts, setExperts] = useState<ExpertInfo[]>([]);
  useEffect(() => { listExperts().then(setExperts).catch(() => setExperts([])); }, []);
  return (
    <select value={value} onChange={(e) => {
      const info = experts.find(x => x.class === e.target.value);
      if (info) onChange(info.class, info);
    }}>
      <option value="" disabled>Select expert…</option>
      {experts.map(x => <option key={x.class} value={x.class}>{x.label}{x.bypasses_classic_rm ? ' (bypass RM)' : ''}</option>)}
    </select>
  );
}
