import { useRef } from 'react';
import { importRules } from '../lib/btApi';
import type { ConditionTree } from './ConditionBuilder';

export function RuleIO({ which, tree, onImport }:
  { which: 'enter' | 'exit'; tree: ConditionTree; onImport: (tree: ConditionTree) => void; }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const doImport = (f: File | undefined) => {
    if (!f) return;
    f.text().then((t) => importRules(JSON.parse(t), which))
      .then((tr) => onImport(tr as ConditionTree))
      .catch((e) => alert(`Import failed: ${e.message}`));
  };
  const doExport = () => {
    // Export uses the in-memory tree. When a strategy is saved, prefer the server export
    // endpoint (exportRulesUrl) for canonical v1.1 output; this direct-blob path covers the
    // unsaved-tree case (the server normalises on import).
    const blob = new Blob(
      [JSON.stringify({ export_version: '1.1', export_type: 'condition_tree', which, tree }, null, 2)],
      { type: 'application/json' },
    );
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `rules-${which}.json`;
    a.click();
  };
  return (
    <span>
      <button type="button" onClick={() => fileRef.current?.click()}>Import JSON</button>
      <button type="button" onClick={doExport}>Export JSON</button>
      <input ref={fileRef} type="file" accept=".json,application/json" style={{ display: 'none' }}
        onChange={(e) => doImport(e.target.files?.[0])} />
    </span>
  );
}
