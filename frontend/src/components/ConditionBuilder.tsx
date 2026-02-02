import React from 'react';
import { Plus, Trash2, ChevronDown, ChevronRight, GitBranch, Settings2 } from 'lucide-react';

// Types for condition tree structure
export interface ConditionNode {
  id: string;
  field: string;
  fieldType: string; // model_probability, model_class, position, time, price
  comparison: string; // gt, lt, eq, gte, lte, neq, between
  value: number | string | [number, number];
  optimizeEnabled: boolean;
  valueMin?: number;
  valueMax?: number;
  valueStep?: number;
  confirmationBars?: number;
  confirmationBarsMin?: number;
  confirmationBarsMax?: number;
  confirmationBarsStep?: number;
}

export interface ConditionGroup {
  id: string;
  operator: 'AND' | 'OR';
  conditions: (ConditionNode | ConditionGroup)[];
}

export type ConditionTree = ConditionNode | ConditionGroup;

export interface AvailableField {
  field: string;
  fieldType: string;
  description: string;
  category?: string;
}

// Helper to check if a tree node is a group
export function isConditionGroup(node: ConditionTree): node is ConditionGroup {
  return 'operator' in node && 'conditions' in node;
}

// Generate unique IDs
let idCounter = 0;
export function generateId(): string {
  idCounter += 1;
  return `cond_${Date.now()}_${idCounter}`;
}

// Create empty condition
export function createEmptyCondition(): ConditionNode {
  return {
    id: generateId(),
    field: '',
    fieldType: 'model_probability',
    comparison: 'gt',
    value: 0.5,
    optimizeEnabled: false,
  };
}

// Create empty group
export function createEmptyGroup(operator: 'AND' | 'OR' = 'AND'): ConditionGroup {
  return {
    id: generateId(),
    operator,
    conditions: [createEmptyCondition()],
  };
}

// Default available fields (when model not selected)
const defaultFields: AvailableField[] = [
  { field: 'position:in_position', fieldType: 'position', description: 'Currently in a position (1=yes, 0=no)', category: 'Position' },
  { field: 'position:position_pnl', fieldType: 'position', description: 'Current position P&L %', category: 'Position' },
  { field: 'position:bars_in_position', fieldType: 'position', description: 'Bars since entry', category: 'Position' },
  { field: 'time:hour', fieldType: 'time', description: 'Hour of day (0-23)', category: 'Time' },
  { field: 'time:day_of_week', fieldType: 'time', description: 'Day of week (0=Mon, 6=Sun)', category: 'Time' },
  { field: 'price:change_pct', fieldType: 'price', description: 'Price change % from previous bar', category: 'Price' },
];

const comparisonOperators = [
  { value: 'gt', label: '>' },
  { value: 'gte', label: '>=' },
  { value: 'lt', label: '<' },
  { value: 'lte', label: '<=' },
  { value: 'eq', label: '==' },
  { value: 'neq', label: '!=' },
  { value: 'between', label: 'between' },
];

interface ConditionBuilderProps {
  value: ConditionTree;
  onChange: (value: ConditionTree) => void;
  availableFields?: AvailableField[];
  isRoot?: boolean;
  level?: number;
  onRemove?: () => void;
  showOptimization?: boolean;
}

const ConditionBuilder: React.FC<ConditionBuilderProps> = ({
  value,
  onChange,
  availableFields = [],
  isRoot = true,
  level = 0,
  onRemove,
  showOptimization = true,
}) => {
  const allFields = [...defaultFields, ...availableFields];

  // Group fields by category
  const groupedFields = allFields.reduce((acc, field) => {
    const category = field.category || 'Model';
    if (!acc[category]) acc[category] = [];
    acc[category].push(field);
    return acc;
  }, {} as Record<string, AvailableField[]>);

  const [isExpanded, setIsExpanded] = React.useState(true);

  // Handle condition group
  if (isConditionGroup(value)) {
    const updateCondition = (index: number, newValue: ConditionTree) => {
      const newConditions = [...value.conditions];
      newConditions[index] = newValue;
      onChange({ ...value, conditions: newConditions });
    };

    const removeCondition = (index: number) => {
      const newConditions = value.conditions.filter((_, i) => i !== index);
      if (newConditions.length === 0) {
        // If group becomes empty and we can remove ourselves, do it
        if (onRemove) {
          onRemove();
        } else {
          // Otherwise add an empty condition
          onChange({ ...value, conditions: [createEmptyCondition()] });
        }
      } else {
        onChange({ ...value, conditions: newConditions });
      }
    };

    const addCondition = () => {
      onChange({
        ...value,
        conditions: [...value.conditions, createEmptyCondition()],
      });
    };

    const addGroup = (operator: 'AND' | 'OR') => {
      onChange({
        ...value,
        conditions: [...value.conditions, createEmptyGroup(operator)],
      });
    };

    const toggleOperator = () => {
      onChange({
        ...value,
        operator: value.operator === 'AND' ? 'OR' : 'AND',
      });
    };

    const indentClass = level > 0 ? 'ml-4 pl-4 border-l-2 border-gray-300 dark:border-gray-600' : '';

    return (
      <div className={`${indentClass} ${level > 0 ? 'mt-2' : ''}`}>
        {/* Group Header */}
        <div className="flex items-center gap-2 mb-2">
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-500" />
            )}
          </button>
          <button
            type="button"
            onClick={toggleOperator}
            className={`px-2 py-1 text-xs font-bold rounded ${
              value.operator === 'AND'
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                : 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300'
            }`}
          >
            {value.operator}
          </button>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Group ({value.conditions.length} condition{value.conditions.length !== 1 ? 's' : ''})
          </span>
          {!isRoot && onRemove && (
            <button
              type="button"
              onClick={onRemove}
              className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-500"
              title="Remove group"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Group Content */}
        {isExpanded && (
          <div className="space-y-2">
            {value.conditions.map((condition, index) => (
              <ConditionBuilder
                key={isConditionGroup(condition) ? condition.id : condition.id}
                value={condition}
                onChange={(newValue) => updateCondition(index, newValue)}
                availableFields={availableFields}
                isRoot={false}
                level={level + 1}
                onRemove={() => removeCondition(index)}
                showOptimization={showOptimization}
              />
            ))}

            {/* Add Buttons */}
            <div className="flex items-center gap-2 mt-2 ml-4">
              <button
                type="button"
                onClick={addCondition}
                className="flex items-center gap-1 px-2 py-1 text-xs text-gray-600 dark:text-gray-400 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <Plus className="w-3 h-3" />
                Condition
              </button>
              <button
                type="button"
                onClick={() => addGroup('AND')}
                className="flex items-center gap-1 px-2 py-1 text-xs text-blue-600 dark:text-blue-400 border border-blue-300 dark:border-blue-600 rounded hover:bg-blue-50 dark:hover:bg-blue-900/30"
              >
                <GitBranch className="w-3 h-3" />
                AND Group
              </button>
              <button
                type="button"
                onClick={() => addGroup('OR')}
                className="flex items-center gap-1 px-2 py-1 text-xs text-purple-600 dark:text-purple-400 border border-purple-300 dark:border-purple-600 rounded hover:bg-purple-50 dark:hover:bg-purple-900/30"
              >
                <GitBranch className="w-3 h-3" />
                OR Group
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Handle single condition
  const condition = value as ConditionNode;
  const selectedField = allFields.find((f) => f.field === condition.field);

  const updateField = (field: string, fieldType: string) => {
    onChange({ ...condition, field, fieldType });
  };

  const updateComparison = (comparison: string) => {
    const newCondition = { ...condition, comparison };
    // Reset value for between operator
    if (comparison === 'between' && !Array.isArray(condition.value)) {
      newCondition.value = [0, 1];
    } else if (comparison !== 'between' && Array.isArray(condition.value)) {
      newCondition.value = condition.value[0];
    }
    onChange(newCondition);
  };

  const updateValue = (newValue: number | string | [number, number]) => {
    onChange({ ...condition, value: newValue });
  };

  const toggleOptimize = () => {
    onChange({ ...condition, optimizeEnabled: !condition.optimizeEnabled });
  };

  const updateOptRange = (key: 'valueMin' | 'valueMax' | 'valueStep', val: number) => {
    onChange({ ...condition, [key]: val });
  };

  return (
    <div className="flex flex-wrap items-start gap-2 p-2 bg-gray-50 dark:bg-gray-700/50 rounded border border-gray-200 dark:border-gray-600">
      {/* Field Selection */}
      <div className="flex-shrink-0">
        <select
          value={condition.field}
          onChange={(e) => {
            const field = allFields.find((f) => f.field === e.target.value);
            updateField(e.target.value, field?.fieldType || 'model_probability');
          }}
          className="px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 min-w-[160px]"
        >
          <option value="">Select field...</option>
          {Object.entries(groupedFields).map(([category, fields]) => (
            <optgroup key={category} label={category}>
              {fields.map((field) => (
                <option key={field.field} value={field.field}>
                  {field.field}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      {/* Comparison Operator */}
      <div className="flex-shrink-0">
        <select
          value={condition.comparison}
          onChange={(e) => updateComparison(e.target.value)}
          className="px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          {comparisonOperators.map((op) => (
            <option key={op.value} value={op.value}>
              {op.label}
            </option>
          ))}
        </select>
      </div>

      {/* Value Input */}
      {condition.comparison === 'between' ? (
        <div className="flex items-center gap-1">
          <input
            type="number"
            step="0.01"
            value={Array.isArray(condition.value) ? condition.value[0] : 0}
            onChange={(e) =>
              updateValue([
                parseFloat(e.target.value),
                Array.isArray(condition.value) ? condition.value[1] : 1,
              ])
            }
            className="w-20 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          />
          <span className="text-sm text-gray-500">and</span>
          <input
            type="number"
            step="0.01"
            value={Array.isArray(condition.value) ? condition.value[1] : 1}
            onChange={(e) =>
              updateValue([
                Array.isArray(condition.value) ? condition.value[0] : 0,
                parseFloat(e.target.value),
              ])
            }
            className="w-20 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          />
        </div>
      ) : (
        <input
          type="number"
          step="0.01"
          value={typeof condition.value === 'number' ? condition.value : 0}
          onChange={(e) => updateValue(parseFloat(e.target.value))}
          className="w-24 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        />
      )}

      {/* Optimization Toggle */}
      {showOptimization && (
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={toggleOptimize}
            className={`p-1.5 rounded ${
              condition.optimizeEnabled
                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
            title={condition.optimizeEnabled ? 'Optimization enabled' : 'Enable optimization'}
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Remove Button */}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className="p-1.5 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-500"
          title="Remove condition"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      )}

      {/* Optimization Range (if enabled) */}
      {showOptimization && condition.optimizeEnabled && (
        <div className="w-full flex items-center gap-2 mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
          <span className="text-xs text-gray-500 dark:text-gray-400">Optimize:</span>
          <div className="flex items-center gap-1">
            <label className="text-xs text-gray-500">Min:</label>
            <input
              type="number"
              step="0.01"
              value={condition.valueMin ?? 0}
              onChange={(e) => updateOptRange('valueMin', parseFloat(e.target.value))}
              className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>
          <div className="flex items-center gap-1">
            <label className="text-xs text-gray-500">Max:</label>
            <input
              type="number"
              step="0.01"
              value={condition.valueMax ?? 1}
              onChange={(e) => updateOptRange('valueMax', parseFloat(e.target.value))}
              className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>
          <div className="flex items-center gap-1">
            <label className="text-xs text-gray-500">Step:</label>
            <input
              type="number"
              step="0.01"
              value={condition.valueStep ?? 0.1}
              onChange={(e) => updateOptRange('valueStep', parseFloat(e.target.value))}
              className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>
        </div>
      )}

      {/* Field Description */}
      {selectedField && (
        <div className="w-full text-xs text-gray-500 dark:text-gray-400 mt-1">
          {selectedField.description}
        </div>
      )}
    </div>
  );
};

export default ConditionBuilder;

// Also export the Exit Condition Builder for exit conditions with actions
export interface ExitConditionSet {
  id: string;
  name: string;
  conditions: ConditionGroup;
  action: 'close' | 'adjust_tp' | 'adjust_sl';
  actionValue?: number;
  actionValueOptimize?: boolean;
  actionValueMin?: number;
  actionValueMax?: number;
  actionValueStep?: number;
}

interface ExitConditionsBuilderProps {
  value: ExitConditionSet[];
  onChange: (value: ExitConditionSet[]) => void;
  availableFields?: AvailableField[];
  showOptimization?: boolean;
}

export const ExitConditionsBuilder: React.FC<ExitConditionsBuilderProps> = ({
  value,
  onChange,
  availableFields = [],
  showOptimization = true,
}) => {
  const addExitCondition = () => {
    const newExit: ExitConditionSet = {
      id: generateId(),
      name: `Exit Rule ${value.length + 1}`,
      conditions: createEmptyGroup('AND'),
      action: 'close',
    };
    onChange([...value, newExit]);
  };

  const updateExitCondition = (index: number, updates: Partial<ExitConditionSet>) => {
    const newValue = [...value];
    newValue[index] = { ...newValue[index], ...updates };
    onChange(newValue);
  };

  const removeExitCondition = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      {value.map((exitCond, index) => (
        <div
          key={exitCond.id}
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-3"
        >
          <div className="flex items-center justify-between mb-3">
            <input
              type="text"
              value={exitCond.name}
              onChange={(e) => updateExitCondition(index, { name: e.target.value })}
              className="px-2 py-1 text-sm font-medium border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              placeholder="Exit rule name"
            />
            <button
              type="button"
              onClick={() => removeExitCondition(index)}
              className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-500"
              title="Remove exit rule"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          {/* Conditions */}
          <div className="mb-3">
            <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">
              When these conditions are met:
            </label>
            <ConditionBuilder
              value={exitCond.conditions}
              onChange={(conds) =>
                updateExitCondition(index, { conditions: conds as ConditionGroup })
              }
              availableFields={availableFields}
              showOptimization={showOptimization}
            />
          </div>

          {/* Action */}
          <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-gray-200 dark:border-gray-600">
            <label className="text-xs text-gray-500 dark:text-gray-400">Action:</label>
            <select
              value={exitCond.action}
              onChange={(e) =>
                updateExitCondition(index, {
                  action: e.target.value as 'close' | 'adjust_tp' | 'adjust_sl',
                })
              }
              className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="close">Close Position</option>
              <option value="adjust_tp">Adjust Take Profit</option>
              <option value="adjust_sl">Adjust Stop Loss</option>
            </select>

            {(exitCond.action === 'adjust_tp' || exitCond.action === 'adjust_sl') && (
              <>
                <input
                  type="number"
                  step="0.1"
                  value={exitCond.actionValue ?? 0}
                  onChange={(e) =>
                    updateExitCondition(index, { actionValue: parseFloat(e.target.value) })
                  }
                  className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  placeholder="%"
                />
                <span className="text-xs text-gray-500">%</span>

                {showOptimization && (
                  <label className="flex items-center gap-1 text-xs text-gray-500">
                    <input
                      type="checkbox"
                      checked={exitCond.actionValueOptimize ?? false}
                      onChange={(e) =>
                        updateExitCondition(index, { actionValueOptimize: e.target.checked })
                      }
                      className="rounded"
                    />
                    Optimize
                  </label>
                )}
              </>
            )}
          </div>

          {/* Action Optimization Range */}
          {showOptimization &&
            exitCond.actionValueOptimize &&
            (exitCond.action === 'adjust_tp' || exitCond.action === 'adjust_sl') && (
              <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
                <span className="text-xs text-gray-500 dark:text-gray-400">Range:</span>
                <div className="flex items-center gap-1">
                  <label className="text-xs text-gray-500">Min:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={exitCond.actionValueMin ?? 0}
                    onChange={(e) =>
                      updateExitCondition(index, { actionValueMin: parseFloat(e.target.value) })
                    }
                    className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div className="flex items-center gap-1">
                  <label className="text-xs text-gray-500">Max:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={exitCond.actionValueMax ?? 10}
                    onChange={(e) =>
                      updateExitCondition(index, { actionValueMax: parseFloat(e.target.value) })
                    }
                    className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div className="flex items-center gap-1">
                  <label className="text-xs text-gray-500">Step:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={exitCond.actionValueStep ?? 0.5}
                    onChange={(e) =>
                      updateExitCondition(index, { actionValueStep: parseFloat(e.target.value) })
                    }
                    className="w-16 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            )}
        </div>
      ))}

      <button
        type="button"
        onClick={addExitCondition}
        className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 dark:text-gray-400 border border-dashed border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 w-full justify-center"
      >
        <Plus className="w-4 h-4" />
        Add Exit Rule
      </button>
    </div>
  );
};
