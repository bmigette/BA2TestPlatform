import React, { useState, useEffect } from 'react';
import { X, Database, Cpu, Target, Trash2, Split, Save, FolderOpen, Play, AlertTriangle, ChevronRight, ChevronLeft, Loader2, Activity, Zap } from 'lucide-react';

interface Dataset {
  id: number;
  name: string;
  ticker: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  rows_count: number;
  created_at: string;
}

interface ParameterRanges {
  layersMin: number;
  layersMax: number;
  layersStep: number;
  layerSizeMin: number;
  layerSizeMax: number;
  layerSizeStep: number;
  learningRateMin: number;
  learningRateMax: number;
  learningRateStep: number;
  dropoutMin: number;
  dropoutMax: number;
  dropoutStep: number;
  activationFunctions: string[];
}

interface GeneticConfig {
  populationSize: number;
  generations: number;
  elitismPercent: number;
  crossoverProb: number;
  mutationProb: number;
  earlyStoppingGenerations: number;
  trainingEpochs: number;
}

interface MetricsConfig {
  optimizeMetric: string;
}

interface PredictionTarget {
  id: string;
  profitPercent: number;
  maxDrawdownPercent: number;
  timePeriodDays: number;
}

interface JobProfile {
  id: number;
  name: string;
  createdAt: string;
  updatedAt?: string;
  selectedModels: string[];
  parameterRanges: ParameterRanges;
  predictionTargets: Omit<PredictionTarget, 'id'>[];
  trainTestSplit: number;
  geneticConfig?: GeneticConfig;
  metricsConfig?: MetricsConfig;
}

interface TargetPreview {
  name: string;
  label: string;
  train_positive: number;
  train_negative: number;
  train_positive_pct: number;
  test_positive: number;
  test_negative: number;
  test_positive_pct: number;
  warnings: string[];
}

interface PreviewResponse {
  dataset_id: number;
  dataset_rows: number;
  train_rows: number;
  test_rows: number;
  targets: TargetPreview[];
}

interface JobWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (job: any) => void;
  datasets: Dataset[];
  profiles: JobProfile[];
  onSaveProfile: (name: string, data: any) => Promise<void>;
  onDeleteProfile: (profileId: number) => Promise<void>;
}

const MODEL_TYPES = [
  { id: 'lstm', name: 'LSTM', description: 'Long Short-Term Memory' },
  { id: 'gru', name: 'GRU', description: 'Gated Recurrent Unit' },
  { id: 'nbeats', name: 'N-BEATS', description: 'Neural Basis Expansion Analysis' },
  { id: 'tcn', name: 'TCN', description: 'Temporal Convolutional Network' },
  { id: 'transformer', name: 'Transformer', description: 'Standard Transformer' },
  { id: 'tft', name: 'TFT', description: 'Temporal Fusion Transformer (Google)' },
];

const CLASSIFICATION_METRICS = [
  { id: 'f1_score', name: 'F1 Score', description: 'Harmonic mean of precision and recall' },
  { id: 'accuracy', name: 'Accuracy', description: 'Overall correctness' },
  { id: 'balanced_accuracy', name: 'Balanced Accuracy', description: 'Average of recall per class' },
  { id: 'precision', name: 'Precision', description: 'Minimize false positives' },
  { id: 'recall', name: 'Recall', description: 'Minimize false negatives' },
  { id: 'mcc', name: 'MCC', description: 'Matthews Correlation Coefficient' },
];

const PREDICTION_PRESETS = [
  { label: '10% / 5% DD / 7d', profit: 10, drawdown: 5, days: 7 },
  { label: '20% / 10% DD / 30d', profit: 20, drawdown: 10, days: 30 },
  { label: '5% / 3% DD / 3d', profit: 5, drawdown: 3, days: 3 },
  { label: '15% / 7% DD / 14d', profit: 15, drawdown: 7, days: 14 },
];

const getDefaultState = () => ({
  selectedDatasetId: null as number | null,
  selectedModels: [] as string[],
  parameterRanges: {
    layersMin: 2,
    layersMax: 4,
    layersStep: 1,
    layerSizeMin: 32,
    layerSizeMax: 128,
    layerSizeStep: 16,
    learningRateMin: 0.001,
    learningRateMax: 0.01,
    learningRateStep: 0.001,
    dropoutMin: 0.0,
    dropoutMax: 0.5,
    dropoutStep: 0.1,
    activationFunctions: ['relu'],
  } as ParameterRanges,
  geneticConfig: {
    populationSize: 20,
    generations: 50,
    elitismPercent: 10,
    crossoverProb: 0.7,
    mutationProb: 0.2,
    earlyStoppingGenerations: 5,
    trainingEpochs: 10,
  } as GeneticConfig,
  metricsConfig: {
    optimizeMetric: 'f1_score',
  } as MetricsConfig,
  predictionTargets: [] as PredictionTarget[],
  trainTestSplit: 80,
});

const JobWizard: React.FC<JobWizardProps> = ({
  isOpen,
  onClose,
  onComplete,
  datasets,
  profiles,
  onSaveProfile,
  onDeleteProfile,
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [state, setState] = useState(getDefaultState());
  const [showCustomTargetForm, setShowCustomTargetForm] = useState(false);
  const [customTarget, setCustomTarget] = useState({ profitPercent: 15, maxDrawdownPercent: 7, timePeriodDays: 14 });
  const [showLoadProfileDialog, setShowLoadProfileDialog] = useState(false);
  const [showSaveProfileDialog, setShowSaveProfileDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewData, setPreviewData] = useState<PreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Reset when opened
  useEffect(() => {
    if (isOpen) {
      setState(getDefaultState());
      setCurrentStep(1);
      setPreviewData(null);
      setPreviewError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const selectedDataset = datasets.find(d => d.id === state.selectedDatasetId);

  const isParameterValid = () => {
    return (
      state.parameterRanges.layersMin <= state.parameterRanges.layersMax &&
      state.parameterRanges.layerSizeMin <= state.parameterRanges.layerSizeMax &&
      state.parameterRanges.learningRateMin <= state.parameterRanges.learningRateMax &&
      state.parameterRanges.dropoutMin <= state.parameterRanges.dropoutMax &&
      state.parameterRanges.activationFunctions.length > 0
    );
  };

  const isStep1Valid = () => {
    return (
      state.selectedDatasetId !== null &&
      state.selectedModels.length > 0 &&
      isParameterValid() &&
      state.predictionTargets.length > 0
    );
  };

  const calculateCombinations = () => {
    const { parameterRanges, selectedModels } = state;
    const layersCount = Math.max(1, Math.floor((parameterRanges.layersMax - parameterRanges.layersMin) / parameterRanges.layersStep) + 1);
    const layerSizeCount = Math.max(1, Math.floor((parameterRanges.layerSizeMax - parameterRanges.layerSizeMin) / parameterRanges.layerSizeStep) + 1);
    const lrCount = Math.max(1, Math.floor((parameterRanges.learningRateMax - parameterRanges.learningRateMin) / parameterRanges.learningRateStep) + 1);
    const dropoutCount = Math.max(1, Math.floor((parameterRanges.dropoutMax - parameterRanges.dropoutMin) / parameterRanges.dropoutStep) + 1);
    const activationCount = parameterRanges.activationFunctions.length;
    const modelCount = selectedModels.length || 1;
    return layersCount * layerSizeCount * lrCount * dropoutCount * activationCount * modelCount;
  };

  const handleModelToggle = (modelId: string) => {
    setState(prev => ({
      ...prev,
      selectedModels: prev.selectedModels.includes(modelId)
        ? prev.selectedModels.filter(id => id !== modelId)
        : [...prev.selectedModels, modelId]
    }));
  };

  const handleAllModelsToggle = () => {
    setState(prev => ({
      ...prev,
      selectedModels: prev.selectedModels.length === MODEL_TYPES.length
        ? []
        : MODEL_TYPES.map(m => m.id)
    }));
  };

  const addPresetTarget = (preset: typeof PREDICTION_PRESETS[0]) => {
    const exists = state.predictionTargets.some(
      t => t.profitPercent === preset.profit &&
           t.maxDrawdownPercent === preset.drawdown &&
           t.timePeriodDays === preset.days
    );
    if (!exists) {
      setState(prev => ({
        ...prev,
        predictionTargets: [...prev.predictionTargets, {
          id: `target_${Date.now()}`,
          profitPercent: preset.profit,
          maxDrawdownPercent: preset.drawdown,
          timePeriodDays: preset.days,
        }]
      }));
    }
  };

  const removeTarget = (targetId: string) => {
    setState(prev => ({
      ...prev,
      predictionTargets: prev.predictionTargets.filter(t => t.id !== targetId)
    }));
  };

  const addCustomTarget = () => {
    const exists = state.predictionTargets.some(
      t => t.profitPercent === customTarget.profitPercent &&
           t.maxDrawdownPercent === customTarget.maxDrawdownPercent &&
           t.timePeriodDays === customTarget.timePeriodDays
    );
    if (!exists && customTarget.profitPercent > 0 && customTarget.maxDrawdownPercent > 0 && customTarget.timePeriodDays > 0) {
      setState(prev => ({
        ...prev,
        predictionTargets: [...prev.predictionTargets, {
          id: `target_${Date.now()}`,
          ...customTarget
        }]
      }));
      setShowCustomTargetForm(false);
      setCustomTarget({ profitPercent: 15, maxDrawdownPercent: 7, timePeriodDays: 14 });
    }
  };

  const loadProfile = (profile: JobProfile) => {
    setState(prev => ({
      ...prev,
      selectedModels: profile.selectedModels || [],
      parameterRanges: profile.parameterRanges || prev.parameterRanges,
      predictionTargets: (profile.predictionTargets || []).map((t, idx) => ({
        ...t,
        id: `target_${Date.now()}_${idx}`,
      })),
      trainTestSplit: profile.trainTestSplit || 80,
      geneticConfig: profile.geneticConfig || prev.geneticConfig,
      metricsConfig: profile.metricsConfig || prev.metricsConfig,
    }));
    setShowLoadProfileDialog(false);
  };

  const saveProfile = async () => {
    if (!newProfileName.trim()) return;
    await onSaveProfile(newProfileName.trim(), {
      selectedModels: state.selectedModels,
      parameterRanges: state.parameterRanges,
      predictionTargets: state.predictionTargets.map(({ profitPercent, maxDrawdownPercent, timePeriodDays }) => ({
        profitPercent, maxDrawdownPercent, timePeriodDays
      })),
      trainTestSplit: state.trainTestSplit,
      geneticConfig: state.geneticConfig,
      metricsConfig: state.metricsConfig,
    });
    setNewProfileName('');
    setShowSaveProfileDialog(false);
  };

  const fetchPreview = async () => {
    if (!state.selectedDatasetId || state.predictionTargets.length === 0) return;

    setPreviewLoading(true);
    setPreviewError(null);

    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${state.selectedDatasetId}/preview-targets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          targets: state.predictionTargets.map(t => ({
            profitPercent: t.profitPercent,
            maxDrawdownPercent: t.maxDrawdownPercent,
            timePeriodDays: t.timePeriodDays,
          })),
          trainRatio: state.trainTestSplit / 100,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch preview');
      }

      const data = await response.json();
      setPreviewData(data);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleNext = async () => {
    if (currentStep === 1) {
      await fetchPreview();
      setCurrentStep(2);
    }
  };

  const handleBack = () => {
    if (currentStep === 2) {
      setCurrentStep(1);
    }
  };

  const submitJob = async () => {
    if (!isStep1Valid()) return;

    setIsSubmitting(true);
    try {
      const response = await fetch('http://localhost:8002/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetId: state.selectedDatasetId,
          selectedModels: state.selectedModels,
          parameterRanges: state.parameterRanges,
          predictionTargets: state.predictionTargets.map(({ profitPercent, maxDrawdownPercent, timePeriodDays }) => ({
            profitPercent, maxDrawdownPercent, timePeriodDays
          })),
          trainTestSplit: state.trainTestSplit,
          geneticConfig: state.geneticConfig,
          metricsConfig: state.metricsConfig,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create job');
      }

      const newJob = await response.json();
      onComplete(newJob);
      onClose();
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : 'Failed to create job');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="fixed inset-0 bg-black bg-opacity-50" onClick={onClose} />

      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                New Optimization Job
              </h2>
              {/* Step indicator */}
              <div className="flex items-center space-x-2">
                <div className={`flex items-center space-x-1 px-3 py-1 rounded-full text-sm ${
                  currentStep === 1 ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'
                }`}>
                  <span className="font-medium">1</span>
                  <span>Settings</span>
                </div>
                <ChevronRight size={16} className="text-gray-400" />
                <div className={`flex items-center space-x-1 px-3 py-1 rounded-full text-sm ${
                  currentStep === 2 ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'
                }`}>
                  <span className="font-medium">2</span>
                  <span>Summary</span>
                </div>
              </div>
            </div>
            <button onClick={onClose} className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto flex-1">
            {currentStep === 1 ? (
              <Step1Settings
                state={state}
                setState={setState}
                datasets={datasets}
                selectedDataset={selectedDataset}
                profiles={profiles}
                showLoadProfileDialog={showLoadProfileDialog}
                setShowLoadProfileDialog={setShowLoadProfileDialog}
                showSaveProfileDialog={showSaveProfileDialog}
                setShowSaveProfileDialog={setShowSaveProfileDialog}
                showCustomTargetForm={showCustomTargetForm}
                setShowCustomTargetForm={setShowCustomTargetForm}
                customTarget={customTarget}
                setCustomTarget={setCustomTarget}
                newProfileName={newProfileName}
                setNewProfileName={setNewProfileName}
                handleModelToggle={handleModelToggle}
                handleAllModelsToggle={handleAllModelsToggle}
                addPresetTarget={addPresetTarget}
                removeTarget={removeTarget}
                addCustomTarget={addCustomTarget}
                loadProfile={loadProfile}
                saveProfile={saveProfile}
                onDeleteProfile={onDeleteProfile}
                calculateCombinations={calculateCombinations}
              />
            ) : (
              <Step2Summary
                state={state}
                selectedDataset={selectedDataset}
                previewData={previewData}
                previewLoading={previewLoading}
                previewError={previewError}
                calculateCombinations={calculateCombinations}
              />
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-4 border-t border-gray-200 dark:border-gray-700">
            <div>
              {currentStep === 2 && (
                <button
                  onClick={handleBack}
                  className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 flex items-center space-x-2"
                >
                  <ChevronLeft size={16} />
                  <span>Back</span>
                </button>
              )}
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={onClose}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                Cancel
              </button>
              {currentStep === 1 ? (
                <button
                  onClick={handleNext}
                  disabled={!isStep1Valid()}
                  className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                    isStep1Valid()
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  <span>Next</span>
                  <ChevronRight size={16} />
                </button>
              ) : (
                <button
                  onClick={submitJob}
                  disabled={isSubmitting}
                  className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                    !isSubmitting
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      <span>Starting...</span>
                    </>
                  ) : (
                    <>
                      <Play size={16} />
                      <span>Start Optimization</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Step 1: Settings Component
interface Step1Props {
  state: ReturnType<typeof getDefaultState>;
  setState: React.Dispatch<React.SetStateAction<ReturnType<typeof getDefaultState>>>;
  datasets: Dataset[];
  selectedDataset: Dataset | undefined;
  profiles: JobProfile[];
  showLoadProfileDialog: boolean;
  setShowLoadProfileDialog: (v: boolean) => void;
  showSaveProfileDialog: boolean;
  setShowSaveProfileDialog: (v: boolean) => void;
  showCustomTargetForm: boolean;
  setShowCustomTargetForm: (v: boolean) => void;
  customTarget: { profitPercent: number; maxDrawdownPercent: number; timePeriodDays: number };
  setCustomTarget: (v: { profitPercent: number; maxDrawdownPercent: number; timePeriodDays: number }) => void;
  newProfileName: string;
  setNewProfileName: (v: string) => void;
  handleModelToggle: (id: string) => void;
  handleAllModelsToggle: () => void;
  addPresetTarget: (preset: typeof PREDICTION_PRESETS[0]) => void;
  removeTarget: (id: string) => void;
  addCustomTarget: () => void;
  loadProfile: (profile: JobProfile) => void;
  saveProfile: () => void;
  onDeleteProfile: (id: number) => Promise<void>;
  calculateCombinations: () => number;
}

const Step1Settings: React.FC<Step1Props> = ({
  state,
  setState,
  datasets,
  selectedDataset,
  profiles,
  showLoadProfileDialog,
  setShowLoadProfileDialog,
  showSaveProfileDialog,
  setShowSaveProfileDialog,
  showCustomTargetForm,
  setShowCustomTargetForm,
  customTarget,
  setCustomTarget,
  newProfileName,
  setNewProfileName,
  handleModelToggle,
  handleAllModelsToggle,
  addPresetTarget,
  removeTarget,
  addCustomTarget,
  loadProfile,
  saveProfile,
  onDeleteProfile,
  calculateCombinations,
}) => {
  const allModelsSelected = state.selectedModels.length === MODEL_TYPES.length;
  const formatDate = (dateString: string) => new Date(dateString).toLocaleDateString();

  return (
    <div className="space-y-6">
      {/* Profile buttons */}
      <div className="flex justify-end space-x-2">
        <button
          onClick={() => setShowLoadProfileDialog(true)}
          className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-1"
        >
          <FolderOpen size={14} />
          <span>Load Profile</span>
        </button>
        <button
          onClick={() => setShowSaveProfileDialog(true)}
          className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-1"
        >
          <Save size={14} />
          <span>Save Profile</span>
        </button>
      </div>

      {/* Dataset Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Dataset
        </label>
        <select
          value={state.selectedDatasetId || ''}
          onChange={(e) => setState(prev => ({ ...prev, selectedDatasetId: e.target.value ? Number(e.target.value) : null }))}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-green-500"
        >
          <option value="">Choose a dataset...</option>
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name} ({dataset.ticker} - {dataset.timeframe} - {dataset.rows_count.toLocaleString()} rows)
            </option>
          ))}
        </select>
      </div>

      {/* Dataset Details */}
      {selectedDataset && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Ticker:</span> <span className="font-medium">{selectedDataset.ticker}</span></div>
            <div><span className="text-gray-500">Timeframe:</span> <span className="font-medium">{selectedDataset.timeframe}</span></div>
            <div><span className="text-gray-500">Rows:</span> <span className="font-medium">{selectedDataset.rows_count.toLocaleString()}</span></div>
            <div><span className="text-gray-500">Range:</span> <span className="font-medium">{formatDate(selectedDataset.start_date)} - {formatDate(selectedDataset.end_date)}</span></div>
          </div>
        </div>
      )}

      {/* Model Types */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Select Model Types</label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={allModelsSelected}
              onChange={handleAllModelsToggle}
              className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">All Models</span>
          </label>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {MODEL_TYPES.map((model) => (
            <label
              key={model.id}
              className={`flex items-start space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                state.selectedModels.includes(model.id)
                  ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
              }`}
            >
              <input
                type="checkbox"
                checked={state.selectedModels.includes(model.id)}
                onChange={() => handleModelToggle(model.id)}
                className="mt-1 w-4 h-4 text-green-600 border-gray-300 rounded"
              />
              <div>
                <div className="font-medium text-sm">{model.name}</div>
                <span className="text-xs text-gray-500">{model.description}</span>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Prediction Targets */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <Target size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Prediction Targets</label>
        </div>
        <div className="flex flex-wrap gap-2 mb-3">
          {PREDICTION_PRESETS.map((preset) => (
            <button
              key={preset.label}
              onClick={() => addPresetTarget(preset)}
              className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              {preset.label}
            </button>
          ))}
          <button
            onClick={() => setShowCustomTargetForm(true)}
            className="px-3 py-1.5 text-sm bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-md hover:bg-green-200"
          >
            + Custom
          </button>
        </div>

        {showCustomTargetForm && (
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-3">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Profit %</label>
                <input
                  type="number"
                  value={customTarget.profitPercent}
                  onChange={(e) => setCustomTarget({ ...customTarget, profitPercent: Number(e.target.value) })}
                  className="w-full px-2 py-1 border rounded text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Max DD %</label>
                <input
                  type="number"
                  value={customTarget.maxDrawdownPercent}
                  onChange={(e) => setCustomTarget({ ...customTarget, maxDrawdownPercent: Number(e.target.value) })}
                  className="w-full px-2 py-1 border rounded text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Days</label>
                <input
                  type="number"
                  value={customTarget.timePeriodDays}
                  onChange={(e) => setCustomTarget({ ...customTarget, timePeriodDays: Number(e.target.value) })}
                  className="w-full px-2 py-1 border rounded text-sm"
                />
              </div>
            </div>
            <div className="flex justify-end mt-3 space-x-2">
              <button onClick={() => setShowCustomTargetForm(false)} className="px-3 py-1 text-sm text-gray-500">Cancel</button>
              <button onClick={addCustomTarget} className="px-3 py-1 text-sm bg-green-600 text-white rounded">Add</button>
            </div>
          </div>
        )}

        {state.predictionTargets.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {state.predictionTargets.map((target) => (
              <div key={target.id} className="flex items-center space-x-2 px-3 py-1.5 bg-green-50 dark:bg-green-900/20 border border-green-500 rounded-full text-sm">
                <span>{target.profitPercent}% / {target.maxDrawdownPercent}% DD / {target.timePeriodDays}d</span>
                <button onClick={() => removeTarget(target.id)} className="text-red-500 hover:text-red-700">
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Train/Test Split */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <Split size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Train/Test Split</label>
        </div>
        <div className="flex items-center space-x-4">
          <input
            type="range"
            min="50"
            max="90"
            value={state.trainTestSplit}
            onChange={(e) => setState(prev => ({ ...prev, trainTestSplit: Number(e.target.value) }))}
            className="flex-1"
          />
          <div className="text-sm font-medium w-24 text-center">
            {state.trainTestSplit}% / {100 - state.trainTestSplit}%
          </div>
        </div>
        {selectedDataset && (
          <p className="text-xs text-gray-500 mt-1">
            Train: {Math.floor(selectedDataset.rows_count * state.trainTestSplit / 100).toLocaleString()} rows,
            Test: {Math.floor(selectedDataset.rows_count * (100 - state.trainTestSplit) / 100).toLocaleString()} rows
          </p>
        )}
      </div>

      {/* Genetic Algorithm Config */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <Activity size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Genetic Algorithm</label>
        </div>
        <div className="grid grid-cols-4 gap-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
          <div>
            <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Population</label>
            <input
              type="number"
              value={state.geneticConfig.populationSize}
              onChange={(e) => setState(prev => ({ ...prev, geneticConfig: { ...prev.geneticConfig, populationSize: Number(e.target.value) } }))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Generations</label>
            <input
              type="number"
              value={state.geneticConfig.generations}
              onChange={(e) => setState(prev => ({ ...prev, geneticConfig: { ...prev.geneticConfig, generations: Number(e.target.value) } }))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Epochs/Individual</label>
            <input
              type="number"
              value={state.geneticConfig.trainingEpochs}
              onChange={(e) => setState(prev => ({ ...prev, geneticConfig: { ...prev.geneticConfig, trainingEpochs: Number(e.target.value) } }))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Early Stop (gens)</label>
            <input
              type="number"
              value={state.geneticConfig.earlyStoppingGenerations}
              onChange={(e) => setState(prev => ({ ...prev, geneticConfig: { ...prev.geneticConfig, earlyStoppingGenerations: Number(e.target.value) } }))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
        </div>
      </div>

      {/* Optimization Metric */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <Zap size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Optimization Metric</label>
        </div>
        <div className="flex flex-wrap gap-2">
          {CLASSIFICATION_METRICS.map((metric) => (
            <label
              key={metric.id}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer text-sm ${
                state.metricsConfig.optimizeMetric === metric.id
                  ? 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700'
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
              }`}
            >
              <input
                type="radio"
                name="metric"
                checked={state.metricsConfig.optimizeMetric === metric.id}
                onChange={() => setState(prev => ({ ...prev, metricsConfig: { optimizeMetric: metric.id } }))}
                className="sr-only"
              />
              <span>{metric.name}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Total Parameter Combinations:</span>
          <span className="font-bold text-blue-600">{calculateCombinations().toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between mt-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">Total Individuals to Evaluate:</span>
          <span className="font-bold text-blue-600">
            {(state.geneticConfig.populationSize * state.geneticConfig.generations).toLocaleString()}
          </span>
        </div>
      </div>

      {/* Load Profile Dialog */}
      {showLoadProfileDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4">Load Profile</h3>
            {profiles.length === 0 ? (
              <p className="text-gray-500">No saved profiles</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {profiles.map((profile) => (
                  <div key={profile.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <button onClick={() => loadProfile(profile)} className="text-left flex-1">
                      <div className="font-medium">{profile.name}</div>
                      <div className="text-xs text-gray-500">{profile.selectedModels?.length || 0} models</div>
                    </button>
                    <button onClick={() => onDeleteProfile(profile.id)} className="text-red-500 p-1">
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex justify-end mt-4">
              <button onClick={() => setShowLoadProfileDialog(false)} className="px-4 py-2 text-gray-600">Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Save Profile Dialog */}
      {showSaveProfileDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4">Save Profile</h3>
            <input
              type="text"
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              placeholder="Profile name..."
              className="w-full px-3 py-2 border rounded mb-4"
            />
            <div className="flex justify-end space-x-2">
              <button onClick={() => setShowSaveProfileDialog(false)} className="px-4 py-2 text-gray-600">Cancel</button>
              <button onClick={saveProfile} disabled={!newProfileName.trim()} className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50">Save</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Step 2: Summary Component
interface Step2Props {
  state: ReturnType<typeof getDefaultState>;
  selectedDataset: Dataset | undefined;
  previewData: PreviewResponse | null;
  previewLoading: boolean;
  previewError: string | null;
  calculateCombinations: () => number;
}

const Step2Summary: React.FC<Step2Props> = ({
  state,
  selectedDataset,
  previewData,
  previewLoading,
  previewError,
  calculateCombinations,
}) => {
  const hasWarnings = previewData?.targets.some(t => t.warnings.length > 0);

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Job Summary</h3>

      {/* Dataset */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center space-x-2">
          <Database size={16} />
          <span>Dataset</span>
        </h4>
        {selectedDataset && (
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Name:</span> <span className="font-medium">{selectedDataset.name}</span></div>
            <div><span className="text-gray-500">Ticker:</span> <span className="font-medium">{selectedDataset.ticker}</span></div>
            <div><span className="text-gray-500">Rows:</span> <span className="font-medium">{selectedDataset.rows_count.toLocaleString()}</span></div>
            <div><span className="text-gray-500">Split:</span> <span className="font-medium">{state.trainTestSplit}% / {100 - state.trainTestSplit}%</span></div>
          </div>
        )}
      </div>

      {/* Models */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center space-x-2">
          <Cpu size={16} />
          <span>Models ({state.selectedModels.length})</span>
        </h4>
        <div className="flex flex-wrap gap-2">
          {state.selectedModels.map((modelId) => {
            const model = MODEL_TYPES.find(m => m.id === modelId);
            return (
              <span key={modelId} className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-sm">
                {model?.name || modelId}
              </span>
            );
          })}
        </div>
      </div>

      {/* Prediction Targets Preview */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center space-x-2">
          <Target size={16} />
          <span>Prediction Targets</span>
          {hasWarnings && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded-full text-xs flex items-center space-x-1">
              <AlertTriangle size={12} />
              <span>Warnings</span>
            </span>
          )}
        </h4>

        {previewLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-gray-400" size={24} />
            <span className="ml-2 text-gray-500">Analyzing targets...</span>
          </div>
        ) : previewError ? (
          <div className="text-red-500 text-sm">{previewError}</div>
        ) : previewData ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div><span className="text-gray-500">Total Rows:</span> <span className="font-medium">{previewData.dataset_rows.toLocaleString()}</span></div>
              <div><span className="text-gray-500">Train:</span> <span className="font-medium">{previewData.train_rows.toLocaleString()}</span></div>
              <div><span className="text-gray-500">Test:</span> <span className="font-medium">{previewData.test_rows.toLocaleString()}</span></div>
            </div>

            {previewData.targets.map((target) => (
              <div key={target.name} className={`p-4 rounded-lg border ${target.warnings.length > 0 ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' : 'border-gray-200 dark:border-gray-600'}`}>
                <div className="font-medium text-sm mb-2">{target.label}</div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Train:</span>
                    <span className={`ml-2 font-medium ${target.train_positive === 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {target.train_positive} positive ({target.train_positive_pct}%)
                    </span>
                    <span className="text-gray-400 ml-1">/ {target.train_negative} negative</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Test:</span>
                    <span className={`ml-2 font-medium ${target.test_positive === 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {target.test_positive} positive ({target.test_positive_pct}%)
                    </span>
                    <span className="text-gray-400 ml-1">/ {target.test_negative} negative</span>
                  </div>
                </div>
                {target.warnings.length > 0 && (
                  <div className="mt-3 space-y-1">
                    {target.warnings.map((warning, idx) => (
                      <div key={idx} className="flex items-start space-x-2 text-sm text-yellow-700 dark:text-yellow-300">
                        <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
                        <span>{warning}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-500 text-sm">No preview data</div>
        )}
      </div>

      {/* Genetic Algorithm */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center space-x-2">
          <Activity size={16} />
          <span>Optimization Settings</span>
        </h4>
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div><span className="text-gray-500">Population:</span> <span className="font-medium">{state.geneticConfig.populationSize}</span></div>
          <div><span className="text-gray-500">Generations:</span> <span className="font-medium">{state.geneticConfig.generations}</span></div>
          <div><span className="text-gray-500">Epochs:</span> <span className="font-medium">{state.geneticConfig.trainingEpochs}</span></div>
          <div><span className="text-gray-500">Metric:</span> <span className="font-medium">{state.metricsConfig.optimizeMetric}</span></div>
        </div>
      </div>

      {/* Totals */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Parameter Combinations:</span>
            <span className="font-bold text-blue-600">{calculateCombinations().toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Total Individuals:</span>
            <span className="font-bold text-blue-600">
              {(state.geneticConfig.populationSize * state.geneticConfig.generations).toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Final Warning */}
      {hasWarnings && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-500 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="text-yellow-600 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <div className="font-medium text-yellow-800 dark:text-yellow-200">Data Imbalance Warning</div>
              <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                Some prediction targets have no positive samples in the test set.
                This means F1/precision/recall metrics will be 0 regardless of model quality.
                Consider using less strict target criteria (lower profit %, higher DD tolerance, or longer time window).
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default JobWizard;
