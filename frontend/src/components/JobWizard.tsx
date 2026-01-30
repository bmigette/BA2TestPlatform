import React, { useState, useEffect, useCallback } from 'react';
import { X, Database, Cpu, Target, Trash2, Split, Save, FolderOpen, Play, AlertTriangle, ChevronRight, ChevronLeft, Loader2, Activity, Zap, Info } from 'lucide-react';
import type { TargetConfig } from '../types/targets';

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
  classificationMetric?: string;
  regressionMetric?: string;
  lossFunction?: string;
}

// Target set from backend
interface TargetSet {
  id: number;
  name: string;
  description?: string;
  targets: TargetConfig[];
  created_at: string;
  updated_at: string;
}

interface JobProfile {
  id: number;
  name: string;
  createdAt: string;
  updatedAt?: string;
  jobType?: 'classification' | 'regression';
  selectedModels: string[];
  parameterRanges: ParameterRanges;
  predictionTargets: Record<string, unknown>[];  // Can be old or new TargetConfig format
  trainTestSplit: number;
  geneticConfig?: GeneticConfig;
  metricsConfig?: MetricsConfig;
  predictionHorizon?: number;
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

const CLASSIFICATION_METRICS = [
  { id: 'f1_score', name: 'F1 Score', description: 'Harmonic mean of precision and recall' },
  { id: 'accuracy', name: 'Accuracy', description: 'Overall correctness' },
  { id: 'balanced_accuracy', name: 'Balanced Accuracy', description: 'Average of recall per class' },
  { id: 'precision', name: 'Precision', description: 'Minimize false positives' },
  { id: 'recall', name: 'Recall', description: 'Minimize false negatives' },
  { id: 'mcc', name: 'MCC', description: 'Matthews Correlation Coefficient' },
];

const REGRESSION_METRICS = [
  { id: 'mse', name: 'MSE', description: 'Mean Squared Error' },
  { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error' },
  { id: 'mae', name: 'MAE', description: 'Mean Absolute Error' },
  { id: 'r2', name: 'R²', description: 'Coefficient of Determination' },
  { id: 'mape', name: 'MAPE', description: 'Mean Absolute Percentage Error' },
];

const LOSS_FUNCTIONS = [
  { id: 'focal_loss', name: 'Focal Loss', description: 'Best for imbalanced classification - reduces weight on easy examples (Recommended)' },
  { id: 'weighted_cross_entropy', name: 'Weighted BCE', description: 'Binary cross-entropy with class weights based on positive/negative ratio' },
  { id: 'cross_entropy', name: 'Cross Entropy', description: 'Standard cross-entropy loss - NOT recommended for imbalanced data' },
  { id: 'mse', name: 'MSE', description: 'Mean squared error - for regression tasks only' },
];

interface TrainingDateRange {
  startDate: string | null;
  endDate: string | null;
}

const getDefaultState = () => ({
  jobType: 'classification' as 'classification' | 'regression',
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
    classificationMetric: 'f1_score',
    regressionMetric: 'rmse',
    lossFunction: 'focal_loss',
  } as MetricsConfig,
  predictionTargets: [] as Record<string, unknown>[],
  selectedTargetSetIds: [] as number[],
  predictionHorizon: 3,
  trainTestSplit: 80,
  trainingDateRange: {
    startDate: null,
    endDate: null,
  } as TrainingDateRange,
  useSubsetDateRange: false,
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
  const [showLoadProfileDialog, setShowLoadProfileDialog] = useState(false);
  const [showSaveProfileDialog, setShowSaveProfileDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewData, setPreviewData] = useState<PreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [targetSets, setTargetSets] = useState<TargetSet[]>([]);
  const [targetSetsLoading, setTargetSetsLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState<Array<{id: string, name: string, description: string}>>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  // Fetch target sets when modal opens
  const fetchTargetSets = useCallback(async () => {
    setTargetSetsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/target-sets');
      if (response.ok) {
        const data = await response.json();
        setTargetSets(data.target_sets || []);
      }
    } catch (error) {
      console.error('Failed to fetch target sets:', error);
    } finally {
      setTargetSetsLoading(false);
    }
  }, []);

  // Fetch models based on job type
  const fetchModels = useCallback(async (jobType: 'classification' | 'regression') => {
    setModelsLoading(true);
    try {
      const endpoint = jobType === 'classification'
        ? 'http://localhost:8000/api/ml/classification-models'
        : 'http://localhost:8000/api/ml/models';
      const response = await fetch(endpoint);
      if (response.ok) {
        const data = await response.json();
        // Transform to array format
        const models = Object.entries(data.models || {}).map(([id, info]: [string, any]) => ({
          id,
          name: info.name || id.toUpperCase(),
          description: info.description || '',
        }));
        setAvailableModels(models);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  // Reset when opened
  useEffect(() => {
    if (isOpen) {
      setState(getDefaultState());
      setCurrentStep(1);
      setPreviewData(null);
      setPreviewError(null);
      fetchTargetSets();
      fetchModels('classification'); // Default job type
    }
  }, [isOpen, fetchTargetSets, fetchModels]);

  // Fetch models when job type changes
  useEffect(() => {
    if (isOpen) {
      fetchModels(state.jobType);
    }
  }, [isOpen, state.jobType, fetchModels]);

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
      state.selectedTargetSetIds.length > 0
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
      selectedModels: prev.selectedModels.length === availableModels.length
        ? []
        : availableModels.map(m => m.id)
    }));
  };

  const toggleTargetSet = (targetSetId: number) => {
    setState(prev => {
      const isSelected = prev.selectedTargetSetIds.includes(targetSetId);
      const newSelectedIds = isSelected
        ? prev.selectedTargetSetIds.filter(id => id !== targetSetId)
        : [...prev.selectedTargetSetIds, targetSetId];

      // Aggregate all targets from selected sets
      const allTargets = targetSets
        .filter(ts => newSelectedIds.includes(ts.id))
        .flatMap(ts => ts.targets as unknown as Record<string, unknown>[]);

      return {
        ...prev,
        selectedTargetSetIds: newSelectedIds,
        predictionTargets: allTargets,
      };
    });
  };

  const removeTargetSet = (targetSetId: number) => {
    setState(prev => {
      const newSelectedIds = prev.selectedTargetSetIds.filter(id => id !== targetSetId);
      const allTargets = targetSets
        .filter(ts => newSelectedIds.includes(ts.id))
        .flatMap(ts => ts.targets as unknown as Record<string, unknown>[]);

      return {
        ...prev,
        selectedTargetSetIds: newSelectedIds,
        predictionTargets: allTargets,
      };
    });
  };

  const loadProfile = (profile: JobProfile) => {
    const jobType = profile.jobType || 'classification';  // Default for old profiles
    setState(prev => ({
      ...prev,
      jobType,
      selectedModels: profile.selectedModels || [],
      parameterRanges: profile.parameterRanges || prev.parameterRanges,
      predictionTargets: profile.predictionTargets || [],
      trainTestSplit: profile.trainTestSplit || 80,
      geneticConfig: profile.geneticConfig || prev.geneticConfig,
      metricsConfig: profile.metricsConfig || prev.metricsConfig,
      predictionHorizon: profile.predictionHorizon || 3,
    }));
    // Fetch models for the profile's job type
    fetchModels(jobType);
    setShowLoadProfileDialog(false);
  };

  const saveProfile = async () => {
    if (!newProfileName.trim()) return;
    await onSaveProfile(newProfileName.trim(), {
      jobType: state.jobType,
      selectedModels: state.selectedModels,
      parameterRanges: state.parameterRanges,
      predictionTargets: state.predictionTargets,
      trainTestSplit: state.trainTestSplit,
      geneticConfig: state.geneticConfig,
      metricsConfig: state.metricsConfig,
      predictionHorizon: state.predictionHorizon,
    });
    setNewProfileName('');
    setShowSaveProfileDialog(false);
  };

  const fetchPreview = async () => {
    if (!state.selectedDatasetId || state.predictionTargets.length === 0) return;

    setPreviewLoading(true);
    setPreviewError(null);

    try {
      // Use the new calculate-targets endpoint with target configs
      const response = await fetch(`http://localhost:8000/api/datasets/${state.selectedDatasetId}/calculate-targets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          targets: state.predictionTargets,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch preview');
      }

      const data = await response.json();

      // Calculate train/test split for each target based on the data array
      const totalRows = data.total_rows || data.targets?.[0]?.stats?.totalRows || 0;
      const trainRows = Math.floor(totalRows * state.trainTestSplit / 100);
      const testRows = totalRows - trainRows;

      // Convert to preview format with proper train/test splits
      const previewResponse: PreviewResponse = {
        dataset_id: state.selectedDatasetId,
        dataset_rows: totalRows,
        train_rows: trainRows,
        test_rows: testRows,
        targets: (data.targets || []).map((t: any) => {
          // Split the data array by train/test boundary
          const targetData = t.data || [];
          const trainData = targetData.slice(0, trainRows);
          const testData = targetData.slice(trainRows);

          // Count positives in train portion (value === 1 for binary classification)
          const trainPositive = trainData.filter((d: any) => d.value === 1).length;
          const trainNegative = trainData.length - trainPositive;
          const trainPositivePct = trainData.length > 0 ? parseFloat((trainPositive / trainData.length * 100).toFixed(2)) : 0;

          // Count positives in test portion
          const testPositive = testData.filter((d: any) => d.value === 1).length;
          const testNegative = testData.length - testPositive;
          const testPositivePct = testData.length > 0 ? parseFloat((testPositive / testData.length * 100).toFixed(2)) : 0;

          // Generate warnings
          const warnings: string[] = [];
          if (trainPositive === 0) warnings.push('No positive samples in training data');
          if (testPositive === 0) warnings.push('No positive samples in test data');
          if (trainPositivePct < 1) warnings.push('Very low positive rate in training data');

          return {
            name: t.columnName,
            label: getTargetLabel(t.config),
            train_positive: trainPositive,
            train_negative: trainNegative,
            train_positive_pct: trainPositivePct,
            test_positive: testPositive,
            test_negative: testNegative,
            test_positive_pct: testPositivePct,
            warnings,
          };
        }),
      };
      setPreviewData(previewResponse);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const getTargetLabel = (config: Record<string, unknown> | undefined): string => {
    if (!config || !config.type) {
      return 'Unknown target';
    }
    switch (config.type) {
      case 'price_based':
        return `Price ${config.direction === 'up' ? '▲' : '▼'} ${config.profitPct || 0}% (${config.timeBars || 0} bars)`;
      case 'directional':
        return `Direction ${config.direction === 'up' ? '▲' : '▼'} (${config.horizon || 0} bars)`;
      case 'triple_barrier':
        return `Triple Barrier TP:${config.profitPct || 0}% SL:${config.stopPct || 0}% (${config.maxBars || 0} bars)`;
      case 'trend_reversal':
        return `${String(config.indicator || 'Unknown').toUpperCase()} ${config.direction || ''} reversal`;
      case 'volatility':
        return `Volatility (${config.method || 'unknown'}, ${config.horizon || 0} bars)`;
      default:
        return `Target: ${String(config.type)}`;
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
      const response = await fetch('http://localhost:8000/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobType: state.jobType,
          datasetId: state.selectedDatasetId,
          selectedModels: state.selectedModels,
          parameterRanges: state.parameterRanges,
          predictionTargets: state.predictionTargets,
          predictionHorizon: state.predictionHorizon,
          trainTestSplit: state.trainTestSplit,
          geneticConfig: state.geneticConfig,
          metricsConfig: state.metricsConfig,
          trainingDateRange: state.useSubsetDateRange ? state.trainingDateRange : null,
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
                newProfileName={newProfileName}
                setNewProfileName={setNewProfileName}
                handleModelToggle={handleModelToggle}
                handleAllModelsToggle={handleAllModelsToggle}
                targetSets={targetSets}
                targetSetsLoading={targetSetsLoading}
                toggleTargetSet={toggleTargetSet}
                removeTargetSet={removeTargetSet}
                loadProfile={loadProfile}
                saveProfile={saveProfile}
                onDeleteProfile={onDeleteProfile}
                calculateCombinations={calculateCombinations}
                availableModels={availableModels}
                modelsLoading={modelsLoading}
              />
            ) : (
              <Step2Summary
                state={state}
                selectedDataset={selectedDataset}
                previewData={previewData}
                previewLoading={previewLoading}
                previewError={previewError}
                calculateCombinations={calculateCombinations}
                availableModels={availableModels}
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
  newProfileName: string;
  setNewProfileName: (v: string) => void;
  handleModelToggle: (id: string) => void;
  handleAllModelsToggle: () => void;
  targetSets: TargetSet[];
  targetSetsLoading: boolean;
  toggleTargetSet: (id: number) => void;
  removeTargetSet: (id: number) => void;
  loadProfile: (profile: JobProfile) => void;
  saveProfile: () => void;
  onDeleteProfile: (id: number) => Promise<void>;
  calculateCombinations: () => number;
  availableModels: Array<{id: string, name: string, description: string}>;
  modelsLoading: boolean;
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
  newProfileName,
  setNewProfileName,
  handleModelToggle,
  handleAllModelsToggle,
  targetSets,
  targetSetsLoading,
  toggleTargetSet,
  removeTargetSet,
  loadProfile,
  saveProfile,
  onDeleteProfile,
  calculateCombinations,
  availableModels,
  modelsLoading,
}) => {
  const allModelsSelected = state.selectedModels.length === availableModels.length && availableModels.length > 0;
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

      {/* Job Type Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Job Type
        </label>
        <div className="flex space-x-4">
          <label
            className={`flex-1 flex items-center justify-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-colors ${
              state.jobType === 'classification'
                ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
            }`}
          >
            <input
              type="radio"
              name="jobType"
              value="classification"
              checked={state.jobType === 'classification'}
              onChange={() => setState(prev => ({ ...prev, jobType: 'classification', selectedModels: [] }))}
              className="sr-only"
              tabIndex={-1}
            />
            <Target size={20} className={state.jobType === 'classification' ? 'text-green-600' : 'text-gray-400'} />
            <div>
              <div className="font-medium">Classification</div>
              <div className="text-xs text-gray-500">Binary prediction (up/down, signal/no-signal)</div>
            </div>
          </label>
          <label
            className={`flex-1 flex items-center justify-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-colors ${
              state.jobType === 'regression'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
            }`}
          >
            <input
              type="radio"
              name="jobType"
              value="regression"
              checked={state.jobType === 'regression'}
              onChange={() => setState(prev => ({ ...prev, jobType: 'regression', selectedModels: [] }))}
              className="sr-only"
              tabIndex={-1}
            />
            <Activity size={20} className={state.jobType === 'regression' ? 'text-blue-600' : 'text-gray-400'} />
            <div>
              <div className="font-medium">Regression</div>
              <div className="text-xs text-gray-500">Continuous value prediction (price, volatility)</div>
            </div>
          </label>
        </div>
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
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 space-y-4">
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Ticker:</span> <span className="font-medium">{selectedDataset.ticker}</span></div>
            <div><span className="text-gray-500">Timeframe:</span> <span className="font-medium">{selectedDataset.timeframe}</span></div>
            <div><span className="text-gray-500">Rows:</span> <span className="font-medium">{selectedDataset.rows_count.toLocaleString()}</span></div>
            <div><span className="text-gray-500">Range:</span> <span className="font-medium">{formatDate(selectedDataset.start_date)} - {formatDate(selectedDataset.end_date)}</span></div>
          </div>

          {/* Training Date Range Subset */}
          <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
            <label className="flex items-center space-x-2 cursor-pointer mb-3">
              <input
                type="checkbox"
                checked={state.useSubsetDateRange}
                onChange={(e) => {
                  const useSubset = e.target.checked;
                  setState(prev => ({
                    ...prev,
                    useSubsetDateRange: useSubset,
                    trainingDateRange: useSubset ? {
                      startDate: selectedDataset.start_date.split('T')[0],
                      endDate: selectedDataset.end_date.split('T')[0]
                    } : { startDate: null, endDate: null }
                  }));
                }}
                className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Use subset of dataset time range
              </span>
            </label>

            {state.useSubsetDateRange && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Training Start Date</label>
                  <input
                    type="date"
                    value={state.trainingDateRange.startDate || ''}
                    min={selectedDataset.start_date.split('T')[0]}
                    max={state.trainingDateRange.endDate || selectedDataset.end_date.split('T')[0]}
                    onChange={(e) => setState(prev => ({
                      ...prev,
                      trainingDateRange: { ...prev.trainingDateRange, startDate: e.target.value }
                    }))}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Training End Date</label>
                  <input
                    type="date"
                    value={state.trainingDateRange.endDate || ''}
                    min={state.trainingDateRange.startDate || selectedDataset.start_date.split('T')[0]}
                    max={selectedDataset.end_date.split('T')[0]}
                    onChange={(e) => setState(prev => ({
                      ...prev,
                      trainingDateRange: { ...prev.trainingDateRange, endDate: e.target.value }
                    }))}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Types */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Select Model Types
            <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${
              state.jobType === 'classification'
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
            }`}>
              {state.jobType === 'classification' ? 'Classification' : 'Regression'}
            </span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={allModelsSelected}
              onChange={handleAllModelsToggle}
              disabled={modelsLoading}
              className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">All Models</span>
          </label>
        </div>
        {modelsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-gray-400" size={24} />
            <span className="ml-2 text-gray-500">Loading models...</span>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            {availableModels.map((model) => (
              <label
                key={model.id}
                className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors cursor-pointer ${
                  state.selectedModels.includes(model.id)
                    ? state.jobType === 'classification'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={state.selectedModels.includes(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                  className={`mt-1 w-4 h-4 border-gray-300 rounded ${
                    state.jobType === 'classification' ? 'text-green-600' : 'text-blue-600'
                  }`}
                />
                <div>
                  <div className="font-medium text-sm">{model.name}</div>
                  <span className="text-xs text-gray-500">{model.description}</span>
                </div>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Prediction Targets */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Target size={16} className="text-gray-400" />
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Prediction Targets</label>
          </div>
        </div>

        {/* Info message about creating targets */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-4">
          <div className="flex items-start space-x-2">
            <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-blue-700 dark:text-blue-300">
              <p>To create new prediction target profiles, go to the <strong>Dataset Details</strong> page and use the Prediction Targets panel.</p>
            </div>
          </div>
        </div>

        {/* Target Set Selection */}
        {targetSetsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-gray-400" size={24} />
            <span className="ml-2 text-gray-500">Loading target profiles...</span>
          </div>
        ) : targetSets.length === 0 ? (
          <div className="text-center py-8 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <Target size={32} className="mx-auto text-gray-400 mb-2" />
            <p className="text-gray-500 dark:text-gray-400">No saved target profiles found</p>
            <p className="text-sm text-gray-400 mt-1">Create target profiles in the Dataset Details page</p>
          </div>
        ) : (
          <div className="space-y-2">
            {targetSets.map((ts) => (
              <label
                key={ts.id}
                className={`flex items-start space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  state.selectedTargetSetIds.includes(ts.id)
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={state.selectedTargetSetIds.includes(ts.id)}
                  onChange={() => toggleTargetSet(ts.id)}
                  className="mt-1 w-4 h-4 text-green-600 border-gray-300 rounded"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm">{ts.name}</div>
                  {ts.description && (
                    <p className="text-xs text-gray-500 mt-0.5">{ts.description}</p>
                  )}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {ts.targets.map((target, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-600 rounded"
                      >
                        {target.type.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}

        {/* Selected targets summary */}
        {state.selectedTargetSetIds.length > 0 && (
          <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-sm font-medium text-green-700 dark:text-green-300 mb-2">
              Selected: {state.predictionTargets.length} target(s) from {state.selectedTargetSetIds.length} profile(s)
            </div>
            <div className="flex flex-wrap gap-2">
              {state.selectedTargetSetIds.map((id) => {
                const ts = targetSets.find(t => t.id === id);
                return ts ? (
                  <div key={id} className="flex items-center space-x-1 px-2 py-1 bg-white dark:bg-gray-700 rounded text-sm border border-green-300">
                    <span>{ts.name}</span>
                    <button onClick={() => removeTargetSet(id)} className="text-red-500 hover:text-red-700">
                      <X size={14} />
                    </button>
                  </div>
                ) : null;
              })}
            </div>
          </div>
        )}
      </div>

      {/* Prediction Horizon */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <ChevronRight size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Prediction Horizon</label>
        </div>
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 space-y-4">
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Bars ahead</label>
              <input
                type="number"
                min={1}
                max={30}
                value={state.predictionHorizon}
                onChange={(e) => {
                  const newHorizon = Math.max(1, Math.min(30, Number(e.target.value)));
                  setState(prev => ({ ...prev, predictionHorizon: newHorizon }));
                }}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
            <div className="text-sm flex-1">
              <p className="text-gray-800 dark:text-white">
                Models will predict target values <strong className="text-blue-600 dark:text-blue-300">{state.predictionHorizon}</strong> bar(s) ahead.
              </p>
              <p className="text-xs mt-1 text-gray-600 dark:text-gray-300">Higher values give more lead time but may reduce accuracy.</p>
            </div>
          </div>

          {/* Model-specific behavior explanation */}
          <div className="border-t border-gray-200 dark:border-gray-600 pt-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2 font-medium">
              How {state.jobType} models use prediction horizon:
            </div>
            {state.jobType === 'classification' ? (
              <div className="bg-white dark:bg-gray-800 rounded p-2 border border-gray-200 dark:border-gray-600 text-xs">
                <div className="font-medium text-green-600 dark:text-green-400 mb-1">Classification Models (tsai)</div>
                <p className="text-gray-600 dark:text-gray-300">
                  <strong>Input:</strong> Bars T-23 to T (24-bar lookback window)<br/>
                  <strong>Target:</strong> Class label at bar T+{state.predictionHorizon} (shifted by prediction horizon)<br/>
                  <strong>Output:</strong> Probability that target condition is true at bar T+{state.predictionHorizon}
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="bg-white dark:bg-gray-800 rounded p-2 border border-gray-200 dark:border-gray-600">
                  <div className="font-medium text-purple-600 dark:text-purple-400 mb-1">LSTM / GRU</div>
                  <p className="text-gray-600 dark:text-gray-300">
                    <strong>Input:</strong> Bars T-23 to T<br/>
                    <strong>Output:</strong> Single value at T+{state.predictionHorizon}
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded p-2 border border-gray-200 dark:border-gray-600">
                  <div className="font-medium text-blue-600 dark:text-blue-400 mb-1">N-BEATS / TCN / Transformer / TFT</div>
                  <p className="text-gray-600 dark:text-gray-300">
                    <strong>Input:</strong> Bars T-23 to T<br/>
                    <strong>Output:</strong> {state.predictionHorizon} values (T+1 to T+{state.predictionHorizon})
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
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

      {/* Optimization Metrics */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <Zap size={16} className="text-gray-400" />
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Optimization Metric
          </label>
        </div>

        {state.jobType === 'classification' ? (
          <>
            {/* Classification Metric */}
            <div className="flex flex-wrap gap-2">
              {CLASSIFICATION_METRICS.map((metric) => (
                <label
                  key={metric.id}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer text-sm ${
                    state.metricsConfig.classificationMetric === metric.id
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700'
                      : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                  }`}
                  title={metric.description}
                >
                  <input
                    type="radio"
                    name="optimizeMetric"
                    checked={state.metricsConfig.classificationMetric === metric.id}
                    onChange={() => setState(prev => ({
                      ...prev,
                      metricsConfig: {
                        ...prev.metricsConfig,
                        classificationMetric: metric.id,
                        optimizeMetric: metric.id
                      }
                    }))}
                    className="sr-only"
                    tabIndex={-1}
                  />
                  <span>{metric.name}</span>
                </label>
              ))}
            </div>

            {/* Loss Function - Only for Classification */}
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-2">Training Loss Function</label>
              <div className="flex flex-wrap gap-2">
                {LOSS_FUNCTIONS.filter(l => l.id !== 'mse').map((loss) => (
                  <label
                    key={loss.id}
                    className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer text-sm ${
                      state.metricsConfig.lossFunction === loss.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20 text-purple-700'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                    }`}
                    title={loss.description}
                  >
                    <input
                      type="radio"
                      name="lossFunction"
                      checked={state.metricsConfig.lossFunction === loss.id}
                      onChange={() => setState(prev => ({
                        ...prev,
                        metricsConfig: {
                          ...prev.metricsConfig,
                          lossFunction: loss.id
                        }
                      }))}
                      className="sr-only"
                      tabIndex={-1}
                    />
                    <span>{loss.name}</span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Focal Loss is recommended for imbalanced classification (when positive samples are rare)
              </p>
            </div>
          </>
        ) : (
          /* Regression Metric */
          <div className="flex flex-wrap gap-2">
            {REGRESSION_METRICS.map((metric) => (
              <label
                key={metric.id}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer text-sm ${
                  state.metricsConfig.regressionMetric === metric.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700'
                    : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                }`}
                title={metric.description}
              >
                <input
                  type="radio"
                  name="optimizeMetric"
                  checked={state.metricsConfig.regressionMetric === metric.id}
                  onChange={() => setState(prev => ({
                    ...prev,
                    metricsConfig: {
                      ...prev.metricsConfig,
                      regressionMetric: metric.id,
                      optimizeMetric: metric.id
                    }
                  }))}
                  className="sr-only"
                  tabIndex={-1}
                />
                <span>{metric.name}</span>
              </label>
            ))}
          </div>
        )}
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
                      <div className="text-xs text-gray-500">
                        {profile.selectedModels?.length || 0} models
                        {profile.createdAt && (
                          <span className="ml-2">
                            · {new Date(profile.createdAt).toLocaleDateString()}
                          </span>
                        )}
                      </div>
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
  availableModels: Array<{id: string, name: string, description: string}>;
}

const Step2Summary: React.FC<Step2Props> = ({
  state,
  selectedDataset,
  previewData,
  previewLoading,
  previewError,
  calculateCombinations,
  availableModels,
}) => {
  const hasWarnings = previewData?.targets.some(t => t.warnings.length > 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Job Summary</h3>
        {/* Job Type Badge */}
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
          state.jobType === 'classification'
            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
            : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
        }`}>
          {state.jobType === 'classification' ? 'Classification' : 'Regression'}
        </span>
      </div>

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
            const model = availableModels.find(m => m.id === modelId);
            return (
              <span key={modelId} className={`px-3 py-1 rounded-full text-sm ${
                state.jobType === 'classification'
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                  : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
              }`}>
                {model?.name || modelId.toUpperCase()}
              </span>
            );
          })}
        </div>
      </div>

      {/* Prediction Targets Preview */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center space-x-2">
          <Target size={16} />
          <span>Prediction Targets ({previewData?.targets?.length || state.predictionTargets.length})</span>
          {hasWarnings && (
            <span className="px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 rounded-full text-xs flex items-center space-x-1">
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

            {previewData.targets.map((target, idx) => (
              <div key={`${target.name}-${idx}`} className={`p-4 rounded-lg border ${target.warnings.length > 0 ? 'border-amber-400 dark:border-amber-600 bg-amber-50 dark:bg-amber-900/10' : 'border-gray-200 dark:border-gray-600'}`}>
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
                    {target.warnings.map((warning, warnIdx) => (
                      <div key={warnIdx} className="flex items-start space-x-2 text-sm text-amber-600 dark:text-amber-400">
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
        <div className="grid grid-cols-3 gap-4 text-sm mb-2">
          <div><span className="text-gray-500">Population:</span> <span className="font-medium">{state.geneticConfig.populationSize}</span></div>
          <div><span className="text-gray-500">Generations:</span> <span className="font-medium">{state.geneticConfig.generations}</span></div>
          <div><span className="text-gray-500">Epochs:</span> <span className="font-medium">{state.geneticConfig.trainingEpochs}</span></div>
        </div>
        <div className={`grid ${state.jobType === 'classification' ? 'grid-cols-4' : 'grid-cols-3'} gap-4 text-sm`}>
          <div>
            <span className="text-gray-500">Optimize:</span>{' '}
            <span className="font-medium">
              {state.jobType === 'classification'
                ? (state.metricsConfig.classificationMetric || 'f1_score')
                : (state.metricsConfig.regressionMetric || 'rmse')}
            </span>
          </div>
          {state.jobType === 'classification' && (
            <div><span className="text-gray-500">Loss:</span> <span className="font-medium">{state.metricsConfig.lossFunction || 'focal_loss'}</span></div>
          )}
          <div><span className="text-gray-500">Horizon:</span> <span className="font-medium">{state.predictionHorizon} bars</span></div>
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
        <div className="bg-amber-50 dark:bg-amber-900/10 border border-amber-400 dark:border-amber-600 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="text-amber-500 dark:text-amber-400 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <div className="font-medium text-amber-700 dark:text-amber-300">Data Imbalance Warning</div>
              <p className="text-sm text-amber-600 dark:text-amber-400 mt-1">
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
