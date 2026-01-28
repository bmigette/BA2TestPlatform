import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, X, Database, Calendar, BarChart2, Cpu, Sliders, Target, Trash2, Split, Save, FolderOpen, Play, Clock, CheckCircle, AlertCircle, Loader2, Pause, SkipForward, XCircle, ArrowLeft, Activity, Timer, Zap, FileText, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

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

const MODEL_TYPES = [
  { id: 'lstm', name: 'LSTM', description: 'Long Short-Term Memory' },
  { id: 'nbeats', name: 'N-BEATS', description: 'Neural Basis Expansion Analysis' },
  { id: 'transformer', name: 'Transformer', description: 'Temporal Fusion Transformer' },
  { id: 'tcn', name: 'TCN', description: 'Temporal Convolutional Network' },
  { id: 'rcnn', name: 'RCNN', description: 'Recurrent Convolutional Neural Network' },
];

const ACTIVATION_FUNCTIONS = [
  { id: 'relu', name: 'ReLU', description: 'Rectified Linear Unit' },
  { id: 'tanh', name: 'tanh', description: 'Hyperbolic Tangent' },
  { id: 'sigmoid', name: 'Sigmoid', description: 'Logistic Function' },
  { id: 'leaky_relu', name: 'LeakyReLU', description: 'Leaky Rectified Linear Unit' },
];

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
}

interface MetricsConfig {
  optimizeMetric: string;
}

const AVAILABLE_METRICS = [
  { id: 'f1_score', name: 'F1 Score', description: 'Harmonic mean of precision and recall' },
  { id: 'accuracy', name: 'Accuracy', description: 'Overall correctness (caution: misleading for imbalanced data)' },
  { id: 'balanced_accuracy', name: 'Balanced Accuracy', description: 'Average of recall per class' },
  { id: 'precision', name: 'Precision', description: 'Minimize false positives' },
  { id: 'recall', name: 'Recall', description: 'Minimize false negatives' },
  { id: 'auc_roc', name: 'AUC-ROC', description: 'Area under ROC curve' },
  { id: 'mcc', name: 'MCC', description: 'Matthews Correlation Coefficient' },
];

interface PredictionTarget {
  id: string;
  profitPercent: number;
  maxDrawdownPercent: number;
  timePeriodDays: number;
}

interface JobProfile {
  id: string;
  name: string;
  createdAt: string;
  selectedModels: string[];
  parameterRanges: ParameterRanges;
  predictionTargets: Omit<PredictionTarget, 'id'>[];
  trainTestSplit: number;
  geneticConfig?: GeneticConfig;
  metricsConfig?: MetricsConfig;
}

const PROFILES_STORAGE_KEY = 'ba2ml_job_profiles';

interface Job {
  id: string;
  datasetId: number;
  selectedModels: string[];
  status: 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  currentGeneration?: number;
  totalGenerations?: number;
  currentLoss?: number;
  currentAccuracy?: number;
  bestFitness?: number;
  gpuUtilization?: number;
  estimatedTimeRemaining?: string;
}

interface TrainingMetric {
  generation: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  fitness: number;
  timestamp: string;
}

interface JobProgress {
  job: Job;
  metrics: TrainingMetric[];
  logs: string[];
}

interface Individual {
  generation: number;
  individual: number;
  model_type: string;
  params: Record<string, number | string>;
  fitness: number;
  metrics: Record<string, number>;
}

interface IndividualsData {
  job_id: string;
  summary: {
    total_individuals: number;
    generations: number[];
    model_types: string[];
    best_fitness: number;
    avg_fitness: number;
  };
  best_individual: Individual | null;
  individuals: Individual[];
}

interface GenerationSummary {
  generation: number;
  individual_count: number;
  best_fitness: number;
  avg_fitness: number;
  min_fitness: number;
  model_types: Record<string, number>;
  best_individual: Individual | null;
}

interface GenerationsData {
  job_id: string;
  total_generations: number;
  generations: GenerationSummary[];
}

const PREDICTION_PRESETS = [
  { label: '10% / 5% DD / 7d', profit: 10, drawdown: 5, days: 7 },
  { label: '20% / 10% DD / 30d', profit: 20, drawdown: 10, days: 30 },
  { label: '5% / 3% DD / 3d', profit: 5, drawdown: 3, days: 3 },
  { label: '15% / 7% DD / 14d', profit: 15, drawdown: 7, days: 14 },
];

const generateTargetFieldNames = (target: PredictionTarget) => {
  const suffix = `${target.profitPercent}pct_${target.maxDrawdownPercent}dd_${target.timePeriodDays}d`;
  return {
    up: `price_up_${suffix}`,
    down: `price_down_${suffix}`,
  };
};

const Training: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);
  const [useMultiDataset, setUseMultiDataset] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showStepConfig, setShowStepConfig] = useState(false);
  const [parameterRanges, setParameterRanges] = useState<ParameterRanges>({
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
  });
  const [geneticConfig, setGeneticConfig] = useState<GeneticConfig>({
    populationSize: 20,
    generations: 50,
    elitismPercent: 10,
    crossoverProb: 0.7,
    mutationProb: 0.2,
    earlyStoppingGenerations: 5,
  });
  const [metricsConfig, setMetricsConfig] = useState<MetricsConfig>({
    optimizeMetric: 'f1_score',
  });
  const [predictionTargets, setPredictionTargets] = useState<PredictionTarget[]>([]);
  const [showCustomTargetForm, setShowCustomTargetForm] = useState(false);
  const [customTarget, setCustomTarget] = useState({
    profitPercent: 15,
    maxDrawdownPercent: 7,
    timePeriodDays: 14,
  });
  const [trainTestSplit, setTrainTestSplit] = useState(80);
  const [profiles, setProfiles] = useState<JobProfile[]>([]);
  const [showLoadProfileDialog, setShowLoadProfileDialog] = useState(false);
  const [showSaveProfileDialog, setShowSaveProfileDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [saveProfileSuccess, setSaveProfileSuccess] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
  const [isLoadingProgress, _setIsLoadingProgress] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showIndividuals, setShowIndividuals] = useState(false);
  const [individualsData, setIndividualsData] = useState<IndividualsData | null>(null);
  const [generationsData, setGenerationsData] = useState<GenerationsData | null>(null);
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const [selectedModelTypeFilter, setSelectedModelTypeFilter] = useState<string>('');

  // Load profiles from localStorage on mount
  useEffect(() => {
    const savedProfiles = localStorage.getItem(PROFILES_STORAGE_KEY);
    if (savedProfiles) {
      try {
        setProfiles(JSON.parse(savedProfiles));
      } catch (e) {
        console.error('Failed to load profiles:', e);
      }
    }
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8002/api/datasets');
      if (!response.ok) {
        throw new Error('Failed to fetch datasets');
      }
      const data = await response.json();
      setDatasets(data.datasets);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
    fetchJobs();
  }, []);

  const fetchJobs = async () => {
    try {
      const response = await fetch('http://localhost:8002/api/jobs');
      if (response.ok) {
        const data = await response.json();
        setJobs(data.jobs);
      }
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    }
  };

  const fetchJobProgress = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/progress`);
      if (response.ok) {
        const data: JobProgress = await response.json();
        setJobProgress(data);
        // Update job in jobs list too
        setJobs(prev => prev.map(j => j.id === jobId ? data.job : j));
      }
    } catch (err) {
      console.error('Failed to fetch job progress:', err);
    }
  }, []);

  const fetchIndividuals = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/individuals`);
      if (response.ok) {
        const data: IndividualsData = await response.json();
        setIndividualsData(data);
      }
    } catch (err) {
      console.error('Failed to fetch individuals:', err);
    }
  }, []);

  const fetchGenerations = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/generations`);
      if (response.ok) {
        const data: GenerationsData = await response.json();
        setGenerationsData(data);
      }
    } catch (err) {
      console.error('Failed to fetch generations:', err);
    }
  }, []);

  const handlePauseJob = async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/pause`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchJobProgress(jobId);
      }
    } catch (err) {
      console.error('Failed to pause job:', err);
    }
  };

  const handleResumeJob = async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/resume`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchJobProgress(jobId);
      }
    } catch (err) {
      console.error('Failed to resume job:', err);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to cancel this job?')) return;
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}/cancel`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchJobProgress(jobId);
      }
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handleDeleteJob = async (jobId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent opening job monitor
    if (!confirm('Are you sure you want to delete this job? This cannot be undone.')) return;
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${jobId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        // Refresh jobs list
        fetchJobs();
        // Close monitor if this job was selected
        if (selectedJobId === jobId) {
          setSelectedJobId(null);
        }
      }
    } catch (err) {
      console.error('Failed to delete job:', err);
    }
  };

  const openJobMonitor = (jobId: string) => {
    navigate(`/training/${jobId}`);
  };

  const closeJobMonitor = () => {
    setSelectedJobId(null);
    setJobProgress(null);
    setShowLogs(false);
  };

  // Auto-refresh job progress when monitoring
  useEffect(() => {
    if (!selectedJobId || !jobProgress) return;

    // Only poll if job is running or paused
    if (!['running', 'paused', 'queued'].includes(jobProgress.job.status)) return;

    const interval = setInterval(() => {
      fetchJobProgress(selectedJobId);
    }, 1000);

    return () => clearInterval(interval);
  }, [selectedJobId, jobProgress?.job.status, fetchJobProgress]);

  // Auto-refresh jobs list when there are active jobs
  useEffect(() => {
    const hasActiveJobs = jobs.some(j => ['running', 'paused', 'queued'].includes(j.status));

    if (!hasActiveJobs) return;

    const interval = setInterval(() => {
      fetchJobs();
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [jobs]);

  const submitJob = async () => {
    if (!selectedDatasetId || selectedModels.length === 0 || !isParameterValid() || predictionTargets.length === 0) {
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch('http://localhost:8002/api/jobs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          datasetId: selectedDatasetId,
          selectedModels,
          parameterRanges,
          predictionTargets: predictionTargets.map(({ profitPercent, maxDrawdownPercent, timePeriodDays }) => ({
            profitPercent,
            maxDrawdownPercent,
            timePeriodDays,
          })),
          trainTestSplit,
          geneticConfig,
          metricsConfig,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create job');
      }

      const newJob = await response.json();
      setJobs(prev => [newJob, ...prev]);
      setIsFormOpen(false);
      // Reset form
      setSelectedDatasetId(null);
      setSelectedModels([]);
      setPredictionTargets([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create job');
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const handleOpenForm = () => {
    setIsFormOpen(true);
    setSelectedDatasetId(null);
    setSelectedDatasetIds([]);
    setUseMultiDataset(false);
    setSelectedModels([]);
    setShowStepConfig(false);
    setParameterRanges({
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
    });
    setPredictionTargets([]);
    setShowCustomTargetForm(false);
    setCustomTarget({ profitPercent: 15, maxDrawdownPercent: 7, timePeriodDays: 14 });
    setTrainTestSplit(80);
    setGeneticConfig({
      populationSize: 20,
      generations: 50,
      elitismPercent: 10,
      crossoverProb: 0.7,
      mutationProb: 0.2,
      earlyStoppingGenerations: 5,
    });
    setMetricsConfig({
      optimizeMetric: 'f1_score',
    });
  };

  const handleCloseForm = () => {
    setIsFormOpen(false);
    setSelectedDatasetId(null);
    setSelectedModels([]);
  };

  const handleParameterChange = (field: keyof ParameterRanges, value: number | string[]) => {
    setParameterRanges(prev => ({ ...prev, [field]: value }));
  };

  const handleActivationToggle = (activationId: string) => {
    setParameterRanges(prev => ({
      ...prev,
      activationFunctions: prev.activationFunctions.includes(activationId)
        ? prev.activationFunctions.filter(id => id !== activationId)
        : [...prev.activationFunctions, activationId]
    }));
  };

  const isParameterValid = () => {
    return (
      parameterRanges.layersMin <= parameterRanges.layersMax &&
      parameterRanges.layerSizeMin <= parameterRanges.layerSizeMax &&
      parameterRanges.learningRateMin <= parameterRanges.learningRateMax &&
      parameterRanges.dropoutMin <= parameterRanges.dropoutMax &&
      parameterRanges.activationFunctions.length > 0
    );
  };

  const calculateCombinations = () => {
    const layersCount = Math.max(1, Math.floor((parameterRanges.layersMax - parameterRanges.layersMin) / parameterRanges.layersStep) + 1);
    const layerSizeCount = Math.max(1, Math.floor((parameterRanges.layerSizeMax - parameterRanges.layerSizeMin) / parameterRanges.layerSizeStep) + 1);
    const lrCount = Math.max(1, Math.floor((parameterRanges.learningRateMax - parameterRanges.learningRateMin) / parameterRanges.learningRateStep) + 1);
    const dropoutCount = Math.max(1, Math.floor((parameterRanges.dropoutMax - parameterRanges.dropoutMin) / parameterRanges.dropoutStep) + 1);
    const activationCount = parameterRanges.activationFunctions.length;
    const modelCount = selectedModels.length || 1;
    return layersCount * layerSizeCount * lrCount * dropoutCount * activationCount * modelCount;
  };

  const addPresetTarget = (preset: typeof PREDICTION_PRESETS[0]) => {
    const newTarget: PredictionTarget = {
      id: `target_${Date.now()}`,
      profitPercent: preset.profit,
      maxDrawdownPercent: preset.drawdown,
      timePeriodDays: preset.days,
    };
    // Check if this preset already exists
    const exists = predictionTargets.some(
      t => t.profitPercent === preset.profit &&
           t.maxDrawdownPercent === preset.drawdown &&
           t.timePeriodDays === preset.days
    );
    if (!exists) {
      setPredictionTargets(prev => [...prev, newTarget]);
    }
  };

  const removeTarget = (targetId: string) => {
    setPredictionTargets(prev => prev.filter(t => t.id !== targetId));
  };

  const addCustomTarget = () => {
    // Check if this exact target already exists
    const exists = predictionTargets.some(
      t => t.profitPercent === customTarget.profitPercent &&
           t.maxDrawdownPercent === customTarget.maxDrawdownPercent &&
           t.timePeriodDays === customTarget.timePeriodDays
    );
    if (!exists && customTarget.profitPercent > 0 && customTarget.maxDrawdownPercent > 0 && customTarget.timePeriodDays > 0) {
      const newTarget: PredictionTarget = {
        id: `target_${Date.now()}`,
        profitPercent: customTarget.profitPercent,
        maxDrawdownPercent: customTarget.maxDrawdownPercent,
        timePeriodDays: customTarget.timePeriodDays,
      };
      setPredictionTargets(prev => [...prev, newTarget]);
      setShowCustomTargetForm(false);
      setCustomTarget({ profitPercent: 15, maxDrawdownPercent: 7, timePeriodDays: 14 });
    }
  };

  const isCustomTargetValid = () => {
    return customTarget.profitPercent > 0 &&
           customTarget.maxDrawdownPercent > 0 &&
           customTarget.timePeriodDays > 0 &&
           customTarget.maxDrawdownPercent < customTarget.profitPercent;
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleAllModelsToggle = () => {
    if (selectedModels.length === MODEL_TYPES.length) {
      setSelectedModels([]);
    } else {
      setSelectedModels(MODEL_TYPES.map(m => m.id));
    }
  };

  const allModelsSelected = selectedModels.length === MODEL_TYPES.length;

  const saveProfile = () => {
    if (!newProfileName.trim()) return;

    const newProfile: JobProfile = {
      id: `profile_${Date.now()}`,
      name: newProfileName.trim(),
      createdAt: new Date().toISOString(),
      selectedModels,
      parameterRanges,
      predictionTargets: predictionTargets.map(({ profitPercent, maxDrawdownPercent, timePeriodDays }) => ({
        profitPercent,
        maxDrawdownPercent,
        timePeriodDays,
      })),
      trainTestSplit,
      geneticConfig,
      metricsConfig,
    };

    const updatedProfiles = [...profiles, newProfile];
    setProfiles(updatedProfiles);
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(updatedProfiles));

    setNewProfileName('');
    setSaveProfileSuccess(true);
    setTimeout(() => {
      setSaveProfileSuccess(false);
      setShowSaveProfileDialog(false);
    }, 1500);
  };

  const loadProfile = (profile: JobProfile) => {
    setSelectedModels(profile.selectedModels);
    setParameterRanges(profile.parameterRanges);
    setPredictionTargets(
      profile.predictionTargets.map((t, idx) => ({
        ...t,
        id: `target_${Date.now()}_${idx}`,
      }))
    );
    setTrainTestSplit(profile.trainTestSplit);
    if (profile.geneticConfig) {
      setGeneticConfig(profile.geneticConfig);
    }
    if (profile.metricsConfig) {
      setMetricsConfig(profile.metricsConfig);
    }
    setShowLoadProfileDialog(false);
  };

  const deleteProfile = (profileId: string) => {
    const updatedProfiles = profiles.filter(p => p.id !== profileId);
    setProfiles(updatedProfiles);
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(updatedProfiles));
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Model Training</h1>
        <button
          onClick={handleOpenForm}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center space-x-2"
        >
          <Plus size={16} />
          <span>Create New Job</span>
        </button>
      </div>

      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Create and monitor optimization jobs here.
      </p>

      {/* Job Creation Form Modal */}
      {isFormOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-5xl mx-4 max-h-[90vh] flex flex-col">
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-semibold">Create Optimization Job</h2>
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => setShowLoadProfileDialog(true)}
                  className="flex items-center space-x-1 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <FolderOpen size={14} />
                  <span>Load Profile</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowSaveProfileDialog(true)}
                  className="flex items-center space-x-1 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <Save size={14} />
                  <span>Save Profile</span>
                </button>
                <button
                  onClick={handleCloseForm}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              {/* Dataset Selection */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Select Dataset{useMultiDataset ? 's' : ''}
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useMultiDataset}
                      onChange={(e) => {
                        setUseMultiDataset(e.target.checked);
                        if (!e.target.checked) {
                          setSelectedDatasetIds([]);
                        } else {
                          setSelectedDatasetId(null);
                        }
                      }}
                      className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                    />
                    <span className="text-sm text-gray-600 dark:text-gray-400">Multi-Dataset Mode</span>
                  </label>
                </div>
                {isLoading ? (
                  <div className="animate-pulse bg-gray-200 dark:bg-gray-700 h-10 rounded-md"></div>
                ) : error ? (
                  <div className="text-red-500 text-sm">{error}</div>
                ) : useMultiDataset ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-md p-2">
                    {datasets.map((dataset) => (
                      <label
                        key={dataset.id}
                        className={`flex items-center space-x-3 p-2 rounded-md cursor-pointer transition-colors ${
                          selectedDatasetIds.includes(dataset.id)
                            ? 'bg-green-50 dark:bg-green-900/20 border border-green-500'
                            : 'hover:bg-gray-50 dark:hover:bg-gray-700 border border-transparent'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={selectedDatasetIds.includes(dataset.id)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedDatasetIds([...selectedDatasetIds, dataset.id]);
                            } else {
                              setSelectedDatasetIds(selectedDatasetIds.filter(id => id !== dataset.id));
                            }
                          }}
                          className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                        />
                        <div className="flex-1">
                          <span className="font-medium">{dataset.name}</span>
                          <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
                            ({dataset.ticker} - {dataset.timeframe} - {dataset.rows_count.toLocaleString()} rows)
                          </span>
                        </div>
                      </label>
                    ))}
                    {selectedDatasetIds.length > 0 && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                        {selectedDatasetIds.length} dataset{selectedDatasetIds.length !== 1 ? 's' : ''} selected
                        {selectedDatasetIds.length > 1 && ' - will be combined chronologically'}
                      </p>
                    )}
                  </div>
                ) : (
                  <select
                    value={selectedDatasetId || ''}
                    onChange={(e) => setSelectedDatasetId(e.target.value ? Number(e.target.value) : null)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  >
                    <option value="">Choose a dataset...</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.ticker} - {dataset.timeframe})
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Dataset Details */}
              {selectedDataset && (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Dataset Details</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Database size={16} className="text-gray-400" />
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Ticker</div>
                        <div className="font-medium">{selectedDataset.ticker}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <BarChart2 size={16} className="text-gray-400" />
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Timeframe</div>
                        <div className="font-medium">{selectedDataset.timeframe}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Calendar size={16} className="text-gray-400" />
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Date Range</div>
                        <div className="font-medium">
                          {formatDate(selectedDataset.start_date)} - {formatDate(selectedDataset.end_date)}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <BarChart2 size={16} className="text-gray-400" />
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Rows</div>
                        <div className="font-medium">{selectedDataset.rows_count.toLocaleString()}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Types Selection */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Select Model Types
                  </label>
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
                <div className="grid grid-cols-2 gap-3">
                  {MODEL_TYPES.map((model) => (
                    <label
                      key={model.id}
                      className={`flex items-start space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedModels.includes(model.id)
                          ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                          : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        className="mt-1 w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                      />
                      <div>
                        <div className="flex items-center space-x-2">
                          <Cpu size={14} className="text-gray-400" />
                          <span className="font-medium text-sm">{model.name}</span>
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">{model.description}</span>
                      </div>
                    </label>
                  ))}
                </div>
                {selectedModels.length > 0 && (
                  <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                    {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''} selected
                  </p>
                )}
              </div>

              {/* Parameter Ranges Configuration */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <Sliders size={16} className="text-gray-400" />
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Parameter Ranges
                    </label>
                  </div>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showStepConfig}
                      onChange={(e) => setShowStepConfig(e.target.checked)}
                      className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                    />
                    <span className="text-sm text-gray-600 dark:text-gray-400">Configure Steps</span>
                  </label>
                </div>

                <div className="space-y-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                  {/* Number of Layers */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Number of Layers
                    </label>
                    <div className="flex items-center space-x-3 flex-wrap gap-y-2">
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={parameterRanges.layersMin}
                        onChange={(e) => handleParameterChange('layersMin', parseInt(e.target.value) || 1)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      <span className="text-gray-400">to</span>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={parameterRanges.layersMax}
                        onChange={(e) => handleParameterChange('layersMax', parseInt(e.target.value) || 1)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      {showStepConfig && (
                        <>
                          <span className="text-gray-400">step</span>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            value={parameterRanges.layersStep}
                            onChange={(e) => handleParameterChange('layersStep', parseInt(e.target.value) || 1)}
                            className="w-16 px-2 py-1 border border-green-400 dark:border-green-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                          />
                        </>
                      )}
                      <span className="text-xs text-gray-400">layers</span>
                    </div>
                    {parameterRanges.layersMin > parameterRanges.layersMax && (
                      <p className="text-red-500 text-xs mt-1">Min must be less than or equal to max</p>
                    )}
                  </div>

                  {/* Layer Size */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Layer Size (neurons)
                    </label>
                    <div className="flex items-center space-x-3 flex-wrap gap-y-2">
                      <input
                        type="number"
                        min="8"
                        max="1024"
                        step="8"
                        value={parameterRanges.layerSizeMin}
                        onChange={(e) => handleParameterChange('layerSizeMin', parseInt(e.target.value) || 8)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      <span className="text-gray-400">to</span>
                      <input
                        type="number"
                        min="8"
                        max="1024"
                        step="8"
                        value={parameterRanges.layerSizeMax}
                        onChange={(e) => handleParameterChange('layerSizeMax', parseInt(e.target.value) || 8)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      {showStepConfig && (
                        <>
                          <span className="text-gray-400">step</span>
                          <input
                            type="number"
                            min="8"
                            max="128"
                            step="8"
                            value={parameterRanges.layerSizeStep}
                            onChange={(e) => handleParameterChange('layerSizeStep', parseInt(e.target.value) || 8)}
                            className="w-16 px-2 py-1 border border-green-400 dark:border-green-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                          />
                        </>
                      )}
                      <span className="text-xs text-gray-400">neurons</span>
                    </div>
                    {parameterRanges.layerSizeMin > parameterRanges.layerSizeMax && (
                      <p className="text-red-500 text-xs mt-1">Min must be less than or equal to max</p>
                    )}
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Base size for Transformer/TCN. Scaled per model: LSTM/GRU 4x, N-BEATS 2x.
                      <br />
                      <span className="text-gray-400">
                        e.g., 128 base = Transformer 128, N-BEATS 256, LSTM 512
                      </span>
                    </p>
                  </div>

                  {/* Learning Rate */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Learning Rate
                    </label>
                    <div className="flex items-center space-x-3 flex-wrap gap-y-2">
                      <input
                        type="number"
                        min="0.0001"
                        max="0.1"
                        step="0.0001"
                        value={parameterRanges.learningRateMin}
                        onChange={(e) => handleParameterChange('learningRateMin', parseFloat(e.target.value) || 0.0001)}
                        className="w-24 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      <span className="text-gray-400">to</span>
                      <input
                        type="number"
                        min="0.0001"
                        max="0.1"
                        step="0.0001"
                        value={parameterRanges.learningRateMax}
                        onChange={(e) => handleParameterChange('learningRateMax', parseFloat(e.target.value) || 0.0001)}
                        className="w-24 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      {showStepConfig && (
                        <>
                          <span className="text-gray-400">step</span>
                          <input
                            type="number"
                            min="0.0001"
                            max="0.01"
                            step="0.0001"
                            value={parameterRanges.learningRateStep}
                            onChange={(e) => handleParameterChange('learningRateStep', parseFloat(e.target.value) || 0.0001)}
                            className="w-20 px-2 py-1 border border-green-400 dark:border-green-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                          />
                        </>
                      )}
                    </div>
                    {parameterRanges.learningRateMin > parameterRanges.learningRateMax && (
                      <p className="text-red-500 text-xs mt-1">Min must be less than or equal to max</p>
                    )}
                  </div>

                  {/* Dropout */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Dropout Rate
                    </label>
                    <div className="flex items-center space-x-3 flex-wrap gap-y-2">
                      <input
                        type="number"
                        min="0"
                        max="0.9"
                        step="0.1"
                        value={parameterRanges.dropoutMin}
                        onChange={(e) => handleParameterChange('dropoutMin', parseFloat(e.target.value) || 0)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      <span className="text-gray-400">to</span>
                      <input
                        type="number"
                        min="0"
                        max="0.9"
                        step="0.1"
                        value={parameterRanges.dropoutMax}
                        onChange={(e) => handleParameterChange('dropoutMax', parseFloat(e.target.value) || 0)}
                        className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                      />
                      {showStepConfig && (
                        <>
                          <span className="text-gray-400">step</span>
                          <input
                            type="number"
                            min="0.05"
                            max="0.5"
                            step="0.05"
                            value={parameterRanges.dropoutStep}
                            onChange={(e) => handleParameterChange('dropoutStep', parseFloat(e.target.value) || 0.1)}
                            className="w-16 px-2 py-1 border border-green-400 dark:border-green-600 rounded bg-white dark:bg-gray-800 text-center text-sm"
                          />
                        </>
                      )}
                    </div>
                    {parameterRanges.dropoutMin > parameterRanges.dropoutMax && (
                      <p className="text-red-500 text-xs mt-1">Min must be less than or equal to max</p>
                    )}
                  </div>

                  {/* Activation Functions */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Activation Functions
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {ACTIVATION_FUNCTIONS.map((activation) => (
                        <label
                          key={activation.id}
                          className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer transition-colors text-sm ${
                            parameterRanges.activationFunctions.includes(activation.id)
                              ? 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={parameterRanges.activationFunctions.includes(activation.id)}
                            onChange={() => handleActivationToggle(activation.id)}
                            className="sr-only"
                          />
                          <span>{activation.name}</span>
                        </label>
                      ))}
                    </div>
                    {parameterRanges.activationFunctions.length === 0 && (
                      <p className="text-red-500 text-xs mt-1">Select at least one activation function</p>
                    )}
                    {parameterRanges.activationFunctions.length > 0 && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {parameterRanges.activationFunctions.length} function{parameterRanges.activationFunctions.length !== 1 ? 's' : ''} selected
                      </p>
                    )}
                  </div>
                  {/* Combinations Count */}
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Total Parameter Combinations:</span>
                      <span className="text-lg font-bold text-green-600 dark:text-green-400">{calculateCombinations().toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Optimization Settings */}
              <div className="mb-6">
                <div className="flex items-center space-x-2 mb-3">
                  <Settings size={16} className="text-gray-400" />
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Optimization Settings
                  </label>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="grid grid-cols-2 gap-6">
                    {/* Left Column - Genetic Algorithm */}
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600 pb-2">Genetic Algorithm</h4>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Population Size</label>
                          <input
                            type="number"
                            min="10"
                            max="200"
                            value={geneticConfig.populationSize}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, populationSize: parseInt(e.target.value) || 20 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Generations</label>
                          <input
                            type="number"
                            min="5"
                            max="500"
                            value={geneticConfig.generations}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, generations: parseInt(e.target.value) || 50 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Elitism %</label>
                          <input
                            type="number"
                            min="0"
                            max="50"
                            value={geneticConfig.elitismPercent}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, elitismPercent: parseFloat(e.target.value) || 10 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Early Stop (gens)</label>
                          <input
                            type="number"
                            min="1"
                            max="50"
                            value={geneticConfig.earlyStoppingGenerations}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, earlyStoppingGenerations: parseInt(e.target.value) || 5 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Crossover Prob</label>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={geneticConfig.crossoverProb}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, crossoverProb: parseFloat(e.target.value) || 0.7 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Mutation Prob</label>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={geneticConfig.mutationProb}
                            onChange={(e) => setGeneticConfig(prev => ({ ...prev, mutationProb: parseFloat(e.target.value) || 0.2 }))}
                            className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-sm"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Right Column - Metrics */}
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600 pb-2">Optimization Metric</h4>

                      <div className="space-y-2">
                        {AVAILABLE_METRICS.map((metric) => (
                          <label
                            key={metric.id}
                            className={`flex items-center space-x-3 p-2 rounded-lg cursor-pointer transition-colors ${
                              metricsConfig.optimizeMetric === metric.id
                                ? 'bg-green-100 dark:bg-green-900/30 border border-green-500'
                                : 'hover:bg-gray-100 dark:hover:bg-gray-600 border border-transparent'
                            }`}
                          >
                            <input
                              type="radio"
                              name="optimizeMetric"
                              value={metric.id}
                              checked={metricsConfig.optimizeMetric === metric.id}
                              onChange={(e) => setMetricsConfig({ optimizeMetric: e.target.value })}
                              className="w-4 h-4 text-green-600"
                            />
                            <div>
                              <div className="text-sm font-medium">{metric.name}</div>
                              <div className="text-xs text-gray-500 dark:text-gray-400">{metric.description}</div>
                            </div>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Prediction Targets */}
              <div className="mb-6">
                <div className="flex items-center space-x-2 mb-3">
                  <Target size={16} className="text-gray-400" />
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Prediction Targets
                  </label>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  {/* Preset Buttons */}
                  <div className="mb-4">
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Add Preset
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {PREDICTION_PRESETS.map((preset, idx) => {
                        const isAdded = predictionTargets.some(
                          t => t.profitPercent === preset.profit &&
                               t.maxDrawdownPercent === preset.drawdown &&
                               t.timePeriodDays === preset.days
                        );
                        return (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => addPresetTarget(preset)}
                            disabled={isAdded}
                            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                              isAdded
                                ? 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 cursor-default'
                                : 'border-gray-300 dark:border-gray-600 hover:border-green-500 hover:bg-green-50 dark:hover:bg-green-900/20'
                            }`}
                          >
                            {isAdded ? '✓ ' : '+ '}{preset.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Custom Target Form */}
                  <div className="mb-4">
                    {!showCustomTargetForm ? (
                      <button
                        type="button"
                        onClick={() => setShowCustomTargetForm(true)}
                        className="px-3 py-1.5 text-sm rounded-md border border-dashed border-gray-400 dark:border-gray-500 text-gray-600 dark:text-gray-400 hover:border-green-500 hover:text-green-600 dark:hover:text-green-400 transition-colors"
                      >
                        + Add Custom Target
                      </button>
                    ) : (
                      <div className="p-3 bg-white dark:bg-gray-800 rounded-md border border-green-500">
                        <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                          Custom Target Configuration
                        </label>
                        <div className="flex flex-wrap items-end gap-3">
                          <div>
                            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                              Profit %
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="100"
                              value={customTarget.profitPercent}
                              onChange={(e) => setCustomTarget(prev => ({ ...prev, profitPercent: parseInt(e.target.value) || 0 }))}
                              className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-center text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                              Max DD %
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="50"
                              value={customTarget.maxDrawdownPercent}
                              onChange={(e) => setCustomTarget(prev => ({ ...prev, maxDrawdownPercent: parseInt(e.target.value) || 0 }))}
                              className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-center text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                              Days
                            </label>
                            <input
                              type="number"
                              min="1"
                              max="365"
                              value={customTarget.timePeriodDays}
                              onChange={(e) => setCustomTarget(prev => ({ ...prev, timePeriodDays: parseInt(e.target.value) || 0 }))}
                              className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-center text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={addCustomTarget}
                              disabled={!isCustomTargetValid()}
                              className={`px-3 py-1 text-sm rounded ${
                                isCustomTargetValid()
                                  ? 'bg-green-600 text-white hover:bg-green-700'
                                  : 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
                              }`}
                            >
                              Add
                            </button>
                            <button
                              type="button"
                              onClick={() => setShowCustomTargetForm(false)}
                              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                        {customTarget.maxDrawdownPercent >= customTarget.profitPercent && customTarget.profitPercent > 0 && (
                          <p className="text-red-500 text-xs mt-2">Max drawdown must be less than profit target</p>
                        )}
                        {isCustomTargetValid() && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 font-mono">
                            Will create: <span className="text-green-600 dark:text-green-400">price_up_{customTarget.profitPercent}pct_{customTarget.maxDrawdownPercent}dd_{customTarget.timePeriodDays}d</span>
                            {' | '}
                            <span className="text-red-600 dark:text-red-400">price_down_{customTarget.profitPercent}pct_{customTarget.maxDrawdownPercent}dd_{customTarget.timePeriodDays}d</span>
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Selected Targets */}
                  {predictionTargets.length > 0 && (
                    <div>
                      <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                        Selected Targets (symmetric up/down)
                      </label>
                      <div className="space-y-2">
                        {predictionTargets.map((target) => {
                          const fieldNames = generateTargetFieldNames(target);
                          return (
                            <div
                              key={target.id}
                              className="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-600"
                            >
                              <div className="flex-1">
                                <div className="flex items-center space-x-4 text-sm">
                                  <span className="font-medium">
                                    {target.profitPercent}% profit
                                  </span>
                                  <span className="text-gray-500">|</span>
                                  <span>
                                    {target.maxDrawdownPercent}% max DD
                                  </span>
                                  <span className="text-gray-500">|</span>
                                  <span>
                                    {target.timePeriodDays} days
                                  </span>
                                </div>
                                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-mono">
                                  <span className="text-green-600 dark:text-green-400">{fieldNames.up}</span>
                                  <span className="mx-2">|</span>
                                  <span className="text-red-600 dark:text-red-400">{fieldNames.down}</span>
                                </div>
                              </div>
                              <button
                                type="button"
                                onClick={() => removeTarget(target.id)}
                                className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                              >
                                <Trash2 size={16} />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                        {predictionTargets.length} target pair{predictionTargets.length !== 1 ? 's' : ''} configured ({predictionTargets.length * 2} output fields)
                      </p>
                    </div>
                  )}

                  {predictionTargets.length === 0 && !showCustomTargetForm && (
                    <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                      Click a preset or add a custom target to define prediction targets
                    </p>
                  )}
                </div>
              </div>

              {/* Train/Test Split */}
              <div className="mb-6">
                <div className="flex items-center space-x-2 mb-3">
                  <Split size={16} className="text-gray-400" />
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Train/Test Split
                  </label>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  {/* Split Slider */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-blue-600 dark:text-blue-400 font-medium">
                        Training: {trainTestSplit}%
                      </span>
                      <span className="text-orange-600 dark:text-orange-400 font-medium">
                        Testing: {100 - trainTestSplit}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="95"
                      step="5"
                      value={trainTestSplit}
                      onChange={(e) => setTrainTestSplit(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>50%</span>
                      <span>95%</span>
                    </div>
                  </div>

                  {/* Visual Representation */}
                  <div className="mb-3">
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Data Split Visualization
                    </label>
                    <div className="flex h-6 rounded-md overflow-hidden">
                      <div
                        className="bg-blue-500 dark:bg-blue-600 flex items-center justify-center text-xs text-white font-medium transition-all duration-200"
                        style={{ width: `${trainTestSplit}%` }}
                      >
                        {trainTestSplit >= 60 && 'Train'}
                      </div>
                      <div
                        className="bg-orange-500 dark:bg-orange-600 flex items-center justify-center text-xs text-white font-medium transition-all duration-200"
                        style={{ width: `${100 - trainTestSplit}%` }}
                      >
                        {100 - trainTestSplit >= 15 && 'Test'}
                      </div>
                    </div>
                  </div>

                  {/* Quick Presets */}
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-300 mb-2">
                      Quick Presets
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {[70, 80, 90].map((split) => (
                        <button
                          key={split}
                          type="button"
                          onClick={() => setTrainTestSplit(split)}
                          className={`px-3 py-1 text-sm rounded-md border transition-colors ${
                            trainTestSplit === split
                              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                              : 'border-gray-300 dark:border-gray-600 hover:border-blue-500'
                          }`}
                        >
                          {split}/{100 - split}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Dataset Rows Info */}
                  {selectedDataset && (
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        With {selectedDataset.rows_count.toLocaleString()} total rows:
                        <span className="text-blue-600 dark:text-blue-400 ml-2">
                          ~{Math.floor(selectedDataset.rows_count * trainTestSplit / 100).toLocaleString()} training
                        </span>
                        <span className="text-orange-600 dark:text-orange-400 ml-2">
                          ~{Math.ceil(selectedDataset.rows_count * (100 - trainTestSplit) / 100).toLocaleString()} testing
                        </span>
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-between items-center p-4 border-t border-gray-200 dark:border-gray-700">
              {/* Show missing requirements */}
              <div className="text-sm text-red-500 dark:text-red-400">
                {!selectedDatasetId && <span className="mr-3">⚠ Select dataset</span>}
                {selectedModels.length === 0 && <span className="mr-3">⚠ Select models</span>}
                {!isParameterValid() && <span className="mr-3">⚠ Fix parameters</span>}
                {predictionTargets.length === 0 && <span className="mr-3">⚠ Add targets</span>}
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={handleCloseForm}
                  className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={submitJob}
                  disabled={!selectedDatasetId || selectedModels.length === 0 || !isParameterValid() || predictionTargets.length === 0 || isSubmitting}
                  className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                    selectedDatasetId && selectedModels.length > 0 && isParameterValid() && predictionTargets.length > 0 && !isSubmitting
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
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Load Profile Dialog */}
      {showLoadProfileDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4">
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold">Load Profile</h3>
              <button
                onClick={() => setShowLoadProfileDialog(false)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <X size={18} />
              </button>
            </div>
            <div className="p-4 max-h-80 overflow-y-auto">
              {profiles.length === 0 ? (
                <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                  No saved profiles yet. Save your first profile to get started.
                </p>
              ) : (
                <div className="space-y-2">
                  {profiles.map((profile) => (
                    <div
                      key={profile.id}
                      className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <div className="flex-1 cursor-pointer" onClick={() => loadProfile(profile)}>
                        <div className="font-medium">{profile.name}</div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {profile.selectedModels.length} model{profile.selectedModels.length !== 1 ? 's' : ''} |
                          {profile.predictionTargets.length} target{profile.predictionTargets.length !== 1 ? 's' : ''} |
                          {profile.trainTestSplit}/{100 - profile.trainTestSplit} split
                        </div>
                        <div className="text-xs text-gray-400 dark:text-gray-500">
                          Created: {new Date(profile.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteProfile(profile.id);
                        }}
                        className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex justify-end p-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowLoadProfileDialog(false)}
                className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Save Profile Dialog */}
      {showSaveProfileDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4">
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold">Save as Profile</h3>
              <button
                onClick={() => {
                  setShowSaveProfileDialog(false);
                  setNewProfileName('');
                  setSaveProfileSuccess(false);
                }}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <X size={18} />
              </button>
            </div>
            <div className="p-4">
              {saveProfileSuccess ? (
                <div className="text-center py-8">
                  <div className="text-green-500 text-4xl mb-2">✓</div>
                  <p className="text-green-600 dark:text-green-400 font-medium">Profile saved successfully!</p>
                </div>
              ) : (
                <>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Profile Name
                  </label>
                  <input
                    type="text"
                    value={newProfileName}
                    onChange={(e) => setNewProfileName(e.target.value)}
                    placeholder="Enter profile name..."
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    autoFocus
                  />
                  <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-md text-sm">
                    <p className="font-medium text-gray-700 dark:text-gray-300 mb-2">Profile will include:</p>
                    <ul className="text-gray-500 dark:text-gray-400 space-y-1">
                      <li>• {selectedModels.length} model type{selectedModels.length !== 1 ? 's' : ''}</li>
                      <li>• Parameter ranges configuration</li>
                      <li>• {predictionTargets.length} prediction target{predictionTargets.length !== 1 ? 's' : ''}</li>
                      <li>• {trainTestSplit}/{100 - trainTestSplit} train/test split</li>
                    </ul>
                  </div>
                </>
              )}
            </div>
            {!saveProfileSuccess && (
              <div className="flex justify-end space-x-3 p-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => {
                    setShowSaveProfileDialog(false);
                    setNewProfileName('');
                  }}
                  className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={saveProfile}
                  disabled={!newProfileName.trim()}
                  className={`px-4 py-2 rounded-md ${
                    newProfileName.trim()
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Save
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Job Monitor View */}
      {selectedJobId && jobProgress && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-5xl mx-4 max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-4">
                <button
                  onClick={closeJobMonitor}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                >
                  <ArrowLeft size={20} />
                </button>
                <div>
                  <h2 className="text-xl font-semibold">Job #{jobProgress.job.id}</h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {datasets.find(d => d.id === jobProgress.job.datasetId)?.ticker || 'Unknown Dataset'}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {/* Control Buttons */}
                {jobProgress.job.status === 'running' && (
                  <button
                    onClick={() => handlePauseJob(jobProgress.job.id)}
                    className="flex items-center space-x-1 px-3 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600 transition-colors"
                  >
                    <Pause size={16} />
                    <span>Pause</span>
                  </button>
                )}
                {jobProgress.job.status === 'paused' && (
                  <button
                    onClick={() => handleResumeJob(jobProgress.job.id)}
                    className="flex items-center space-x-1 px-3 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
                  >
                    <SkipForward size={16} />
                    <span>Resume</span>
                  </button>
                )}
                {['running', 'paused', 'queued'].includes(jobProgress.job.status) && (
                  <button
                    onClick={() => handleCancelJob(jobProgress.job.id)}
                    className="flex items-center space-x-1 px-3 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors"
                  >
                    <XCircle size={16} />
                    <span>Cancel</span>
                  </button>
                )}
                <button
                  onClick={closeJobMonitor}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              {isLoadingProgress ? (
                <div className="flex items-center justify-center h-64">
                  <Loader2 className="animate-spin text-blue-500" size={48} />
                </div>
              ) : (
                <>
                  {/* Status Overview */}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    {/* Generation Progress */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
                        <Activity size={16} />
                        <span className="text-xs">Generation</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {jobProgress.job.currentGeneration || 0}
                        <span className="text-sm text-gray-500 dark:text-gray-400 font-normal">
                          /{jobProgress.job.totalGenerations || 50}
                        </span>
                      </div>
                    </div>

                    {/* Best Fitness */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
                        <Target size={16} />
                        <span className="text-xs">Best Fitness</span>
                      </div>
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {jobProgress.job.bestFitness?.toFixed(2) || '--'}
                      </div>
                    </div>

                    {/* GPU Utilization */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
                        <Zap size={16} />
                        <span className="text-xs">GPU Usage</span>
                      </div>
                      <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                        {jobProgress.job.gpuUtilization?.toFixed(0) || '--'}%
                      </div>
                    </div>

                    {/* ETA */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
                        <Timer size={16} />
                        <span className="text-xs">Time Remaining</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {jobProgress.job.estimatedTimeRemaining || '--'}
                      </div>
                    </div>

                    {/* Status */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
                        <Clock size={16} />
                        <span className="text-xs">Status</span>
                      </div>
                      <div className={`text-2xl font-bold ${
                        jobProgress.job.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                        jobProgress.job.status === 'running' ? 'text-blue-600 dark:text-blue-400' :
                        jobProgress.job.status === 'paused' ? 'text-yellow-600 dark:text-yellow-400' :
                        jobProgress.job.status === 'failed' || jobProgress.job.status === 'cancelled' ? 'text-red-600 dark:text-red-400' :
                        'text-gray-600 dark:text-gray-400'
                      }`}>
                        {jobProgress.job.status.charAt(0).toUpperCase() + jobProgress.job.status.slice(1)}
                      </div>
                    </div>
                  </div>

                  {/* Overall Progress Bar */}
                  <div className="mb-6">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-500 dark:text-gray-400">Progress</span>
                      <span className="font-medium">{jobProgress.job.progress.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full transition-all duration-300 ${
                          jobProgress.job.status === 'completed' ? 'bg-green-500' :
                          jobProgress.job.status === 'paused' ? 'bg-yellow-500' :
                          'bg-blue-500'
                        }`}
                        style={{ width: `${jobProgress.job.progress}%` }}
                      />
                    </div>
                  </div>

                  {/* Charts */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    {/* Loss Chart */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Loss Over Generations</h3>
                      {jobProgress.metrics.length > 0 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={jobProgress.metrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="generation" stroke="#6B7280" fontSize={12} />
                            <YAxis stroke="#6B7280" fontSize={12} />
                            <Tooltip
                              contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                              labelStyle={{ color: '#9CA3AF' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#EF4444" name="Train Loss" dot={false} strokeWidth={2} />
                            <Line type="monotone" dataKey="valLoss" stroke="#F97316" name="Val Loss" dot={false} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-48 flex items-center justify-center text-gray-500 dark:text-gray-400">
                          Waiting for training data...
                        </div>
                      )}
                    </div>

                    {/* Accuracy Chart */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Accuracy Over Generations</h3>
                      {jobProgress.metrics.length > 0 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={jobProgress.metrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="generation" stroke="#6B7280" fontSize={12} />
                            <YAxis stroke="#6B7280" fontSize={12} domain={[0, 1]} />
                            <Tooltip
                              contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                              labelStyle={{ color: '#9CA3AF' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#22C55E" name="Train Acc" dot={false} strokeWidth={2} />
                            <Line type="monotone" dataKey="valAccuracy" stroke="#10B981" name="Val Acc" dot={false} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-48 flex items-center justify-center text-gray-500 dark:text-gray-400">
                          Waiting for training data...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Logs Section */}
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <button
                      onClick={() => setShowLogs(!showLogs)}
                      className="flex items-center justify-between w-full text-left"
                    >
                      <div className="flex items-center space-x-2">
                        <FileText size={16} className="text-gray-500" />
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Training Logs</h3>
                      </div>
                      <span className="text-gray-500">{showLogs ? '▲' : '▼'}</span>
                    </button>
                    {showLogs && (
                      <div className="mt-4 bg-gray-900 rounded-md p-4 max-h-48 overflow-y-auto font-mono text-sm">
                        {jobProgress.logs.length > 0 ? (
                          jobProgress.logs.map((log, idx) => (
                            <div key={idx} className="text-green-400">
                              {log}
                            </div>
                          ))
                        ) : (
                          <div className="text-gray-500">No logs yet...</div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Model Individuals Visualization Section */}
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mt-6">
                    <button
                      onClick={() => {
                        setShowIndividuals(!showIndividuals);
                        if (!showIndividuals && !individualsData) {
                          fetchIndividuals(jobProgress.job.id);
                          fetchGenerations(jobProgress.job.id);
                        }
                      }}
                      className="flex items-center justify-between w-full text-left"
                    >
                      <div className="flex items-center space-x-2">
                        <BarChart2 size={16} className="text-gray-500" />
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          Model Individuals ({individualsData?.summary?.total_individuals || 0} total)
                        </h3>
                      </div>
                      <span className="text-gray-500">{showIndividuals ? '▲' : '▼'}</span>
                    </button>

                    {showIndividuals && (
                      <div className="mt-4 space-y-4">
                        {/* Summary Stats */}
                        {individualsData && (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <div className="text-xs text-gray-500 dark:text-gray-400">Total Individuals</div>
                              <div className="text-xl font-bold">{individualsData.summary.total_individuals}</div>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <div className="text-xs text-gray-500 dark:text-gray-400">Generations</div>
                              <div className="text-xl font-bold">{individualsData.summary.generations.length}</div>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <div className="text-xs text-gray-500 dark:text-gray-400">Best Fitness</div>
                              <div className="text-xl font-bold text-green-600">{individualsData.summary.best_fitness.toFixed(4)}</div>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <div className="text-xs text-gray-500 dark:text-gray-400">Avg Fitness</div>
                              <div className="text-xl font-bold">{individualsData.summary.avg_fitness.toFixed(4)}</div>
                            </div>
                          </div>
                        )}

                        {/* Best Individual */}
                        {individualsData?.best_individual && (
                          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-green-500">
                            <h4 className="text-sm font-semibold text-green-600 mb-2">Best Individual</h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                              <div>
                                <span className="text-gray-500">Model:</span>{' '}
                                <span className="font-medium">{individualsData.best_individual.model_type.toUpperCase()}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">Generation:</span>{' '}
                                <span className="font-medium">{individualsData.best_individual.generation}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">Fitness:</span>{' '}
                                <span className="font-medium text-green-600">{individualsData.best_individual.fitness.toFixed(4)}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">MAPE:</span>{' '}
                                <span className="font-medium">{(individualsData.best_individual.metrics?.mape || 0).toFixed(2)}%</span>
                              </div>
                            </div>
                            <div className="mt-2 text-xs text-gray-500">
                              <strong>Params:</strong>{' '}
                              {Object.entries(individualsData.best_individual.params || {}).map(([k, v]) => (
                                <span key={k} className="mr-2">{k}={typeof v === 'number' ? v.toFixed(4) : v}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Filters */}
                        <div className="flex items-center space-x-4">
                          <div>
                            <label className="text-xs text-gray-500 dark:text-gray-400 mr-2">Generation:</label>
                            <select
                              value={selectedGeneration ?? ''}
                              onChange={(e) => setSelectedGeneration(e.target.value ? parseInt(e.target.value) : null)}
                              className="px-2 py-1 text-sm border rounded dark:bg-gray-800 dark:border-gray-600"
                            >
                              <option value="">All</option>
                              {individualsData?.summary.generations.map(g => (
                                <option key={g} value={g}>Gen {g}</option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="text-xs text-gray-500 dark:text-gray-400 mr-2">Model:</label>
                            <select
                              value={selectedModelTypeFilter}
                              onChange={(e) => setSelectedModelTypeFilter(e.target.value)}
                              className="px-2 py-1 text-sm border rounded dark:bg-gray-800 dark:border-gray-600"
                            >
                              <option value="">All</option>
                              {individualsData?.summary.model_types.map(m => (
                                <option key={m} value={m}>{m.toUpperCase()}</option>
                              ))}
                            </select>
                          </div>
                          <button
                            onClick={() => {
                              fetchIndividuals(jobProgress.job.id);
                              fetchGenerations(jobProgress.job.id);
                            }}
                            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                          >
                            Refresh
                          </button>
                        </div>

                        {/* Fitness by Generation Chart */}
                        {generationsData && generationsData.generations.length > 0 && (
                          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Fitness by Generation</h4>
                            <ResponsiveContainer width="100%" height={200}>
                              <LineChart data={generationsData.generations}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis dataKey="generation" stroke="#6B7280" fontSize={12} />
                                <YAxis stroke="#6B7280" fontSize={12} domain={[0, 1]} />
                                <Tooltip
                                  contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                                  labelStyle={{ color: '#9CA3AF' }}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="best_fitness" stroke="#22C55E" name="Best" strokeWidth={2} />
                                <Line type="monotone" dataKey="avg_fitness" stroke="#3B82F6" name="Average" strokeWidth={2} />
                                <Line type="monotone" dataKey="min_fitness" stroke="#EF4444" name="Min" strokeWidth={1} dot={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* Model Type Distribution */}
                        {generationsData && generationsData.generations.length > 0 && (
                          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Model Type Distribution (Latest Gen)</h4>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(generationsData.generations[generationsData.generations.length - 1]?.model_types || {}).map(([type, count]) => {
                                const colors: Record<string, string> = {
                                  lstm: 'bg-blue-500',
                                  gru: 'bg-green-500',
                                  nbeats: 'bg-purple-500',
                                  tcn: 'bg-orange-500',
                                  transformer: 'bg-pink-500',
                                };
                                return (
                                  <div key={type} className={`${colors[type] || 'bg-gray-500'} text-white px-3 py-1 rounded-full text-sm`}>
                                    {type.toUpperCase()}: {count}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Individuals Table */}
                        {individualsData && (
                          <div className="bg-white dark:bg-gray-800 rounded-lg overflow-hidden">
                            <div className="max-h-64 overflow-y-auto">
                              <table className="w-full text-sm">
                                <thead className="bg-gray-100 dark:bg-gray-700 sticky top-0">
                                  <tr>
                                    <th className="px-3 py-2 text-left">Gen</th>
                                    <th className="px-3 py-2 text-left">Model</th>
                                    <th className="px-3 py-2 text-left">Fitness</th>
                                    <th className="px-3 py-2 text-left">MAPE</th>
                                    <th className="px-3 py-2 text-left">Key Params</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {individualsData.individuals
                                    .filter(ind =>
                                      (selectedGeneration === null || ind.generation === selectedGeneration) &&
                                      (selectedModelTypeFilter === '' || ind.model_type === selectedModelTypeFilter)
                                    )
                                    .sort((a, b) => b.fitness - a.fitness)
                                    .slice(0, 50)
                                    .map((ind, idx) => (
                                      <tr key={idx} className={`border-t dark:border-gray-700 ${idx === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''}`}>
                                        <td className="px-3 py-2">{ind.generation}</td>
                                        <td className="px-3 py-2">
                                          <span className={`px-2 py-0.5 rounded text-xs ${
                                            ind.model_type === 'lstm' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300' :
                                            ind.model_type === 'gru' ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300' :
                                            ind.model_type === 'nbeats' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300' :
                                            ind.model_type === 'tcn' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300' :
                                            'bg-pink-100 text-pink-800 dark:bg-pink-900/50 dark:text-pink-300'
                                          }`}>
                                            {ind.model_type.toUpperCase()}
                                          </span>
                                        </td>
                                        <td className="px-3 py-2 font-medium">{ind.fitness.toFixed(4)}</td>
                                        <td className="px-3 py-2">{(ind.metrics?.mape || 0).toFixed(2)}%</td>
                                        <td className="px-3 py-2 text-xs text-gray-500">
                                          {ind.params?.hidden_dim && `dim=${ind.params.hidden_dim}`}
                                          {ind.params?.n_rnn_layers && ` layers=${ind.params.n_rnn_layers}`}
                                          {ind.params?.learning_rate && ` lr=${Number(ind.params.learning_rate).toFixed(4)}`}
                                        </td>
                                      </tr>
                                    ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Jobs List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Optimization Jobs</h2>
        {jobs.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400">
            No optimization jobs yet. Click "Create New Job" to get started.
          </p>
        ) : (
          <div className="space-y-3">
            {jobs.map((job) => {
              const dataset = datasets.find(d => d.id === job.datasetId);
              const statusIcons: Record<string, React.ReactNode> = {
                queued: <Clock size={16} className="text-yellow-500" />,
                running: <Loader2 size={16} className="text-blue-500 animate-spin" />,
                paused: <Pause size={16} className="text-yellow-500" />,
                stopped: <AlertCircle size={16} className="text-orange-500" />,
                completed: <CheckCircle size={16} className="text-green-500" />,
                failed: <AlertCircle size={16} className="text-red-500" />,
                cancelled: <XCircle size={16} className="text-gray-500" />,
              };
              const statusIcon = statusIcons[job.status] || statusIcons.queued;
              const statusColors: Record<string, string> = {
                queued: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300',
                running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300',
                paused: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300',
                stopped: 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-300',
                completed: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
                failed: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300',
                cancelled: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
              };
              const statusColor = statusColors[job.status] || statusColors.queued;

              return (
                <div
                  key={job.id}
                  onClick={() => openJobMonitor(job.id)}
                  className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors cursor-pointer"
                >
                  <div className="flex items-center space-x-4">
                    {statusIcon}
                    <div>
                      <div className="font-medium">
                        Job #{job.id}
                        {dataset && (
                          <span className="text-gray-500 dark:text-gray-400 ml-2">
                            - {dataset.ticker} ({dataset.timeframe})
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {job.selectedModels.length} model{job.selectedModels.length !== 1 ? 's' : ''} |
                        Created: {new Date(job.createdAt).toLocaleString()}
                        {job.status === 'running' && job.currentGeneration !== undefined && (
                          <span className="ml-2">| Gen {job.currentGeneration}/{job.totalGenerations || 50}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 text-xs rounded-full ${statusColor}`}>
                      {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </span>
                    {(job.status === 'running' || job.status === 'paused') && (
                      <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            job.status === 'paused' ? 'bg-yellow-500' : 'bg-blue-500'
                          }`}
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    )}
                    <button
                      onClick={(e) => handleDeleteJob(job.id, e)}
                      className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                      title="Delete job"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default Training;
