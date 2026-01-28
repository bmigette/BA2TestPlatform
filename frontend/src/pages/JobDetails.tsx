import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft, Clock, CheckCircle, AlertCircle, Loader2, Pause, Play,
  XCircle, Activity, Target, Zap, Timer, ChevronDown, ChevronRight,
  Info, FileText, Cpu, MemoryStick, RefreshCw
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts';

interface Job {
  id: string;
  datasetId: number;
  selectedModels: string[];
  status: 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'stopped';
  progress: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  currentGeneration?: number;
  totalGenerations?: number;
  bestFitness?: number;
  gpuUtilization?: number;
  // Training progress
  currentEpoch?: number;
  totalEpochs?: number;
  currentIndividual?: number;
  populationSize?: number;
  currentModelType?: string;
}

interface Individual {
  generation: number;
  individual: number;
  model_type: string;
  params: Record<string, number | string>;
  fitness: number;
  metrics: Record<string, number>;
  training_history?: Array<{ epoch: number; loss: number; val_loss?: number }>;
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

interface SystemResources {
  cpuPercent: number;
  memoryUsedMB: number;
  memoryTotalMB: number;
  memoryPercent: number;
  gpuUtilization: number | null;
  gpuMemoryUsedMB: number | null;
  gpuMemoryTotalMB: number | null;
}

const MODEL_COLORS: Record<string, string> = {
  lstm: 'bg-blue-500',
  gru: 'bg-green-500',
  nbeats: 'bg-purple-500',
  tcn: 'bg-orange-500',
  transformer: 'bg-pink-500',
};

const MODEL_TEXT_COLORS: Record<string, string> = {
  lstm: 'text-blue-600 bg-blue-100 dark:bg-blue-900/50 dark:text-blue-300',
  gru: 'text-green-600 bg-green-100 dark:bg-green-900/50 dark:text-green-300',
  nbeats: 'text-purple-600 bg-purple-100 dark:bg-purple-900/50 dark:text-purple-300',
  tcn: 'text-orange-600 bg-orange-100 dark:bg-orange-900/50 dark:text-orange-300',
  transformer: 'text-pink-600 bg-pink-100 dark:bg-pink-900/50 dark:text-pink-300',
};

const JobDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<Job | null>(null);
  const [generationsData, setGenerationsData] = useState<GenerationsData | null>(null);
  const [individualsData, setIndividualsData] = useState<IndividualsData | null>(null);
  const [resources, setResources] = useState<SystemResources | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedGenerations, setExpandedGenerations] = useState<Set<number>>(new Set());
  const [showLogs, setShowLogs] = useState(false);
  const [selectedIndividual, setSelectedIndividual] = useState<Individual | null>(null);
  const [elapsedTime, setElapsedTime] = useState<string>('');

  const fetchJob = useCallback(async () => {
    if (!id) return;
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${id}/progress`);
      if (!response.ok) throw new Error('Failed to fetch job');
      const data = await response.json();
      setJob(data.job);
      setLogs(data.logs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  const fetchGenerations = useCallback(async () => {
    if (!id) return;
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${id}/generations`);
      if (response.ok) {
        const data: GenerationsData = await response.json();
        setGenerationsData(data);
      }
    } catch (err) {
      console.error('Failed to fetch generations:', err);
    }
  }, [id]);

  const fetchIndividuals = useCallback(async () => {
    if (!id) return;
    try {
      const response = await fetch(`http://localhost:8002/api/jobs/${id}/individuals`);
      if (response.ok) {
        const data: IndividualsData = await response.json();
        setIndividualsData(data);
      }
    } catch (err) {
      console.error('Failed to fetch individuals:', err);
    }
  }, [id]);

  const fetchResources = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8002/api/dashboard/stats');
      if (response.ok) {
        const data = await response.json();
        setResources(data.systemResources);
      }
    } catch (err) {
      console.error('Failed to fetch resources:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchJob();
    fetchGenerations();
    fetchIndividuals();
    fetchResources();
  }, [fetchJob, fetchGenerations, fetchIndividuals, fetchResources]);

  // Auto-refresh for running jobs - fast refresh for resources and progress
  useEffect(() => {
    if (job?.status === 'running' || job?.status === 'paused') {
      // Fast refresh for job progress and resources (1.5s)
      const fastInterval = setInterval(() => {
        fetchJob();
        fetchResources();
      }, 1500);

      // Slower refresh for generations and individuals (5s)
      const slowInterval = setInterval(() => {
        fetchGenerations();
        fetchIndividuals();
      }, 5000);

      return () => {
        clearInterval(fastInterval);
        clearInterval(slowInterval);
      };
    }
  }, [job?.status, fetchJob, fetchGenerations, fetchIndividuals, fetchResources]);

  // Update elapsed time
  useEffect(() => {
    const updateElapsed = () => {
      if (!job?.startedAt) {
        setElapsedTime('--');
        return;
      }
      const start = new Date(job.startedAt).getTime();
      const end = job.completedAt ? new Date(job.completedAt).getTime() : Date.now();
      const diff = end - start;
      const hours = Math.floor(diff / 3600000);
      const minutes = Math.floor((diff % 3600000) / 60000);
      const seconds = Math.floor((diff % 60000) / 1000);
      setElapsedTime(
        hours > 0 ? `${hours}h ${minutes}m ${seconds}s` : `${minutes}m ${seconds}s`
      );
    };

    updateElapsed();
    if (job?.status === 'running') {
      const interval = setInterval(updateElapsed, 1000);
      return () => clearInterval(interval);
    }
  }, [job?.startedAt, job?.completedAt, job?.status]);

  const handlePause = async () => {
    try {
      await fetch(`http://localhost:8002/api/jobs/${id}/pause`, { method: 'POST' });
      fetchJob();
    } catch (err) {
      console.error('Failed to pause job:', err);
    }
  };

  const handleResume = async () => {
    try {
      await fetch(`http://localhost:8002/api/jobs/${id}/resume`, { method: 'POST' });
      fetchJob();
    } catch (err) {
      console.error('Failed to resume job:', err);
    }
  };

  const handleCancel = async () => {
    if (!confirm('Are you sure you want to cancel this job?')) return;
    try {
      await fetch(`http://localhost:8002/api/jobs/${id}/cancel`, { method: 'POST' });
      fetchJob();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const toggleGeneration = (gen: number) => {
    setExpandedGenerations(prev => {
      const next = new Set(prev);
      if (next.has(gen)) {
        next.delete(gen);
      } else {
        next.add(gen);
      }
      return next;
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Loader2 size={20} className="text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle size={20} className="text-green-500" />;
      case 'failed': return <AlertCircle size={20} className="text-red-500" />;
      case 'paused': return <Pause size={20} className="text-yellow-500" />;
      case 'stopped': return <AlertCircle size={20} className="text-orange-500" />;
      case 'cancelled': return <XCircle size={20} className="text-gray-500" />;
      default: return <Clock size={20} className="text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'paused': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'stopped': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 size={48} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="p-6">
        <div className="bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 p-4 rounded-lg">
          {error || 'Job not found'}
        </div>
        <button
          onClick={() => navigate('/training')}
          className="mt-4 flex items-center space-x-2 text-blue-500 hover:underline"
        >
          <ArrowLeft size={16} />
          <span>Back to Training</span>
        </button>
      </div>
    );
  }

  const chartData = generationsData?.generations.map(g => ({
    generation: g.generation + 1,
    best: g.best_fitness,
    avg: g.avg_fitness,
    min: g.min_fitness,
  })) || [];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/training')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold">Job #{job.id}</h1>
              <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center space-x-1 ${getStatusColor(job.status)}`}>
                {getStatusIcon(job.status)}
                <span className="ml-1">{job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
              </span>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Models: {job.selectedModels.map(m => m.toUpperCase()).join(', ')}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => { fetchJob(); fetchGenerations(); fetchIndividuals(); }}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
            title="Refresh"
          >
            <RefreshCw size={18} />
          </button>
          {job.status === 'running' && (
            <button
              onClick={handlePause}
              className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 flex items-center space-x-2"
            >
              <Pause size={16} />
              <span>Pause</span>
            </button>
          )}
          {(job.status === 'paused' || job.status === 'stopped') && (
            <button
              onClick={handleResume}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 flex items-center space-x-2"
            >
              <Play size={16} />
              <span>Resume</span>
            </button>
          )}
          {['running', 'paused', 'queued'].includes(job.status) && (
            <button
              onClick={handleCancel}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 flex items-center space-x-2"
            >
              <XCircle size={16} />
              <span>Cancel</span>
            </button>
          )}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Target size={16} />
            <span className="text-xs">Best Fitness</span>
          </div>
          <div className="text-2xl font-bold text-green-600">
            {individualsData?.summary?.best_fitness?.toFixed(4) || job.bestFitness?.toFixed(4) || '--'}
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Activity size={16} />
            <span className="text-xs">Generations</span>
          </div>
          <div className="text-2xl font-bold">
            {generationsData?.total_generations || job.currentGeneration || 0}
            <span className="text-sm text-gray-500 font-normal">/{job.totalGenerations || 50}</span>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Timer size={16} />
            <span className="text-xs">{job.completedAt ? 'Total Time' : 'Elapsed'}</span>
          </div>
          <div className="text-2xl font-bold">{elapsedTime}</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Info size={16} />
            <span className="text-xs">Individuals</span>
          </div>
          <div className="text-2xl font-bold">{individualsData?.summary?.total_individuals || 0}</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Zap size={16} />
            <span className="text-xs">GPU Usage</span>
          </div>
          <div className="text-2xl font-bold text-purple-600">
            {resources?.gpuUtilization != null ? `${resources.gpuUtilization}%` : '--'}
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 mb-1">
            <Clock size={16} />
            <span className="text-xs">Progress</span>
          </div>
          <div className="text-2xl font-bold">{job.progress.toFixed(1)}%</div>
        </div>
      </div>

      {/* Training Progress (for running jobs) */}
      {job.status === 'running' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Training Progress</h3>
          <div className="space-y-4">
            {/* Current Model Epoch Progress */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-2">
                  <Activity size={14} className="text-orange-500" />
                  <span>Current Model</span>
                  {job.currentModelType && (
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${MODEL_TEXT_COLORS[job.currentModelType] || 'bg-gray-100'}`}>
                      {job.currentModelType.toUpperCase()}
                    </span>
                  )}
                </span>
                <span className="font-mono">
                  Epoch {job.currentEpoch || 0}/{job.totalEpochs || 10}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div
                  className="h-4 rounded-full bg-gradient-to-r from-orange-400 to-orange-500 transition-all duration-300 flex items-center justify-center"
                  style={{ width: `${job.totalEpochs ? ((job.currentEpoch || 0) / job.totalEpochs) * 100 : 0}%`, minWidth: job.currentEpoch ? '2rem' : 0 }}
                >
                  {job.currentEpoch ? (
                    <span className="text-xs text-white font-medium">
                      {Math.round(((job.currentEpoch || 0) / (job.totalEpochs || 10)) * 100)}%
                    </span>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Generation Progress (individuals in current generation) */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-2">
                  <Target size={14} className="text-blue-500" />
                  <span>Generation {(job.currentGeneration || 0) + 1}</span>
                </span>
                <span className="font-mono">
                  Individual {job.currentIndividual || 0}/{job.populationSize || 20}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div
                  className="h-4 rounded-full bg-gradient-to-r from-blue-400 to-blue-500 transition-all duration-300 flex items-center justify-center"
                  style={{ width: `${job.populationSize ? ((job.currentIndividual || 0) / job.populationSize) * 100 : 0}%`, minWidth: job.currentIndividual ? '2rem' : 0 }}
                >
                  {job.currentIndividual ? (
                    <span className="text-xs text-white font-medium">
                      {Math.round(((job.currentIndividual || 0) / (job.populationSize || 20)) * 100)}%
                    </span>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Overall Progress (generations) */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-2">
                  <Timer size={14} className="text-green-500" />
                  <span>Overall Progress</span>
                </span>
                <span className="font-mono">
                  Generation {(job.currentGeneration || 0) + 1}/{job.totalGenerations || 50}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div
                  className="h-4 rounded-full bg-gradient-to-r from-green-400 to-green-500 transition-all duration-300 flex items-center justify-center"
                  style={{ width: `${job.progress}%`, minWidth: job.progress > 0 ? '2rem' : 0 }}
                >
                  <span className="text-xs text-white font-medium">
                    {job.progress.toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* System Resources (for running jobs) */}
      {job.status === 'running' && resources && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">System Resources</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* CPU */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-1">
                  <Cpu size={14} />
                  <span>CPU</span>
                </span>
                <span>{resources.cpuPercent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                <div
                  className="h-3 rounded-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${resources.cpuPercent}%` }}
                />
              </div>
            </div>
            {/* Memory */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-1">
                  <MemoryStick size={14} />
                  <span>Memory</span>
                </span>
                <span>{resources.memoryPercent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                <div
                  className="h-3 rounded-full bg-green-500 transition-all duration-300"
                  style={{ width: `${resources.memoryPercent}%` }}
                />
              </div>
            </div>
            {/* GPU */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center space-x-1">
                  <Zap size={14} />
                  <span>GPU</span>
                </span>
                <span>{resources.gpuUtilization != null ? `${resources.gpuUtilization}%` : 'N/A'}</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                <div
                  className="h-3 rounded-full bg-purple-500 transition-all duration-300"
                  style={{ width: `${resources.gpuUtilization || 0}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fitness Chart */}
      {chartData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Fitness Over Generations</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="generation" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={12} domain={[0, 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Line type="monotone" dataKey="best" stroke="#22C55E" name="Best" strokeWidth={2} />
              <Line type="monotone" dataKey="avg" stroke="#3B82F6" name="Average" strokeWidth={2} />
              <Line type="monotone" dataKey="min" stroke="#EF4444" name="Min" strokeWidth={1} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Best Individual */}
      {individualsData?.best_individual && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow border-2 border-green-500">
          <h3 className="text-sm font-semibold text-green-600 mb-3">Best Individual</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Model:</span>{' '}
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${MODEL_TEXT_COLORS[individualsData.best_individual.model_type] || 'bg-gray-100'}`}>
                {individualsData.best_individual.model_type.toUpperCase()}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Generation:</span>{' '}
              <span className="font-medium">{individualsData.best_individual.generation + 1}</span>
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
          <div className="mt-3 text-xs text-gray-500 flex flex-wrap gap-2">
            <strong>Params:</strong>
            {Object.entries(individualsData.best_individual.params || {}).map(([k, v]) => (
              <span key={k} className="bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                {k}={typeof v === 'number' ? v.toFixed(4) : v}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Generations Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Generations ({generationsData?.total_generations || 0})
          </h3>
        </div>
        <div className="max-h-96 overflow-y-auto">
          {generationsData?.generations.map((gen) => {
            const isExpanded = expandedGenerations.has(gen.generation);
            const genIndividuals = individualsData?.individuals.filter(i => i.generation === gen.generation) || [];

            return (
              <div key={gen.generation} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                <button
                  onClick={() => toggleGeneration(gen.generation)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    <span className="font-medium">Gen {gen.generation + 1}</span>
                    <span className="text-sm text-gray-500">
                      Best: <span className="text-green-600 font-medium">{gen.best_fitness.toFixed(4)}</span>
                    </span>
                    <span className="text-sm text-gray-500">
                      Avg: {gen.avg_fitness.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {Object.entries(gen.model_types).map(([type, count]) => (
                      <span key={type} className={`px-2 py-0.5 rounded text-xs text-white ${MODEL_COLORS[type] || 'bg-gray-500'}`}>
                        {type.toUpperCase()}: {count}
                      </span>
                    ))}
                    <span className="text-sm text-gray-500">{gen.individual_count} ind.</span>
                  </div>
                </button>

                {isExpanded && genIndividuals.length > 0 && (
                  <div className="bg-gray-50 dark:bg-gray-900/50 px-4 py-2">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-500 text-xs">
                          <th className="text-left py-1 px-2">#</th>
                          <th className="text-left py-1 px-2">Model</th>
                          <th className="text-left py-1 px-2">Fitness</th>
                          <th className="text-left py-1 px-2">MAPE</th>
                          <th className="text-left py-1 px-2">Key Params</th>
                          <th className="text-right py-1 px-2">Details</th>
                        </tr>
                      </thead>
                      <tbody>
                        {genIndividuals
                          .sort((a, b) => b.fitness - a.fitness)
                          .map((ind, idx) => (
                            <tr key={idx} className={`border-t border-gray-200 dark:border-gray-700 ${idx === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''}`}>
                              <td className="py-2 px-2">{ind.individual}</td>
                              <td className="py-2 px-2">
                                <span className={`px-2 py-0.5 rounded text-xs font-medium ${MODEL_TEXT_COLORS[ind.model_type] || 'bg-gray-100'}`}>
                                  {ind.model_type.toUpperCase()}
                                </span>
                              </td>
                              <td className="py-2 px-2 font-medium">{ind.fitness.toFixed(4)}</td>
                              <td className="py-2 px-2">{(ind.metrics?.mape || 0).toFixed(2)}%</td>
                              <td className="py-2 px-2 text-xs text-gray-500">
                                {ind.params?.hidden_dim && `dim=${ind.params.hidden_dim}`}
                                {ind.params?.n_rnn_layers && ` layers=${ind.params.n_rnn_layers}`}
                                {ind.params?.learning_rate && ` lr=${Number(ind.params.learning_rate).toFixed(4)}`}
                              </td>
                              <td className="py-2 px-2 text-right">
                                <button
                                  onClick={() => setSelectedIndividual(ind)}
                                  className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                                  title="View details"
                                >
                                  <Info size={14} />
                                </button>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Training Logs */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <FileText size={16} className="text-gray-500" />
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Training Logs</h3>
          </div>
          {showLogs ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        {showLogs && (
          <div className="px-4 pb-4">
            <div className="bg-gray-900 rounded-md p-4 max-h-48 overflow-y-auto font-mono text-sm">
              {logs.length > 0 ? (
                logs.map((log, idx) => (
                  <div key={idx} className="text-green-400">{log}</div>
                ))
              ) : (
                <div className="text-gray-500">No logs yet...</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Individual Detail Modal */}
      {selectedIndividual && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setSelectedIndividual(null)}>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-lg font-semibold">
                Individual Details - {selectedIndividual.model_type.toUpperCase()}
              </h3>
              <button onClick={() => setSelectedIndividual(null)} className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                <XCircle size={20} />
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-gray-500">Generation</span>
                  <div className="font-medium">{selectedIndividual.generation + 1}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-500">Individual #</span>
                  <div className="font-medium">{selectedIndividual.individual}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-500">Fitness</span>
                  <div className="font-medium text-green-600">{selectedIndividual.fitness.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-500">MAPE</span>
                  <div className="font-medium">{(selectedIndividual.metrics?.mape || 0).toFixed(2)}%</div>
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-500 block mb-2">Parameters</span>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3 text-sm">
                  {Object.entries(selectedIndividual.params || {}).map(([k, v]) => (
                    <div key={k} className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-600 last:border-b-0">
                      <span className="text-gray-600 dark:text-gray-400">{k}</span>
                      <span className="font-mono">{typeof v === 'number' ? v.toFixed(6) : v}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-500 block mb-2">Metrics</span>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3 text-sm">
                  {Object.entries(selectedIndividual.metrics || {}).map(([k, v]) => (
                    <div key={k} className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-600 last:border-b-0">
                      <span className="text-gray-600 dark:text-gray-400">{k}</span>
                      <span className="font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                    </div>
                  ))}
                </div>
              </div>

              {selectedIndividual.training_history && selectedIndividual.training_history.length > 0 && (
                <div>
                  <span className="text-sm text-gray-500 block mb-2">Training History</span>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={selectedIndividual.training_history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="epoch" stroke="#6B7280" fontSize={12} />
                      <YAxis stroke="#6B7280" fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="loss" stroke="#EF4444" name="Loss" />
                      <Line type="monotone" dataKey="val_loss" stroke="#F97316" name="Val Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default JobDetails;
