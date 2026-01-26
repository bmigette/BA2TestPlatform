import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Brain,
  Cpu,
  Download,
  Trash2,
  Copy,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Target,
  Activity,
  BarChart3,
  Layers,
  Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';

interface HyperParameters {
  layers: number;
  layerSize: number;
  learningRate: number;
  activationFunction: string;
  dropout: number;
  batchSize: number;
  epochs: number;
}

interface TrainingHistory {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  sharpeRatio: number | null;
  maxDrawdown: number | null;
}

interface Model {
  id: string;
  name: string;
  modelType: string;
  datasetId: number;
  jobId: string;
  status: string;
  hyperparameters: HyperParameters;
  trainingHistory: TrainingHistory[];
  performanceMetrics: PerformanceMetrics;
  createdAt: string;
  trainedAt: string | null;
  filePath: string | null;
  fileSize: number | null;
  generations: number;
  bestGeneration: number;
  fitness: number;
}

interface Prediction {
  index: number;
  actual: number;
  predicted: number;
  error: number;
}

interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    specificity: number;
  };
}

const API_BASE = 'http://localhost:8002/api';

const ModelDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [model, setModel] = useState<Model | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'training' | 'predictions' | 'confusion'>('overview');
  const [exporting, setExporting] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchModelDetails();
  }, [id]);

  const fetchModelDetails = async () => {
    try {
      setLoading(true);

      // Fetch model details
      const modelRes = await fetch(`${API_BASE}/models/${id}`);
      if (!modelRes.ok) throw new Error('Model not found');
      const modelData = await modelRes.json();
      setModel(modelData);

      // Fetch predictions
      const predRes = await fetch(`${API_BASE}/models/${id}/predictions?limit=50`);
      if (predRes.ok) {
        const predData = await predRes.json();
        setPredictions(predData.predictions || []);
      }

      // Fetch confusion matrix
      const cmRes = await fetch(`${API_BASE}/models/${id}/confusion-matrix`);
      if (cmRes.ok) {
        const cmData = await cmRes.json();
        setConfusionMatrix(cmData);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: string) => {
    setExporting(true);
    try {
      const res = await fetch(`${API_BASE}/models/${id}/export?format=${format}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        alert(`Model exported: ${data.path}`);
      }
    } finally {
      setExporting(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this model?')) return;

    setDeleting(true);
    try {
      const res = await fetch(`${API_BASE}/models/${id}`, { method: 'DELETE' });
      if (res.ok) {
        navigate('/models');
      }
    } finally {
      setDeleting(false);
    }
  };

  const handleClone = async () => {
    try {
      const res = await fetch(`${API_BASE}/models/${id}/clone`, { method: 'POST' });
      if (res.ok) {
        const cloned = await res.json();
        navigate(`/models/${cloned.id}`);
      }
    } catch (err) {
      alert('Failed to clone model');
    }
  };

  const formatBytes = (bytes: number | null) => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center min-h-96">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="p-6">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>{error || 'Model not found'}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/models')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-500" />
              {model.name}
            </h1>
            <p className="text-sm text-gray-500">
              {model.modelType} Model - Dataset #{model.datasetId}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleClone}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg"
          >
            <Copy className="w-4 h-4" />
            Clone
          </button>
          <div className="relative group">
            <button
              disabled={exporting}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-500 text-white hover:bg-blue-600 rounded-lg disabled:opacity-50"
            >
              {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              Export
            </button>
            <div className="absolute right-0 mt-1 w-40 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
              <button
                onClick={() => handleExport('pytorch')}
                className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                PyTorch (.pt)
              </button>
              <button
                onClick={() => handleExport('onnx')}
                className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                ONNX (.onnx)
              </button>
              <button
                onClick={() => handleExport('tensorflow')}
                className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                TensorFlow (.h5)
              </button>
            </div>
          </div>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-red-500 text-white hover:bg-red-600 rounded-lg disabled:opacity-50"
          >
            {deleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
            Delete
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex gap-4">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'training', label: 'Training History', icon: TrendingUp },
            { id: 'predictions', label: 'Predictions', icon: Target },
            { id: 'confusion', label: 'Confusion Matrix', icon: BarChart3 }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Metrics */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-green-500" />
              Performance Metrics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Accuracy</p>
                <p className="text-2xl font-bold text-green-600">{(model.performanceMetrics.accuracy * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Precision</p>
                <p className="text-2xl font-bold text-blue-600">{(model.performanceMetrics.precision * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Recall</p>
                <p className="text-2xl font-bold text-purple-600">{(model.performanceMetrics.recall * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">F1 Score</p>
                <p className="text-2xl font-bold text-orange-600">{(model.performanceMetrics.f1Score * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">AUC</p>
                <p className="text-2xl font-bold text-indigo-600">{(model.performanceMetrics.auc * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Fitness</p>
                <p className="text-2xl font-bold text-teal-600">{model.fitness.toFixed(1)}</p>
              </div>
              {model.performanceMetrics.sharpeRatio && (
                <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <p className="text-sm text-gray-500">Sharpe Ratio</p>
                  <p className="text-2xl font-bold text-cyan-600">{model.performanceMetrics.sharpeRatio.toFixed(2)}</p>
                </div>
              )}
              {model.performanceMetrics.maxDrawdown && (
                <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <p className="text-sm text-gray-500">Max Drawdown</p>
                  <p className="text-2xl font-bold text-red-600">{(model.performanceMetrics.maxDrawdown * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-blue-500" />
              Hyperparameters
            </h3>
            <table className="w-full">
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                <tr>
                  <td className="py-2 text-sm text-gray-500">Layers</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.layers}</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Layer Size</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.layerSize} neurons</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Learning Rate</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.learningRate}</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Activation</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.activationFunction}</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Dropout</td>
                  <td className="py-2 text-sm font-medium text-right">{(model.hyperparameters.dropout * 100).toFixed(0)}%</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Batch Size</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.batchSize}</td>
                </tr>
                <tr>
                  <td className="py-2 text-sm text-gray-500">Epochs</td>
                  <td className="py-2 text-sm font-medium text-right">{model.hyperparameters.epochs}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Architecture Diagram */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-purple-500" />
              Architecture
            </h3>
            <div className="flex items-center justify-center gap-2 py-8">
              {/* Input Layer */}
              <div className="flex flex-col items-center">
                <div className="w-16 h-24 bg-blue-100 dark:bg-blue-900 border-2 border-blue-500 rounded-lg flex items-center justify-center">
                  <span className="text-xs font-medium text-blue-700 dark:text-blue-300">Input</span>
                </div>
                <span className="text-xs mt-1 text-gray-500">Features</span>
              </div>

              <div className="text-gray-400">→</div>

              {/* Hidden Layers */}
              {Array.from({ length: model.hyperparameters.layers }).map((_, i) => (
                <React.Fragment key={i}>
                  <div className="flex flex-col items-center">
                    <div className="w-16 h-24 bg-purple-100 dark:bg-purple-900 border-2 border-purple-500 rounded-lg flex flex-col items-center justify-center">
                      <span className="text-xs font-medium text-purple-700 dark:text-purple-300">{model.modelType}</span>
                      <span className="text-xs text-purple-600 dark:text-purple-400">{model.hyperparameters.layerSize}</span>
                    </div>
                    <span className="text-xs mt-1 text-gray-500">Layer {i + 1}</span>
                  </div>
                  <div className="text-gray-400">→</div>
                </React.Fragment>
              ))}

              {/* Output Layer */}
              <div className="flex flex-col items-center">
                <div className="w-16 h-24 bg-green-100 dark:bg-green-900 border-2 border-green-500 rounded-lg flex items-center justify-center">
                  <span className="text-xs font-medium text-green-700 dark:text-green-300">Output</span>
                </div>
                <span className="text-xs mt-1 text-gray-500">Prediction</span>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              Model Info
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Status</span>
                <span className={`flex items-center gap-1 text-sm font-medium ${
                  model.status === 'trained' ? 'text-green-600' : 'text-gray-600'
                }`}>
                  {model.status === 'trained' && <CheckCircle className="w-4 h-4" />}
                  {model.status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Created</span>
                <span className="text-sm font-medium">{new Date(model.createdAt).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Trained</span>
                <span className="text-sm font-medium">{model.trainedAt ? new Date(model.trainedAt).toLocaleString() : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Generations</span>
                <span className="text-sm font-medium">{model.generations} (best: #{model.bestGeneration})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">File Size</span>
                <span className="text-sm font-medium">{formatBytes(model.fileSize)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Job ID</span>
                <span className="text-sm font-mono text-blue-600">{model.jobId}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'training' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Training History</h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={model.trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom' }} />
                <YAxis yAxisId="loss" label={{ value: 'Loss', angle: -90, position: 'left' }} />
                <YAxis yAxisId="accuracy" orientation="right" label={{ value: 'Accuracy', angle: 90, position: 'right' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#ef4444" name="Train Loss" />
                <Line yAxisId="loss" type="monotone" dataKey="valLoss" stroke="#f97316" name="Val Loss" strokeDasharray="5 5" />
                <Line yAxisId="accuracy" type="monotone" dataKey="accuracy" stroke="#22c55e" name="Train Accuracy" />
                <Line yAxisId="accuracy" type="monotone" dataKey="valAccuracy" stroke="#3b82f6" name="Val Accuracy" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === 'predictions' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Prediction Visualization</h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="actual" name="Actual" label={{ value: 'Actual Value', position: 'bottom' }} />
                <YAxis dataKey="predicted" name="Predicted" label={{ value: 'Predicted Value', angle: -90, position: 'left' }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={predictions} fill="#8884d8">
                  {predictions.map((entry, index) => (
                    <Cell key={index} fill={entry.error < 3 ? '#22c55e' : entry.error < 5 ? '#f97316' : '#ef4444'} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 flex justify-center gap-6 text-sm">
            <span className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" /> Low error (&lt;3)
            </span>
            <span className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" /> Medium error (3-5)
            </span>
            <span className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" /> High error (&gt;5)
            </span>
          </div>
        </div>
      )}

      {activeTab === 'confusion' && confusionMatrix && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
          <div className="flex gap-8">
            {/* Matrix */}
            <div className="flex-1">
              <div className="flex items-center justify-center">
                <table className="border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      <th className="p-2"></th>
                      <th colSpan={2} className="p-2 text-center text-sm font-medium text-gray-600">Predicted</th>
                    </tr>
                    <tr>
                      <th className="p-2"></th>
                      <th className="p-2"></th>
                      {confusionMatrix.labels.map(label => (
                        <th key={label} className="p-2 text-sm font-medium text-gray-600 w-24 text-center">{label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrix.matrix.map((row, i) => (
                      <tr key={i}>
                        {i === 0 && (
                          <th rowSpan={2} className="p-2 text-sm font-medium text-gray-600 writing-mode-vertical" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
                            Actual
                          </th>
                        )}
                        <th className="p-2 text-sm font-medium text-gray-600">{confusionMatrix.labels[i]}</th>
                        {row.map((val, j) => (
                          <td
                            key={j}
                            className={`p-4 text-center text-lg font-bold w-24 h-24 ${
                              i === j ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                              'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                            }`}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Metrics */}
            <div className="w-64 space-y-3">
              <h4 className="font-medium">Classification Metrics</h4>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Accuracy</p>
                <p className="text-xl font-bold">{(confusionMatrix.metrics.accuracy * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Precision</p>
                <p className="text-xl font-bold">{(confusionMatrix.metrics.precision * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Recall (Sensitivity)</p>
                <p className="text-xl font-bold">{(confusionMatrix.metrics.recall * 100).toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500">Specificity</p>
                <p className="text-xl font-bold">{(confusionMatrix.metrics.specificity * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelDetails;
