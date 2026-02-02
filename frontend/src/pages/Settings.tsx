import React, { useState, useEffect, useCallback } from 'react';
import {
  Server, Plus, Trash2, Edit2, X, RefreshCw, Cpu, HardDrive,
  Activity, Clock, Download, Upload, Power, PowerOff, AlertCircle,
  CheckCircle, Loader2, BarChart3, ChevronDown, ChevronRight
} from 'lucide-react';
import ConfirmDialog from '../components/ConfirmDialog';

interface WorkerCapabilities {
  train: boolean;
  infer: boolean;
}

interface GpuInfo {
  name: string;
  memory: number;
  count: number;
}

interface CpuInfo {
  cores: number;
  model: string;
}

interface Worker {
  id: number;
  name: string;
  url: string;
  description: string | null;
  workerType: 'local' | 'remote';
  capabilities: WorkerCapabilities;
  isEnabled: boolean;
  isLocal: boolean;
  status: 'online' | 'offline' | 'busy';
  gpuInfo: GpuInfo | null;
  cpuInfo: CpuInfo | null;
  lastHeartbeat: string | null;
  activeJobsCount: number;
  totalJobsCompleted: number;
  createdAt: string | null;
  updatedAt: string | null;
}

interface WorkerFormData {
  name: string;
  url: string;
  description: string;
  capabilities: WorkerCapabilities;
}

interface IndicatorConfig {
  type: string;
  name: string;
  timeframe?: string;
  period?: number;
  [key: string]: unknown;
}

interface IndicatorCollection {
  id: number;
  name: string;
  description: string | null;
  is_default: boolean;
  indicators: IndicatorConfig[];
  created_at: string | null;
  updated_at: string | null;
}

const API_BASE = 'http://localhost:8000/api';

const Settings: React.FC = () => {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingWorker, setEditingWorker] = useState<Worker | null>(null);
  const [healthChecking, setHealthChecking] = useState<number | null>(null);

  // Indicator Collections state
  const [indicatorCollections, setIndicatorCollections] = useState<IndicatorCollection[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(true);
  const [expandedCollections, setExpandedCollections] = useState<Set<number>>(new Set());

  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    variant: 'danger' | 'warning' | 'info';
    onConfirm: () => void;
  }>({ isOpen: false, title: '', message: '', variant: 'warning', onConfirm: () => {} });
  const [formData, setFormData] = useState<WorkerFormData>({
    name: '',
    url: '',
    description: '',
    capabilities: { train: true, infer: true }
  });

  const fetchWorkers = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/workers`);
      if (!response.ok) throw new Error('Failed to fetch workers');
      const data = await response.json();
      setWorkers(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch workers');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchIndicatorCollections = useCallback(async () => {
    try {
      setCollectionsLoading(true);
      const response = await fetch(`${API_BASE}/indicator-collections`);
      if (!response.ok) throw new Error('Failed to fetch indicator collections');
      const data = await response.json();
      setIndicatorCollections(data.collections || []);
    } catch (err) {
      console.error('Failed to fetch indicator collections:', err);
    } finally {
      setCollectionsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkers();
    fetchIndicatorCollections();
  }, [fetchWorkers, fetchIndicatorCollections]);

  const handleAddWorker = async () => {
    try {
      const response = await fetch(`${API_BASE}/workers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          url: formData.url,
          description: formData.description || null,
          workerType: 'remote',
          capabilities: formData.capabilities
        })
      });
      if (!response.ok) throw new Error('Failed to add worker');
      setShowAddModal(false);
      setFormData({ name: '', url: '', description: '', capabilities: { train: true, infer: true } });
      fetchWorkers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add worker');
    }
  };

  const handleUpdateWorker = async () => {
    if (!editingWorker) return;
    try {
      const response = await fetch(`${API_BASE}/workers/${editingWorker.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          url: formData.url,
          description: formData.description || null,
          capabilities: formData.capabilities
        })
      });
      if (!response.ok) throw new Error('Failed to update worker');
      setEditingWorker(null);
      setFormData({ name: '', url: '', description: '', capabilities: { train: true, infer: true } });
      fetchWorkers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update worker');
    }
  };

  const handleDeleteWorker = (workerId: number) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Worker',
      message: 'Are you sure you want to delete this worker?',
      variant: 'danger',
      onConfirm: async () => {
        try {
          const response = await fetch(`${API_BASE}/workers/${workerId}`, { method: 'DELETE' });
          if (!response.ok) throw new Error('Failed to delete worker');
          fetchWorkers();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to delete worker');
        }
      },
    });
  };

  const handleToggleEnabled = async (worker: Worker) => {
    try {
      const endpoint = worker.isEnabled ? 'disable' : 'enable';
      const response = await fetch(`${API_BASE}/workers/${worker.id}/${endpoint}`, { method: 'POST' });
      if (!response.ok) throw new Error(`Failed to ${endpoint} worker`);
      fetchWorkers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle worker');
    }
  };

  const handleHealthCheck = async (workerId: number) => {
    setHealthChecking(workerId);
    try {
      const response = await fetch(`${API_BASE}/workers/${workerId}/health-check`, { method: 'POST' });
      if (!response.ok) throw new Error('Health check failed');
      fetchWorkers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
    } finally {
      setHealthChecking(null);
    }
  };

  const handleExport = async () => {
    try {
      const response = await fetch(`${API_BASE}/workers/export`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to export workers');
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `workers-export-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export workers');
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const response = await fetch(`${API_BASE}/workers/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      if (!response.ok) throw new Error('Failed to import workers');
      fetchWorkers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import workers');
    }
    event.target.value = '';
  };

  const handleDeleteCollection = (collection: IndicatorCollection) => {
    if (collection.is_default) {
      setError('Cannot delete default collections');
      return;
    }
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Indicator Collection',
      message: `Are you sure you want to delete "${collection.name}"? This action cannot be undone.`,
      variant: 'danger',
      onConfirm: async () => {
        try {
          const response = await fetch(`${API_BASE}/indicator-collections/${collection.id}`, { method: 'DELETE' });
          if (!response.ok) throw new Error('Failed to delete collection');
          fetchIndicatorCollections();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to delete collection');
        }
      },
    });
  };

  const toggleCollectionExpanded = (id: number) => {
    setExpandedCollections(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const openEditModal = (worker: Worker) => {
    setEditingWorker(worker);
    setFormData({
      name: worker.name,
      url: worker.url,
      description: worker.description || '',
      capabilities: worker.capabilities
    });
  };

  const getStatusIcon = (status: string, isEnabled: boolean) => {
    if (!isEnabled) return <PowerOff className="w-4 h-4 text-gray-400" />;
    switch (status) {
      case 'online': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'busy': return <Activity className="w-4 h-4 text-yellow-500" />;
      default: return <AlertCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getStatusColor = (status: string, isEnabled: boolean) => {
    if (!isEnabled) return 'bg-gray-100 text-gray-600';
    switch (status) {
      case 'online': return 'bg-green-100 text-green-800';
      case 'busy': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-red-100 text-red-800';
    }
  };

  const formatMemory = (mb: number) => {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb} MB`;
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Configure workers, API keys, and application settings
          </p>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          {error}
          <button onClick={() => setError(null)} className="ml-auto">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Workers Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-blue-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Workers</h2>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              ({workers.filter(w => w.isEnabled).length} enabled)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchWorkers}
              className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <label className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg cursor-pointer" title="Import">
              <Upload className="w-4 h-4" />
              <input type="file" accept=".json" onChange={handleImport} className="hidden" />
            </label>
            <button
              onClick={handleExport}
              className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
              title="Export"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              <Plus className="w-4 h-4" />
              Add Worker
            </button>
          </div>
        </div>

        <div className="p-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
            </div>
          ) : workers.length === 0 ? (
            <p className="text-center text-gray-500 dark:text-gray-400 py-8">No workers configured</p>
          ) : (
            <div className="grid gap-4">
              {workers.map(worker => (
                <div
                  key={worker.id}
                  className={`p-4 border rounded-lg ${
                    worker.isLocal
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                      : 'bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <div className="mt-1">
                        {getStatusIcon(worker.status, worker.isEnabled)}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-gray-900 dark:text-gray-100">{worker.name}</h3>
                          {worker.isLocal && (
                            <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-800 rounded">
                              Local
                            </span>
                          )}
                          <span className={`px-2 py-0.5 text-xs rounded ${getStatusColor(worker.status, worker.isEnabled)}`}>
                            {worker.isEnabled ? worker.status : 'disabled'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          {worker.isLocal ? 'Running on backend host' : worker.url}
                        </p>
                        {worker.description && (
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{worker.description}</p>
                        )}

                        {/* Hardware Info */}
                        <div className="flex flex-wrap gap-4 mt-3 text-sm">
                          {worker.gpuInfo && (
                            <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                              <HardDrive className="w-4 h-4" />
                              <span>{worker.gpuInfo.name} ({formatMemory(worker.gpuInfo.memory)})</span>
                              {worker.gpuInfo.count > 1 && <span>x{worker.gpuInfo.count}</span>}
                            </div>
                          )}
                          {worker.cpuInfo && (
                            <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                              <Cpu className="w-4 h-4" />
                              <span>{worker.cpuInfo.cores} cores</span>
                            </div>
                          )}
                          <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                            <Activity className="w-4 h-4" />
                            <span>{worker.activeJobsCount} active jobs</span>
                          </div>
                          {worker.lastHeartbeat && (
                            <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                              <Clock className="w-4 h-4" />
                              <span>Last seen: {new Date(worker.lastHeartbeat).toLocaleTimeString()}</span>
                            </div>
                          )}
                        </div>

                        {/* Capabilities */}
                        <div className="flex gap-2 mt-2">
                          {worker.capabilities.train && (
                            <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">
                              Training
                            </span>
                          )}
                          {worker.capabilities.infer && (
                            <span className="px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded">
                              Inference
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => handleHealthCheck(worker.id)}
                        disabled={healthChecking === worker.id}
                        className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg disabled:opacity-50"
                        title="Health Check"
                      >
                        {healthChecking === worker.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <RefreshCw className="w-4 h-4" />
                        )}
                      </button>
                      <button
                        onClick={() => handleToggleEnabled(worker)}
                        className={`p-2 rounded-lg ${
                          worker.isEnabled ? 'text-green-600 hover:bg-green-50' : 'text-gray-400 hover:bg-gray-100'
                        }`}
                        title={worker.isEnabled ? 'Disable' : 'Enable'}
                      >
                        {worker.isEnabled ? <Power className="w-4 h-4" /> : <PowerOff className="w-4 h-4" />}
                      </button>
                      {!worker.isLocal && (
                        <>
                          <button
                            onClick={() => openEditModal(worker)}
                            className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                            title="Edit"
                          >
                            <Edit2 className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteWorker(worker.id)}
                            className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Indicator Collections Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mt-6">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Indicator Collections</h2>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              ({indicatorCollections.length} collections)
            </span>
          </div>
          <button
            onClick={fetchIndicatorCollections}
            className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4">
          {collectionsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
            </div>
          ) : indicatorCollections.length === 0 ? (
            <p className="text-center text-gray-500 dark:text-gray-400 py-8">No indicator collections found</p>
          ) : (
            <div className="space-y-2">
              {indicatorCollections.map(collection => (
                <div
                  key={collection.id}
                  className={`border rounded-lg ${
                    collection.is_default
                      ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800'
                      : 'bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <div className="p-3 flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1 cursor-pointer" onClick={() => toggleCollectionExpanded(collection.id)}>
                      {expandedCollections.has(collection.id) ? (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-gray-500" />
                      )}
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium text-gray-900 dark:text-gray-100">{collection.name}</h3>
                          {collection.is_default && (
                            <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-800 rounded">
                              Default
                            </span>
                          )}
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            ({collection.indicators.length} indicators)
                          </span>
                        </div>
                        {collection.description && (
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">{collection.description}</p>
                        )}
                      </div>
                    </div>
                    {!collection.is_default && (
                      <button
                        onClick={() => handleDeleteCollection(collection)}
                        className="p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>

                  {/* Expanded indicator list */}
                  {expandedCollections.has(collection.id) && (
                    <div className="px-4 pb-3 pt-0">
                      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 max-h-64 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="text-left text-gray-500 dark:text-gray-400">
                              <th className="pb-2 font-medium">Type</th>
                              <th className="pb-2 font-medium">Name</th>
                              <th className="pb-2 font-medium">Timeframe</th>
                              <th className="pb-2 font-medium">Parameters</th>
                            </tr>
                          </thead>
                          <tbody className="text-gray-700 dark:text-gray-300">
                            {collection.indicators.map((ind, idx) => (
                              <tr key={idx} className="border-t border-gray-200 dark:border-gray-700">
                                <td className="py-1.5 font-mono text-xs">{ind.type}</td>
                                <td className="py-1.5">{ind.name}</td>
                                <td className="py-1.5">
                                  {ind.timeframe ? (
                                    <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs">
                                      {ind.timeframe.toUpperCase()}
                                    </span>
                                  ) : (
                                    <span className="text-gray-400">-</span>
                                  )}
                                </td>
                                <td className="py-1.5 font-mono text-xs text-gray-500 dark:text-gray-400">
                                  {Object.entries(ind)
                                    .filter(([k]) => !['type', 'name', 'timeframe'].includes(k))
                                    .map(([k, v]) => `${k}=${v}`)
                                    .join(', ') || '-'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Add/Edit Worker Modal */}
      {(showAddModal || editingWorker) && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {editingWorker ? 'Edit Worker' : 'Add Worker'}
              </h3>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setEditingWorker(null);
                  setFormData({ name: '', url: '', description: '', capabilities: { train: true, infer: true } });
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  placeholder="GPU Server 1"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">URL</label>
                <input
                  type="text"
                  value={formData.url}
                  onChange={e => setFormData({ ...formData, url: e.target.value })}
                  placeholder="http://192.168.1.100:8001"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Description (optional)</label>
                <input
                  type="text"
                  value={formData.description}
                  onChange={e => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Description of this worker"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">Capabilities</label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={formData.capabilities.train}
                      onChange={e => setFormData({
                        ...formData,
                        capabilities: { ...formData.capabilities, train: e.target.checked }
                      })}
                      className="rounded border-gray-300"
                    />
                    <span>Training</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={formData.capabilities.infer}
                      onChange={e => setFormData({
                        ...formData,
                        capabilities: { ...formData.capabilities, infer: e.target.checked }
                      })}
                      className="rounded border-gray-300"
                    />
                    <span>Inference</span>
                  </label>
                </div>
              </div>
            </div>
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-2">
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setEditingWorker(null);
                  setFormData({ name: '', url: '', description: '', capabilities: { train: true, infer: true } });
                }}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={editingWorker ? handleUpdateWorker : handleAddWorker}
                disabled={!formData.name || !formData.url}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {editingWorker ? 'Save Changes' : 'Add Worker'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog(prev => ({ ...prev, isOpen: false }))}
        onConfirm={confirmDialog.onConfirm}
        title={confirmDialog.title}
        message={confirmDialog.message}
        variant={confirmDialog.variant}
        confirmText="Delete"
      />
    </div>
  );
};

export default Settings;
