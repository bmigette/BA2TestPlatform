import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Trash2, RefreshCw, Copy, Edit, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import DatasetWizard from '../components/DatasetWizard';
import ConfirmDialog from '../components/ConfirmDialog';
import Toast from '../components/Toast';

interface Dataset {
  id: number;
  name: string;
  ticker: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  rows_count: number;
  status: 'pending' | 'building' | 'ready' | 'error';
  error_message: string | null;
  created_at: string;
  normalization_buffer_pct?: number;
  technical_indicators?: any;
  generation_config?: any;
}

type WizardMode = 'create' | 'duplicate' | 'edit';

const Datasets: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isWizardOpen, setIsWizardOpen] = useState(false);
  const [wizardMode, setWizardMode] = useState<WizardMode>('create');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<number | null>(null);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'info' | 'warning' } | null>(null);

  const fetchDatasets = async () => {
    setIsLoading(true);
    setError(null);
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
  }, []);

  const handleDeleteClick = (id: number) => {
    setDatasetToDelete(id);
    setDeleteConfirmOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (datasetToDelete === null) return;

    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${datasetToDelete}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete dataset');
      }

      // Show success message
      setToast({
        message: 'Dataset deleted successfully',
        type: 'success',
      });

      // Refresh the list
      fetchDatasets();
    } catch (err) {
      setToast({
        message: err instanceof Error ? err.message : 'An error occurred',
        type: 'error',
      });
    } finally {
      setDatasetToDelete(null);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const handleDuplicate = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setWizardMode('duplicate');
    setIsWizardOpen(true);
  };

  const handleEdit = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setWizardMode('edit');
    setIsWizardOpen(true);
  };

  const handleCreateNew = () => {
    setSelectedDataset(null);
    setWizardMode('create');
    setIsWizardOpen(true);
  };

  const handleWizardClose = () => {
    setIsWizardOpen(false);
    setSelectedDataset(null);
    setWizardMode('create');
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Datasets</h1>
        <div className="flex space-x-2">
          <button
            onClick={fetchDatasets}
            className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-2"
          >
            <RefreshCw size={16} />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleCreateNew}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 flex items-center space-x-2"
          >
            <Plus size={16} />
            <span>Create New Dataset</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200 rounded-md">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-12">
          <p className="text-gray-600 dark:text-gray-400">Loading datasets...</p>
        </div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-gray-600 dark:text-gray-400 mb-4">No datasets yet</p>
          <button
            onClick={handleCreateNew}
            className="px-6 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 inline-flex items-center space-x-2"
          >
            <Plus size={20} />
            <span>Create Your First Dataset</span>
          </button>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Ticker
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Timeframe
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Date Range
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Rows
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {datasets.map((dataset) => (
                <tr
                  key={dataset.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                  onClick={() => navigate(`/datasets/${dataset.id}`)}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                    {dataset.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {dataset.status === 'ready' && (
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300 rounded-full inline-flex items-center gap-1">
                        <CheckCircle size={10} />
                        Ready
                      </span>
                    )}
                    {dataset.status === 'building' && (
                      <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 rounded-full inline-flex items-center gap-1">
                        <Loader size={10} className="animate-spin" />
                        Building
                      </span>
                    )}
                    {dataset.status === 'pending' && (
                      <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 rounded-full inline-flex items-center gap-1">
                        <Loader size={10} />
                        Pending
                      </span>
                    )}
                    {dataset.status === 'error' && (
                      <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300 rounded-full inline-flex items-center gap-1" title={dataset.error_message || 'Error'}>
                        <AlertCircle size={10} />
                        Error
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {dataset.ticker}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {dataset.timeframe}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {formatDate(dataset.start_date)} - {formatDate(dataset.end_date)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {dataset.rows_count.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {formatDate(dataset.created_at)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEdit(dataset);
                        }}
                        className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                        title="Edit dataset"
                      >
                        <Edit size={16} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDuplicate(dataset);
                        }}
                        className="text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-300"
                        title="Duplicate dataset"
                      >
                        <Copy size={16} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteClick(dataset.id);
                        }}
                        className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                        title="Delete dataset"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <DatasetWizard
        isOpen={isWizardOpen}
        onClose={handleWizardClose}
        onComplete={fetchDatasets}
        mode={wizardMode}
        initialData={selectedDataset}
      />

      <ConfirmDialog
        isOpen={deleteConfirmOpen}
        onClose={() => setDeleteConfirmOpen(false)}
        onConfirm={handleDeleteConfirm}
        title="Delete Dataset"
        message="Are you sure you want to delete this dataset? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
      />

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          duration={5000}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
};

export default Datasets;
