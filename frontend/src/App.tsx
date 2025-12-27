import { FormEvent, useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  FileText,
  LayoutDashboard,
  Loader2,
  LogOut,
  Moon,
  Settings,
  Sun,
  Terminal,
  UploadCloud,
  Zap,
} from 'lucide-react';
import {
  ApiClient,
  AnalysisResult,
  AnalysisStatus,
  AnalysisListItem,
  AvailableModel,
  Dataset,
  DatasetPreview,
  TraceStep,
  User as UserType,
} from './api/client';
import { useTheme } from './hooks/useTheme';

const resolveDefaultApi = () => {
  const envBase = import.meta.env.VITE_API_BASE;
  if (envBase) return envBase;
  if (typeof window === 'undefined') return 'http://localhost:8000/api/v1';
  const { protocol, hostname, port } = window.location;
  if (hostname.endsWith('.app.github.dev')) {
    return `${protocol}//${hostname.replace(/-\d+\.app\.github\.dev$/, '-8000.app.github.dev')}/api/v1`;
  }
  return port ? `${protocol}//${hostname}:8000/api/v1` : `${protocol}//${hostname}:8000/api/v1`;
};

const defaultApi = resolveDefaultApi();

type Tab = 'datasets' | 'analysis' | 'logs';

type NavItemProps = {
  icon: JSX.Element;
  label: string;
  active: boolean;
  isOpen: boolean;
  onClick: () => void;
  theme: 'light' | 'dark';
};

type StatRow = { name: string; pValue?: number; effectSize?: number };

const formatTimestamp = (value?: string) => (value ? new Date(value).toLocaleTimeString() : '');

const mergeUniqueSteps = (existing: TraceStep[], next: TraceStep[]) => {
  const seen = new Set(existing.map((s) => `${s.timestamp || ''}-${s.step}-${s.detail || ''}`));
  const merged = [...existing];
  next.forEach((step) => {
    const key = `${step.timestamp || ''}-${step.step}-${step.detail || ''}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(step);
    }
  });
  return merged;
};

function App() {
  // Theming + layout
  const [theme, toggleTheme] = useTheme();
  const [activeTab, setActiveTab] = useState<Tab>('datasets');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isRightPanelOpen, setRightPanelOpen] = useState(true);

  // Core app state
  const [apiBase, setApiBase] = useState(defaultApi);
  const [token, setToken] = useState<string | undefined>();
  const [user, setUser] = useState<UserType | undefined>();
  const [health, setHealth] = useState('');
  const [error, setError] = useState('');

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [preview, setPreview] = useState<DatasetPreview | undefined>();
  const [datasetNotes, setDatasetNotes] = useState('');

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [registering, setRegistering] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [description, setDescription] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const [analysisId, setAnalysisId] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | undefined>();
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | undefined>();
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisListItem[]>([]);
  const [logContent, setLogContent] = useState('');
  const [streamSteps, setStreamSteps] = useState<TraceStep[]>([]);
  const [streaming, setStreaming] = useState(false);
  const streamAbortRef = useRef<AbortController | null>(null);

  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [modelName, setModelName] = useState('gpt-4o');
  const [provider, setProvider] = useState('openai');
  const [exporting, setExporting] = useState<'pdf' | 'docx' | 'csv' | null>(null);
  const [commentDraft, setCommentDraft] = useState('');
  const [renameDrafts, setRenameDrafts] = useState<Record<string, string>>({});
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [configuredProviders, setConfiguredProviders] = useState<string[]>([]);
  const [credentialInputs, setCredentialInputs] = useState({
    openai: '',
    anthropic: '',
    groq: '',
    gemini: '',
    google: '',
    ollama_base_url: '',
    ollama_default_model: '',
  });
  const [showKeys, setShowKeys] = useState(false);
  const [overwriteLatest, setOverwriteLatest] = useState(false);

  const api = useMemo(() => new ApiClient(apiBase, token), [apiBase, token]);

  const clearMessages = () => {
    setError('');
    setHealth('');
  };

  const connect = async () => {
    clearMessages();
    try {
      const info = await api.health();
      setHealth(info.version ? `API online (v${info.version})` : 'API online');
    } catch (e) {
      setError((e as Error).message || 'Unable to reach API');
    }
  };

  const handleLogin = async (evt: FormEvent) => {
    evt.preventDefault();
    clearMessages();
    try {
      const res = await api.login(email, password);
      setToken(res.access_token);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleRegister = async (evt: FormEvent) => {
    evt.preventDefault();
    clearMessages();
    try {
      await api.register(email, password);
      setRegistering(false);
      setHealth('Account created. Sign in to continue.');
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadDatasets = async () => {
    if (!token) return;
    try {
      const list = await api.datasets();
      setDatasets(list);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadModelMeta = async () => {
    try {
      const res = await api.availableModels();
      setAvailableModels(res.models || []);
      if (res.default_model) setModelName((prev) => prev || res.default_model);
      if (res.default_provider) setProvider((prev) => prev || res.default_provider);
    } catch (e) {
      // silent; model listing may be unavailable without credentials
      console.warn(e);
    }
  };

  const loadCredentialMeta = async () => {
    try {
      const res = await api.configuredCredentials();
      setConfiguredProviders(res.configured_providers || []);
      if (res.stored_credentials) {
        setCredentialInputs((prev) => ({
          ...prev,
          openai: res.stored_credentials.openai || prev.openai,
          anthropic: res.stored_credentials.anthropic || prev.anthropic,
          groq: res.stored_credentials.groq || prev.groq,
          google: res.stored_credentials.google || res.stored_credentials.gemini || prev.google,
          gemini: res.stored_credentials.gemini || res.stored_credentials.google || prev.gemini,
        }));
      }
    } catch (e) {
      console.warn(e);
    }
  };

  const openAnalysis = useCallback(async (analysisItem: AnalysisListItem) => {
    setAnalysisResults(undefined);
    setAnalysisStatus(undefined);
    setStreamSteps([]);
    setLogContent('');
    setAnalysisId(analysisItem.id);
    setStreaming(analysisItem.status === 'running' || analysisItem.status === 'pending');
    try {
      const status = await api.analysisStatus(analysisItem.id);
      setAnalysisStatus(status);
      if (status.comment !== undefined) {
        setCommentDraft(status.comment || '');
      }
      if (status.status === 'completed') {
        const resJson = await api.analysisResults(analysisItem.id);
        setAnalysisResults(resJson);
        setCommentDraft(resJson.comment || '');
        setStreaming(false);
      }
    } catch (e) {
      setError((e as Error).message);
      setStreaming(false);
    }
  }, [api]);

  const loadAnalyses = useCallback(async (datasetId: string, autoSelect = true) => {
    if (!datasetId) return;
    try {
      const list = await api.analyses({ dataset_id: datasetId });
      const sorted = [...list].sort((a, b) => (b.version || 0) - (a.version || 0));
      setAnalysisHistory(sorted);
      if (autoSelect && sorted.length) {
        const latestCompleted = sorted.find((item) => item.status === 'completed') || sorted[0];
        await openAnalysis(latestCompleted);
      } else if (!sorted.length) {
        setAnalysisId('');
        setAnalysisStatus(undefined);
        setAnalysisResults(undefined);
        setCommentDraft('');
      }
    } catch (e) {
      // Ignore not-found/empty responses when no analyses exist yet
      const msg = (e as Error).message || '';
      if (!msg.toLowerCase().includes('not found')) {
        setError(msg);
      } else {
        setAnalysisHistory([]);
      }
    }
  }, [api, openAnalysis]);

  const selectDataset = async (id: string) => {
    setSelectedDatasetId(id);
    setAnalysisResults(undefined);
    setAnalysisStatus(undefined);
    setAnalysisId('');
    setLogContent('');
    setAnalysisHistory([]);
    setCommentDraft('');
    setDatasetNotes('');
    if (!id) return;
    try {
      const p = await api.previewDataset(id);
      setPreview(p);
      setDatasetNotes(p.description || '');
      setSelectedColumns([]);
      setRenameDrafts({});
      await loadAnalyses(id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    clearMessages();
    try {
      const result = await api.uploadDataset(file, description || undefined);
      setDescription('');
      setFile(null);
      await loadDatasets();
      await selectDataset(result.dataset_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleRunAnalysis = async (forceOverwrite?: boolean) => {
    if (!selectedDatasetId) return;
    clearMessages();
    try {
      const overwriteFlag = typeof forceOverwrite === 'boolean' ? forceOverwrite : overwriteLatest;
      const { id } = await api.runAnalysis({
        dataset_id: selectedDatasetId,
        selected_columns: selectedColumns.length ? selectedColumns : undefined,
        model_name: modelName,
        provider,
        overwrite: overwriteFlag,
      });
      streamAbortRef.current?.abort();
      setAnalysisId(id);
      setStreamSteps([]);
      setStreaming(true);
      setAnalysisResults(undefined);
      setAnalysisStatus({ id, status: 'running' });
      setCommentDraft('');
      await loadAnalyses(selectedDatasetId, false);
      setActiveTab('analysis');
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleDeleteAnalysis = async (id: string) => {
    try {
      await api.deleteAnalysis(id);
      await loadAnalyses(selectedDatasetId, false);
      if (analysisId === id) {
        setAnalysisId('');
        setAnalysisResults(undefined);
        setAnalysisStatus(undefined);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleExport = async (format: 'pdf' | 'docx' | 'csv') => {
    const id = analysisId || analysisResults?.id;
    if (!id) return;
    try {
      setExporting(format);
      const blob = await api.exportAnalysis(id, format);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analysis-${id}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setExporting(null);
    }
  };

  const loadLog = async () => {
    if (!analysisId) return;
    try {
      const log = await api.analysisLog(analysisId);
      setLogContent(log.log_content);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const applyRenames = async () => {
    if (!selectedDatasetId) return;
    const mapping = Object.fromEntries(
      Object.entries(renameDrafts).filter(([col, next]) => next && next !== col)
    );
    if (!Object.keys(mapping).length) return;
    try {
      const updated = await api.renameColumns(selectedDatasetId, mapping);
      setPreview(updated);
      setSelectedColumns((cols) =>
        cols
          .map((col) => mapping[col] || col)
          .filter((col) => updated.column_names.includes(col))
      );
      setRenameDrafts({});
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const saveCredentials = async () => {
    try {
      const res = await api.setCredentials({
        openai_api_key: credentialInputs.openai,
        anthropic_api_key: credentialInputs.anthropic,
        google_api_key: credentialInputs.google || credentialInputs.gemini,
        gemini_api_key: credentialInputs.gemini || credentialInputs.google,
        groq_api_key: credentialInputs.groq,
        ollama_enabled: credentialInputs.ollama_base_url ? 'true' : undefined,
        ollama_base_url: credentialInputs.ollama_base_url || undefined,
        ollama_default_model: credentialInputs.ollama_default_model || undefined,
      });
      setConfiguredProviders(res.configured_providers || []);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // Effects
  useEffect(() => {
    const storedToken = localStorage.getItem('statmate-token');
    if (storedToken) setToken(storedToken);
  }, []);

  useEffect(() => {
    const storedProvider = localStorage.getItem('statmate-provider');
    const storedModel = localStorage.getItem('statmate-model');
    if (storedProvider) setProvider(storedProvider);
    if (storedModel) setModelName(storedModel);
  }, []);

  useEffect(() => {
    if (provider) localStorage.setItem('statmate-provider', provider);
    if (modelName) localStorage.setItem('statmate-model', modelName);
  }, [provider, modelName]);

  useEffect(() => {
    if (token) {
      localStorage.setItem('statmate-token', token);
      api
        .withToken(token)
        .me()
        .then(setUser)
        .catch(() => setToken(undefined));
      loadDatasets();
      connect();
      loadModelMeta();
      loadCredentialMeta();
    } else {
      localStorage.removeItem('statmate-token');
      setUser(undefined);
      setDatasets([]);
      setPreview(undefined);
      setDatasetNotes('');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    const shouldStream = streaming || analysisStatus?.status === 'running' || analysisStatus?.status === 'pending';
    if (!analysisId || !shouldStream) return undefined;
    const controller = new AbortController();
    streamAbortRef.current = controller;

    const connectStream = async () => {
      try {
        const res = await api.analysisStream(analysisId, controller.signal);
        if (!res.body) return;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || '';

          for (const raw of parts) {
            if (!raw.trim()) continue;
            const lines = raw.split('\n');
            const eventLine = lines.find((l) => l.startsWith('event:'));
            const dataLine = lines.find((l) => l.startsWith('data:'));
            const event = eventLine?.replace('event:', '').trim();
            const data = dataLine ? JSON.parse(dataLine.replace('data:', '').trim()) : {};

            if (event === 'step') {
              setStreamSteps((prev) => mergeUniqueSteps(prev, [data as TraceStep]));
            }
            if (event === 'log') {
              setLogContent((prev) => `${prev}${data.chunk || ''}`);
            }
            if (event === 'done') {
              setStreaming(false);
              const resJson = await api.analysisResults(analysisId);
              setAnalysisResults(resJson);
              setAnalysisStatus((prev) => ({ ...(prev || { id: analysisId, status: 'completed' }), status: 'completed' }));
              if (selectedDatasetId) {
                await loadAnalyses(selectedDatasetId, false);
              }
            }
          }
        }
      } catch (e) {
        if (!controller.signal.aborted) {
          setStreaming(false);
          setError((e as Error).message);
        }
      }
    };

    connectStream();
    return () => controller.abort();
  }, [analysisId, api, streaming, analysisStatus, selectedDatasetId, loadAnalyses]);

  useEffect(() => {
    if (!analysisId) return undefined;
    const interval = setInterval(async () => {
      try {
        const status = await api.analysisStatus(analysisId);
        setAnalysisStatus(status);
        if (status.comment !== undefined) {
          setCommentDraft(status.comment || '');
        }
        setAnalysisHistory((prev) =>
          prev.map((item) =>
            item.id === analysisId
              ? {
                  ...item,
                  status: status.status,
                  version: status.version || item.version,
                  superseded_at: status.superseded_at || item.superseded_at,
                  comment: status.comment ?? item.comment,
                }
              : item
          )
        );
        if (status.decision_steps?.length) {
          setStreamSteps((prev) => mergeUniqueSteps(prev, status.decision_steps || []));
        }
        if (status.intermediate_log) {
          setLogContent(status.intermediate_log);
        }
        if (status.status === 'completed' && !analysisResults) {
          const resJson = await api.analysisResults(analysisId);
          setAnalysisResults(resJson);
          setStreaming(false);
        }
      } catch (e) {
        setError((e as Error).message);
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [analysisId, api, analysisResults]);

  // Derived data
  const trace = useMemo(() => {
    const fromResults = analysisResults?.decision_steps || analysisResults?.execution_trace || analysisResults?.results_detail?.execution_trace || [];
    const fromStatus = analysisStatus?.decision_steps || analysisStatus?.execution_trace || [];
    return mergeUniqueSteps([], [...streamSteps, ...fromStatus, ...fromResults]);
  }, [analysisResults, analysisStatus, streamSteps]);

  const plots = useMemo(() => analysisResults?.plots || analysisResults?.results_detail?.plots || [], [analysisResults]);

  const effectSizes = useMemo(() => {
    if (!analysisResults) return {} as Record<string, number>;
    return analysisResults.effect_sizes || analysisResults.results_detail?.effect_sizes || {};
  }, [analysisResults]);

  const statsRows: StatRow[] = useMemo(() => {
    if (!analysisResults) return [];
    const probabilities = analysisResults.probabilities || analysisResults.results_detail?.probabilities || {};
    const keys = new Set([...Object.keys(probabilities), ...Object.keys(effectSizes)]);
    return Array.from(keys).map((key) => ({ name: key, pValue: probabilities[key], effectSize: effectSizes[key] }));
  }, [analysisResults, effectSizes]);

  const summaryText =
    analysisResults?.summary || analysisResults?.results_detail?.summary || (streaming ? 'Generating summary…' : 'Waiting for results');

  const providerOptions = useMemo(() => {
    if (availableModels.length) {
      return Array.from(new Set(availableModels.map((m) => m.provider)));
    }
    return ['openai', 'anthropic', 'google', 'groq', 'ollama'];
  }, [availableModels]);

  const connectionLabel = health || (error && !token ? error : 'API status unknown');

  useEffect(() => {
    setCommentDraft(analysisResults?.comment || '');
  }, [analysisResults?.id]);

  useEffect(() => {
    const currentId = analysisResults?.id || analysisId;
    if (!currentId) return undefined;
    const handler = setTimeout(async () => {
      try {
        await api.updateAnalysisComment(currentId, commentDraft);
        setAnalysisHistory((prev) => prev.map((item) => (item.id === currentId ? { ...item, comment: commentDraft } : item)));
      } catch (e) {
        setError((e as Error).message);
      }
    }, 700);
    return () => clearTimeout(handler);
  }, [analysisResults?.id, analysisId, commentDraft, api]);

  useEffect(() => {
    if (!selectedDatasetId || !preview || preview.dataset_id !== selectedDatasetId) return undefined;
    const currentPreviewDescription = preview.description || '';
    if (datasetNotes === currentPreviewDescription) return undefined;
    const handler = setTimeout(async () => {
      try {
        await api.updateDatasetDescription(selectedDatasetId, datasetNotes || null);
        setPreview((prev) =>
          prev && prev.dataset_id === selectedDatasetId ? { ...prev, description: datasetNotes } : prev
        );
        setDatasets((prev) => prev.map((d) => (d.id === selectedDatasetId ? { ...d, description: datasetNotes } : d)));
      } catch (e) {
        setError((e as Error).message);
      }
    }, 700);
    return () => clearTimeout(handler);
  }, [selectedDatasetId, preview?.dataset_id, datasetNotes, api]);

  // Render: Auth
  if (!token) {
    return (
      <div className={`relative min-h-screen overflow-hidden ${theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
        <div className="pointer-events-none absolute inset-0 opacity-80">
          <div className="absolute -left-10 top-10 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />
          <div className="absolute -right-10 top-32 h-72 w-72 rounded-full bg-indigo-500/10 blur-3xl" />
        </div>
        <div className="relative mx-auto flex min-h-screen max-w-6xl items-center justify-center px-6 py-12">
          <div className="grid w-full grid-cols-1 gap-8 lg:grid-cols-2">
            <div className={`rounded-3xl border p-10 shadow-2xl backdrop-blur ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/90'}`}>
              <div className="mb-10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600 text-2xl font-black text-slate-950 shadow-glow">
                    Σ
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-cyan-400">Statmate AI</p>
                    <h1 className="text-2xl font-bold">Secure workspace</h1>
                  </div>
                </div>
                <button
                  onClick={toggleTheme}
                  className={`rounded-full border p-3 transition-colors ${theme === 'dark' ? 'border-slate-800 bg-slate-900 hover:bg-slate-800' : 'border-slate-200 bg-white hover:bg-slate-100'}`}
                  aria-label="Toggle theme"
                >
                  {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                </button>
              </div>

              <div className="mb-6 flex items-center gap-3 rounded-2xl border px-4 py-3 text-sm shadow-sm backdrop-blur-sm">
                <span className={`h-2 w-2 rounded-full ${health ? 'bg-emerald-400 shadow-[0_0_10px_rgba(74,222,128,0.7)]' : 'bg-amber-400 animate-pulse'}`} />
                <div className="flex flex-col">
                  <span className="text-xs uppercase tracking-[0.18em] text-slate-400">API</span>
                  <span className="font-medium text-slate-200">{connectionLabel}</span>
                </div>
                <button onClick={connect} className="ml-auto rounded-full border px-3 py-1 text-xs font-semibold text-cyan-400">
                  Check
                </button>
              </div>

              {error && <div className="mb-4 rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-100">{error}</div>}
              {health && <div className="mb-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-100">{health}</div>}

              <form onSubmit={registering ? handleRegister : handleLogin} className="space-y-5">
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className={`w-full rounded-xl border px-4 py-3 outline-none transition ${theme === 'dark' ? 'border-slate-800 bg-slate-900/80 focus:border-cyan-500' : 'border-slate-200 bg-white focus:border-cyan-500'}`}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className={`w-full rounded-xl border px-4 py-3 outline-none transition ${theme === 'dark' ? 'border-slate-800 bg-slate-900/80 focus:border-cyan-500' : 'border-slate-200 bg-white focus:border-cyan-500'}`}
                  />
                </div>
                <button
                  type="submit"
                  className="w-full rounded-xl bg-gradient-to-r from-cyan-400 to-blue-600 py-3 font-bold text-slate-950 shadow-lg shadow-cyan-500/25 transition hover:translate-y-[1px] active:translate-y-[2px]"
                >
                  {registering ? 'Create account' : 'Sign in'}
                </button>
                <button
                  type="button"
                  onClick={() => setRegistering((v) => !v)}
                  className="w-full text-sm font-semibold text-cyan-300 hover:text-white"
                >
                  {registering ? 'Have an account? Sign in' : 'Need an account? Register'}
                </button>
              </form>
            </div>

            <div
              className={`flex flex-col justify-between gap-6 rounded-3xl border p-10 shadow-2xl backdrop-blur ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/90'}`}
            >
              <div>
                <div className="mb-6 flex items-center gap-3">
                  <div className="rounded-full bg-cyan-500/10 p-3 text-cyan-300">
                    <Zap size={18} />
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Workspace</p>
                    <h2 className="text-xl font-bold">Live analytics studio</h2>
                  </div>
                </div>
                <div className="space-y-3 text-sm text-slate-400">
                  <div className="flex items-start gap-3 rounded-2xl border border-slate-800/50 bg-slate-900/30 p-3 shadow-inner">
                    <CheckCircle2 size={16} className="mt-0.5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-slate-200">Upload smart datasets</p>
                      <p className="text-slate-400">CSV, TSV, Parquet, Excel, and docs—automatically profiled.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 rounded-2xl border border-slate-800/50 bg-slate-900/30 p-3 shadow-inner">
                    <Zap size={16} className="mt-0.5 text-cyan-400" />
                    <div>
                      <p className="font-semibold text-slate-200">Streaming analysis</p>
                      <p className="text-slate-400">Watch the agent trace, summary, plots, and export PDFs/Word.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-2xl border border-slate-800/40 bg-slate-900/30 p-4">
                <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">API base</label>
                <input
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  className={`w-full rounded-xl border px-4 py-3 text-sm outline-none transition ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70 focus:border-cyan-500' : 'border-slate-200 bg-white focus:border-cyan-500'}`}
                  placeholder="http://localhost:8000/api/v1"
                />
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>Environment: {import.meta.env.MODE}</span>
                  <span>Theme: {theme}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render: Main app
  return (
    <div className={`relative flex min-h-screen ${theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="absolute left-20 top-10 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute right-10 top-40 h-72 w-72 rounded-full bg-indigo-500/10 blur-3xl" />
      </div>

      {/* Sidebar */}
      <aside
        className={`relative z-10 flex-shrink-0 border-r backdrop-blur transition-all duration-300 ${
          isSidebarOpen ? 'w-64' : 'w-20'
        } ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/70'}`}
      >
        <div className="flex h-16 items-center gap-3 px-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 text-lg font-extrabold text-slate-950 shadow-glow">
            Σ
          </div>
          {isSidebarOpen && (
            <div>
              <div className="text-sm uppercase tracking-[0.2em] text-slate-500">Statmate</div>
              <div className="text-lg font-bold">AI Studio</div>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className={`ml-auto rounded-lg border p-2 text-slate-500 transition ${
              theme === 'dark' ? 'border-slate-800 hover:bg-slate-800' : 'border-slate-200 hover:bg-slate-100'
            }`}
            aria-label="Toggle sidebar"
          >
            <LayoutDashboard size={16} />
          </button>
        </div>

        <nav className="mt-4 space-y-2 px-2">
          <NavItem
            icon={<Database size={18} />}
            label="Datasets"
            active={activeTab === 'datasets'}
            isOpen={isSidebarOpen}
            onClick={() => setActiveTab('datasets')}
            theme={theme}
          />
          <NavItem
            icon={<Activity size={18} />}
            label="Analysis"
            active={activeTab === 'analysis'}
            isOpen={isSidebarOpen}
            onClick={() => setActiveTab('analysis')}
            theme={theme}
          />
          <NavItem
            icon={<Terminal size={18} />}
            label="Logs"
            active={activeTab === 'logs'}
            isOpen={isSidebarOpen}
            onClick={() => setActiveTab('logs')}
            theme={theme}
          />
        </nav>

        <div className="absolute bottom-4 w-full px-3">
          <div className={`mb-2 rounded-xl border px-3 py-2 text-xs ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/70'}`}>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${health ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
              <span className="font-semibold">{connectionLabel}</span>
            </div>
            <p className="mt-1 text-[11px] text-slate-500">{apiBase}</p>
          </div>
          <button
            onClick={toggleTheme}
            className={`flex w-full items-center gap-3 rounded-xl p-3 text-sm font-semibold transition ${
              theme === 'dark' ? 'hover:bg-slate-800' : 'hover:bg-slate-100'
            }`}
          >
            {theme === 'dark' ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} className="text-indigo-600" />}
            {isSidebarOpen && <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
          </button>
        </div>
      </aside>

      {/* Main area */}
      <main className="relative z-0 flex flex-1 flex-col overflow-hidden">
        <header
          className={`flex h-16 items-center justify-between border-b px-6 backdrop-blur ${
            theme === 'dark' ? 'border-slate-800 bg-slate-950/70' : 'border-slate-200 bg-white/70'
          }`}
        >
          <div className="flex items-center gap-3">
            <span className={`h-2 w-2 rounded-full ${streaming ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Workspace</p>
              <p className="text-sm font-semibold text-slate-200">{analysisResults?.dataset_name || preview?.original_filename || 'Ready'}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-sm md:flex ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/70'}`}>
              <span className="text-slate-400">API</span>
              <input
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                className={`w-48 bg-transparent text-sm outline-none ${theme === 'dark' ? 'text-slate-200' : 'text-slate-700'}`}
              />
              <button onClick={connect} className="rounded-full border px-2 py-1 text-[11px] font-semibold text-cyan-400">
                Ping
              </button>
            </div>
            <div className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-sm md:flex ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white/70'}`}>
              <span className="text-slate-400">{user?.email}</span>
            </div>
            <button
              onClick={() => setToken(undefined)}
              className={`rounded-full border p-2 transition ${theme === 'dark' ? 'border-slate-800 hover:bg-slate-800' : 'border-slate-200 hover:bg-slate-100'}`}
              aria-label="Log out"
            >
              <LogOut size={16} />
            </button>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6">
            {error && <div className="mb-4 rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-100">{error}</div>}
            {health && <div className="mb-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-100">{health}</div>}

            {activeTab === 'datasets' && (
              <div className="space-y-8">
                <div className="flex flex-col gap-3">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Library</p>
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-3xl font-bold tracking-tight">Dataset library</h2>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={loadDatasets}
                        className="rounded-full border border-slate-800/70 px-3 py-1 text-xs font-semibold text-cyan-400"
                      >
                        Refresh
                      </button>
                      <button
                        onClick={() => setActiveTab('analysis')}
                        className="flex items-center gap-2 rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-100"
                      >
                        <BarChart3 size={14} />
                        Go to analysis
                      </button>
                    </div>
                  </div>
                  <p className="text-slate-400">Upload datasets, review previews, and pick columns before running the agent.</p>
                </div>

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
                  <div className={`rounded-2xl border p-6 shadow-lg ${theme === 'dark' ? 'border-slate-800/80 bg-slate-900/80' : 'border-slate-200 bg-white'}`}>
                    <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <UploadCloud size={16} className="text-cyan-400" /> Upload
                    </h3>
                    <div className="space-y-3 text-sm">
                      <label className="block cursor-pointer rounded-xl border border-dashed border-slate-700/80 bg-slate-900/40 p-3 text-center text-slate-400 hover:border-cyan-500">
                        <input
                          type="file"
                          className="hidden"
                          onChange={(e) => setFile(e.target.files?.[0] || null)}
                          accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet,.md,.doc,.docx"
                        />
                        {file ? file.name : 'Select a CSV, TSV, Excel, JSON, or Parquet file'}
                      </label>
                      <textarea
                        placeholder="Optional notes for the agent"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        className={`w-full rounded-xl border px-3 py-2 text-sm outline-none ${
                          theme === 'dark' ? 'border-slate-800 bg-slate-900/70 focus:border-cyan-500' : 'border-slate-200 bg-white focus:border-cyan-500'
                        }`}
                        rows={3}
                      />
                      <button
                        disabled={!file || uploading}
                        onClick={handleUpload}
                        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-400 to-blue-600 px-4 py-3 text-sm font-bold text-slate-950 shadow-lg shadow-cyan-500/25 disabled:opacity-60"
                      >
                        {uploading ? <Loader2 className="animate-spin" size={16} /> : <UploadCloud size={16} />}
                        {uploading ? 'Uploading…' : 'Process file'}
                      </button>
                    </div>
                  </div>

                  <div className={`xl:col-span-2 rounded-2xl border p-6 shadow-lg ${theme === 'dark' ? 'border-slate-800/80 bg-slate-900/80' : 'border-slate-200 bg-white'}`}>
                    <div className="mb-4 flex items-center justify-between gap-2">
                      <h3 className="text-lg font-semibold">Available datasets</h3>
                      <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">{datasets.length} files</span>
                    </div>
                    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                      {datasets.map((ds) => {
                        const active = selectedDatasetId === ds.id;
                        return (
                          <button
                            key={ds.id}
                            onClick={() => selectDataset(ds.id)}
                            className={`group flex w-full flex-col rounded-2xl border p-4 text-left transition ${
                              active
                                ? 'border-cyan-500/70 bg-cyan-500/10 shadow-lg shadow-cyan-500/10'
                                : theme === 'dark'
                                  ? 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                                  : 'border-slate-200 bg-white hover:border-slate-300'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-sm font-semibold text-slate-100">{ds.original_filename}</span>
                              {active && <CheckCircle2 size={16} className="text-cyan-400" />}
                            </div>
                            <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
                              <span>{ds.row_count ? `${ds.row_count} rows` : 'Row count pending'}</span>
                              <span>•</span>
                              <span>{ds.id.slice(0, 6)}…</span>
                            </div>
                          </button>
                        );
                      })}
                      {!datasets.length && (
                        <div className="flex min-h-[120px] items-center justify-center rounded-2xl border border-dashed border-slate-800/70 text-sm text-slate-500">
                          Upload a dataset to get started.
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {preview && (
                  <div className={`rounded-2xl border shadow-xl ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                    <div className="flex items-center justify-between border-b border-slate-800/60 px-6 py-4">
                      <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Data preview</p>
                        <h3 className="text-lg font-semibold">{preview.original_filename}</h3>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-2 rounded-full border border-slate-800/50 px-3 py-1 text-xs text-slate-400">
                          <Settings size={14} />
                          <select
                            value={provider}
                            onChange={(e) => setProvider(e.target.value)}
                            className="bg-transparent text-sm outline-none"
                          >
                            {providerOptions.map((p) => (
                              <option key={p} value={p}>
                                {p.charAt(0).toUpperCase() + p.slice(1)}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="flex items-center gap-2 rounded-full border border-slate-800/50 px-3 py-1 text-xs text-slate-400">
                          <input
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            className="bg-transparent text-sm outline-none"
                            placeholder="Model name"
                          />
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleRunAnalysis(false)}
                            className="flex items-center gap-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-600 px-4 py-2 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20"
                          >
                            <Zap size={16} /> Run new version
                          </button>
                          <button
                            onClick={() => handleRunAnalysis(true)}
                            className="rounded-full border border-amber-400/60 px-3 py-2 text-xs font-semibold text-amber-200 hover:bg-amber-500/10"
                          >
                            Overwrite latest
                          </button>
                        </div>
                      </div>
                    </div>
                    <div className="px-6 py-4">
                      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                        <span className="rounded-full bg-slate-800/60 px-3 py-1">{preview.row_count} rows</span>
                        <span className="rounded-full bg-slate-800/60 px-3 py-1">{preview.column_names.length} columns</span>
                        <span className="rounded-full bg-slate-800/60 px-3 py-1">Select columns below</span>
                      </div>
                      <div className="mb-4">
                        <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-slate-500">
                          <span>Dataset notes</span>
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">Autosaves</span>
                        </div>
                        <textarea
                          value={datasetNotes}
                          onChange={(e) => setDatasetNotes(e.target.value)}
                          placeholder="Add collection context, quirks, or exclusions..."
                          className={`w-full rounded-xl border px-3 py-2 text-sm outline-none ${
                            theme === 'dark'
                              ? 'border-slate-800 bg-slate-900/70 focus:border-cyan-500'
                              : 'border-slate-200 bg-white focus:border-cyan-500'
                          }`}
                          rows={3}
                        />
                      </div>
                      <div className="mb-4 flex flex-wrap gap-2">
                        {preview.column_names.map((name) => {
                          const active = selectedColumns.includes(name);
                          return (
                            <button
                              key={name}
                              onClick={() =>
                                setSelectedColumns((cols) =>
                                  cols.includes(name) ? cols.filter((c) => c !== name) : [...cols, name]
                                )
                              }
                              className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                                active
                                  ? 'border-cyan-500 bg-cyan-500/10 text-cyan-200 shadow-cyan-500/10'
                                  : 'border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700'
                              }`}
                              type="button"
                            >
                              {active ? '✓ ' : ''}
                              {name}
                            </button>
                          );
                        })}
                      </div>
                      <div className="mb-4 rounded-xl border border-slate-800/60 bg-slate-900/40 p-3">
                        <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-slate-500">
                          <span>Rename columns</span>
                          <button
                            onClick={applyRenames}
                            className="rounded-full border border-cyan-500/60 px-2 py-1 text-[11px] font-semibold text-cyan-300"
                          >
                            Apply
                          </button>
                        </div>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                          {preview.column_names.map((name) => (
                            <div key={name} className="flex items-center gap-2 rounded-lg border border-slate-800/60 bg-slate-900/50 px-2 py-1">
                              <span className="text-[11px] text-slate-400">{name}</span>
                              <span className="text-slate-600">→</span>
                              <input
                                value={renameDrafts[name] ?? name}
                                onChange={(e) =>
                                  setRenameDrafts((prev) => ({
                                    ...prev,
                                    [name]: e.target.value,
                                  }))
                                }
                                className="w-full rounded-md bg-slate-950/80 px-2 py-1 text-sm text-slate-100 outline-none"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                          <thead className="bg-slate-900/60">
                            <tr>
                              {preview.column_names.map((c) => (
                                <th key={c} className="px-4 py-3 font-semibold text-slate-400">
                                  {c}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800">
                            {preview.preview_data.slice(0, 5).map((row, i) => (
                              <tr key={i} className="hover:bg-white/5">
                                {preview.column_names.map((c) => (
                                  <td key={c} className="px-4 py-3 font-mono text-xs text-slate-200">
                                    {String((row as Record<string, unknown>)[c])}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                <div className={`rounded-2xl border p-6 shadow-lg ${theme === 'dark' ? 'border-slate-800/80 bg-slate-900/80' : 'border-slate-200 bg-white'}`}>
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <div>
                      <h3 className="text-lg font-semibold">Provider credentials</h3>
                      <p className="text-sm text-slate-400">Store API keys for your preferred provider. Keys stay per-user.</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowKeys((v) => !v)}
                        className="rounded-full border border-slate-700 px-3 py-1 text-xs font-semibold text-slate-200"
                      >
                        {showKeys ? 'Hide keys' : 'Show keys'}
                      </button>
                      <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                        {configuredProviders.length ? `${configuredProviders.length} configured` : 'None configured'}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">OpenAI</span>
                      <input
                        value={credentialInputs.openai}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, openai: e.target.value }))}
                        type={showKeys ? 'text' : 'password'}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="OPENAI_API_KEY"
                      />
                    </label>
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">Anthropic</span>
                      <input
                        value={credentialInputs.anthropic}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, anthropic: e.target.value }))}
                        type={showKeys ? 'text' : 'password'}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="ANTHROPIC_API_KEY"
                      />
                    </label>
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">Gemini/Google</span>
                      <input
                        value={credentialInputs.gemini}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, gemini: e.target.value, google: e.target.value }))}
                        type={showKeys ? 'text' : 'password'}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="GEMINI_API_KEY"
                      />
                    </label>
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">Groq</span>
                      <input
                        value={credentialInputs.groq}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, groq: e.target.value }))}
                        type={showKeys ? 'text' : 'password'}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="GROQ_API_KEY"
                      />
                    </label>
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">Ollama base URL</span>
                      <input
                        value={credentialInputs.ollama_base_url}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, ollama_base_url: e.target.value }))}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="http://localhost:11434/v1"
                      />
                    </label>
                    <label className="text-sm text-slate-300">
                      <span className="text-xs uppercase tracking-[0.18em] text-slate-500">Ollama default model</span>
                      <input
                        value={credentialInputs.ollama_default_model}
                        onChange={(e) => setCredentialInputs((prev) => ({ ...prev, ollama_default_model: e.target.value }))}
                        className="mt-1 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm outline-none"
                        placeholder="deepseek-r1:8b"
                      />
                    </label>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <div className="text-xs text-slate-400">
                      Configured: {configuredProviders.length ? configuredProviders.join(', ') : 'None'}
                    </div>
                    <button
                      onClick={saveCredentials}
                      className="rounded-full bg-gradient-to-r from-cyan-400 to-blue-600 px-4 py-2 text-sm font-semibold text-slate-950"
                    >
                      Save keys
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'analysis' && (
              <div className="space-y-6">
                <div className="flex flex-col gap-2">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Analysis</p>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-3xl font-bold tracking-tight">Results & diagnostics</h2>
                    {analysisResults && (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleExport('pdf')}
                          disabled={exporting !== null}
                          className="flex items-center gap-2 rounded-full border border-slate-700/80 px-4 py-2 text-xs font-semibold text-slate-100 hover:border-cyan-500"
                        >
                          <Download size={14} /> PDF
                        </button>
                        <button
                          onClick={() => handleExport('docx')}
                          disabled={exporting !== null}
                          className="flex items-center gap-2 rounded-full border border-slate-700/80 px-4 py-2 text-xs font-semibold text-slate-100 hover:border-cyan-500"
                        >
                          <Download size={14} /> Word
                        </button>
                        <button
                          onClick={() => handleExport('csv')}
                          disabled={exporting !== null}
                          className="flex items-center gap-2 rounded-full border border-slate-700/80 px-4 py-2 text-xs font-semibold text-slate-100 hover:border-cyan-500"
                        >
                          <Download size={14} /> CSV
                        </button>
                      </div>
                    )}
                  </div>
                  <p className="text-slate-400">Live status, executive summary, and visuals from the latest run.</p>
                </div>

                {!analysisId && !analysisResults ? (
                  <div className="flex min-h-[220px] flex-col items-center justify-center rounded-3xl border border-dashed border-slate-800/70 bg-slate-900/60 text-center">
                    <BarChart3 size={48} className="mb-3 text-slate-700" />
                    <p className="text-slate-500">Select a dataset and run analysis to see results here.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
                    <div className={`xl:col-span-2 rounded-2xl border p-6 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Executive summary</p>
                          <h3 className="text-xl font-bold">{analysisResults?.dataset_name || 'Latest run'}</h3>
                        </div>
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-bold ${
                            streaming
                              ? 'bg-amber-500/20 text-amber-300'
                              : analysisStatus?.status === 'completed' || analysisResults
                                ? 'bg-emerald-500/20 text-emerald-300'
                                : 'bg-slate-800 text-slate-300'
                          }`}
                        >
                          {streaming ? 'Running' : analysisStatus?.status || analysisResults?.status || 'Ready'}
                        </span>
                      </div>
                      <div className={`prose max-w-none text-base leading-relaxed ${theme === 'dark' ? 'prose-invert text-slate-200' : 'text-slate-800'}`}>
                        {summaryText}
                      </div>
                      <div className="mt-4">
                        <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-500">
                          <span>Comment</span>
                          {analysisResults?.version && (
                            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">v{analysisResults.version}</span>
                          )}
                        </div>
                        <textarea
                          value={commentDraft}
                          onChange={(e) => setCommentDraft(e.target.value)}
                          placeholder="Add context or reviewer notes..."
                          className={`w-full rounded-xl border px-3 py-2 text-sm outline-none ${
                            theme === 'dark' ? 'border-slate-800 bg-slate-900/70 focus:border-cyan-500' : 'border-slate-200 bg-white focus:border-cyan-500'
                          }`}
                          rows={3}
                        />
                      </div>
                    </div>

                    <div className="space-y-4">
                    <div className={`rounded-2xl border p-4 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                      <div className="flex items-center justify-between">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Version history</p>
                        <div className="flex items-center gap-2 text-xs">
                          <label className="flex items-center gap-1 text-slate-400">
                              <input type="checkbox" checked={overwriteLatest} onChange={(e) => setOverwriteLatest(e.target.checked)} />
                              Overwrite latest
                            </label>
                            <button onClick={() => loadAnalyses(selectedDatasetId, false)} className="rounded-full border px-2 py-1 text-[11px] font-semibold text-cyan-300">
                              Refresh
                            </button>
                          </div>
                        </div>
                        <div className="mt-3 space-y-2 max-h-64 overflow-y-auto pr-1">
                          {analysisHistory.length ? (
                            analysisHistory.map((item) => (
                              <div
                                key={item.id}
                                className={`flex items-start gap-2 rounded-xl border px-3 py-2 text-sm ${
                                  item.id === analysisId ? 'border-cyan-500 bg-cyan-500/10' : theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'
                                }`}
                              >
                                <div className="flex-1">
                                  <div className="flex items-center justify-between text-xs text-slate-400">
                                    <span className="font-semibold text-slate-200">v{item.version}</span>
                                    <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] uppercase text-slate-300">
                                      {item.status}
                                    </span>
                                  </div>
                                  <button
                                    onClick={() => openAnalysis(item)}
                                    className="text-left text-sm font-semibold text-slate-100 hover:text-cyan-300"
                                  >
                                    {item.summary || 'Open run'}
                                  </button>
                                  <div className="mt-1 text-[11px] text-slate-400">{item.comment || 'No comment yet.'}</div>
                                  {item.superseded_at && <div className="text-[10px] uppercase text-amber-400">Superseded</div>}
                                </div>
                                <button
                                  onClick={() => handleDeleteAnalysis(item.id)}
                                  className="rounded-full border border-red-500/50 px-2 py-1 text-[10px] uppercase text-red-300"
                                >
                                  Delete
                                </button>
                              </div>
                            ))
                          ) : (
                            <div className="rounded-xl border border-dashed border-slate-800/70 bg-slate-900/50 p-3 text-xs text-slate-400">
                              No runs yet. Kick off an analysis to see versions here.
                            </div>
                          )}
                        </div>
                        <div className="mt-3 text-[11px] text-slate-500">
                          Use <span className="font-semibold text-cyan-300">Run analysis</span> to start a new version, or enable overwrite to supersede the latest run.
                        </div>
                      </div>

                      {analysisId && (
                        <div className={`rounded-2xl border p-4 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Analysis notes</p>
                            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">Autosaves</span>
                          </div>
                          <textarea
                            value={commentDraft}
                            onChange={(e) => setCommentDraft(e.target.value)}
                            placeholder="Add interpretation, caveats, or next steps..."
                            className={`min-h-[120px] w-full rounded-xl border px-3 py-2 text-sm outline-none ${
                              theme === 'dark'
                                ? 'border-slate-800 bg-slate-900/70 focus:border-cyan-500'
                                : 'border-slate-200 bg-white focus:border-cyan-500'
                            }`}
                          />
                        </div>
                      )}

                      <div className={`rounded-2xl border p-4 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Metadata</p>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Model</span>
                            <span className="font-mono text-cyan-300">{analysisResults?.model_name || modelName}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Provider</span>
                            <span className="capitalize text-slate-200">{analysisResults?.provider || provider}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Dataset</span>
                            <span className="truncate text-slate-200">{preview?.original_filename || analysisResults?.dataset_name || '—'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Version</span>
                            <span className="text-slate-200">
                              {analysisResults?.version || analysisStatus?.version || '—'}
                              {analysisStatus?.superseded_at ? ' (superseded)' : ''}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Status</span>
                            <span className="text-emerald-300">{analysisStatus?.status || analysisResults?.status || (streaming ? 'running' : 'idle')}</span>
                          </div>
                        </div>
                      </div>

                      <div className={`rounded-2xl border p-4 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                        <div className="flex items-center justify-between">
                          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Live log</p>
                          <button onClick={loadLog} className="text-xs font-semibold text-cyan-300">
                            Fetch latest
                          </button>
                        </div>
                        <pre className="mt-3 max-h-48 overflow-auto rounded-xl bg-slate-950/80 p-3 text-xs text-emerald-200 shadow-inner">
                          {logContent || '// Waiting for execution logs…'}
                        </pre>
                      </div>
                    </div>

                    <div className="xl:col-span-3 grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <div className={`rounded-2xl border p-5 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                        <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
                          <FileText size={16} className="text-indigo-400" />
                          Statistical signals
                        </div>
                        {statsRows.length ? (
                          <div className="overflow-x-auto rounded-xl border border-slate-800/60">
                            <table className="w-full text-sm">
                              <thead className="bg-slate-900/60 text-left text-xs uppercase text-slate-500">
                                <tr>
                                  <th className="px-4 py-3">Metric</th>
                                  <th className="px-4 py-3">P-value</th>
                                  <th className="px-4 py-3">Effect size</th>
                                  <th className="px-4 py-3">Callout</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-800/70">
                                {statsRows.map((row) => (
                                  <tr key={row.name} className="hover:bg-white/5">
                                    <td className="px-4 py-3 font-semibold text-slate-100">{row.name}</td>
                                    <td className="px-4 py-3 font-mono text-xs">{typeof row.pValue === 'number' ? row.pValue.toFixed(4) : '—'}</td>
                                    <td className="px-4 py-3">
                                      {typeof row.effectSize === 'number' ? (
                                        <span className="rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-200">
                                          {row.effectSize.toFixed(3)}
                                        </span>
                                      ) : (
                                        '—'
                                      )}
                                    </td>
                                    <td className="px-4 py-3 text-xs text-slate-400">
                                      {typeof row.pValue === 'number' ? (row.pValue < 0.05 ? 'Significant' : 'Not significant') : 'Exploratory'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed border-slate-800/70 bg-slate-900/50 p-6 text-sm text-slate-400">
                            Run an analysis to view probabilities and effect sizes.
                          </div>
                        )}
                      </div>

                      <div className={`rounded-2xl border p-5 shadow-lg ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                        <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
                          <Activity size={16} className="text-cyan-400" />
                          Plots
                        </div>
                        {plots.length ? (
                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            {plots.map((plot, idx) => (
                              <div key={idx} className="overflow-hidden rounded-xl border border-slate-800/70 bg-slate-900/60">
                                <div className="border-b border-slate-800/70 px-3 py-2 text-xs font-semibold text-slate-200">
                                  {plot.title}
                                </div>
                                <img
                                  src={`data:image/png;base64,${plot.image_base64}`}
                                  alt={plot.title}
                                  className="h-48 w-full object-cover"
                                />
                                {plot.description && <div className="px-3 py-2 text-xs text-slate-400">{plot.description}</div>}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed border-slate-800/70 bg-slate-900/50 p-6 text-sm text-slate-400">
                            Visuals will appear after the next run.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'logs' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Logs</p>
                    <h2 className="text-3xl font-bold">System stream</h2>
                    <p className="text-slate-400">Raw execution output from the statistical agent.</p>
                  </div>
                  <button onClick={loadLog} className="rounded-full border border-slate-800 px-3 py-1 text-xs font-semibold text-cyan-300">
                    Fetch latest
                  </button>
                </div>
                <pre className={`h-[70vh] overflow-auto rounded-2xl border p-6 text-xs leading-relaxed ${
                  theme === 'dark'
                    ? 'border-slate-800 bg-slate-950 text-emerald-200'
                    : 'border-slate-200 bg-slate-900 text-slate-50'
                }`}>
                  {logContent || '// Waiting for execution logs...'}
                </pre>
              </div>
            )}
          </div>

          {/* Right panel */}
          {isRightPanelOpen && (
            <aside
              className={`hidden w-80 border-l p-5 backdrop-blur xl:block ${
                theme === 'dark' ? 'border-slate-800 bg-slate-950/70' : 'border-slate-200 bg-white/70'
              }`}
            >
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Agent decisions</p>
                  <h4 className="text-lg font-bold">Live trace</h4>
                </div>
                <button
                  onClick={() => setRightPanelOpen(false)}
                  className={`rounded-full border p-2 text-slate-500 transition ${
                    theme === 'dark' ? 'border-slate-800 hover:bg-slate-800' : 'border-slate-200 hover:bg-slate-100'
                  }`}
                  aria-label="Close panel"
                >
                  <LayoutDashboard size={16} />
                </button>
              </div>

              <div className="relative space-y-4">
                {trace.length ? (
                  trace.map((step, idx) => {
                    const isCurrent = idx === trace.length - 1;
                    return (
                      <div key={`${step.step}-${idx}`} className="relative pl-6">
                        {idx !== trace.length - 1 && <div className="absolute left-[10px] top-5 h-full w-[1px] bg-slate-800" />}
                        <div
                          className={`absolute left-0 top-0 flex h-6 w-6 items-center justify-center rounded-full border-2 ${
                            isCurrent && streaming
                              ? 'border-amber-400 bg-amber-500/10'
                              : 'border-cyan-400 bg-cyan-500/10'
                          }`}
                        >
                          <CheckCircle2 size={12} className={isCurrent && streaming ? 'text-amber-300 animate-pulse' : 'text-cyan-300'} />
                        </div>
                        <div className={`rounded-xl border p-3 ${theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'}`}>
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold text-slate-100">{step.step}</span>
                            <div className="flex items-center gap-2 text-[11px] text-slate-400">
                              {step.timestamp && <span>{formatTimestamp(step.timestamp)}</span>}
                              {typeof step.p_value === 'number' && <span>p={step.p_value.toFixed(3)}</span>}
                            </div>
                          </div>
                          {step.detail && <p className="mt-1 text-xs text-slate-400">{step.detail}</p>}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-800/70 bg-slate-900/60 p-6 text-center text-sm text-slate-500">
                    Waiting for live trace…
                  </div>
                )}
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}

const NavItem = ({ icon, label, active, isOpen, onClick, theme }: NavItemProps) => (
  <button
    onClick={onClick}
    className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm font-semibold transition ${
      active
        ? theme === 'dark'
          ? 'bg-cyan-500/10 text-cyan-300'
          : 'bg-cyan-50 text-cyan-600'
        : theme === 'dark'
          ? 'text-slate-400 hover:bg-slate-800 hover:text-white'
          : 'text-slate-500 hover:bg-slate-100'
    }`}
  >
    <span className={active ? 'scale-110' : ''}>{icon}</span>
    {isOpen && <span>{label}</span>}
    {active && isOpen && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]" />}
  </button>
);

export default App;
