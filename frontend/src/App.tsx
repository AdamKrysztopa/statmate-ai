import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClient, AnalysisResult, AnalysisStatus, DatasetPreview, Dataset, TraceStep, User } from './api/client';
import { useTheme } from './hooks/useTheme';

const resolveDefaultApi = () => {
  const envBase = import.meta.env.VITE_API_BASE;
  if (envBase) return envBase;

  if (typeof window === 'undefined') return 'http://localhost:8000/api/v1';

  const { protocol, hostname, port } = window.location;

  // GitHub Codespaces/VS Code remote: ports encoded in subdomain (e.g., -3000 → -8000)
  if (hostname.endsWith('.app.github.dev')) {
    return `${protocol}//${hostname.replace(/-\d+\.app\.github\.dev$/, '-8000.app.github.dev')}/api/v1`;
  }

  // Local/dev servers: swap current port for API port
  if (port) {
    return `${protocol}//${hostname}:8000/api/v1`;
  }

  return `${protocol}//${hostname}:8000/api/v1`;
};

const defaultApi = resolveDefaultApi();

function App() {
  const [theme, toggleTheme] = useTheme();
  const [apiBase, setApiBase] = useState(defaultApi);
  const [token, setToken] = useState<string | undefined>(undefined);
  const [user, setUser] = useState<User | undefined>();
  const [health, setHealth] = useState<string>('');
  const [error, setError] = useState<string>('');

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [preview, setPreview] = useState<DatasetPreview | undefined>();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [registering, setRegistering] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [description, setDescription] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const [analysisId, setAnalysisId] = useState<string>('');
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | undefined>();
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | undefined>();
  const [logContent, setLogContent] = useState<string>('');
  const [loadingLog, setLoadingLog] = useState(false);
  const [streamSteps, setStreamSteps] = useState<TraceStep[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerDismissed, setViewerDismissed] = useState(false);
  const [lastTraceUpdate, setLastTraceUpdate] = useState<string | null>(null);

  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [modelName, setModelName] = useState('gpt-4o');
  const [provider, setProvider] = useState('openai');
  const [exporting, setExporting] = useState<'pdf' | 'docx' | 'csv' | null>(null);

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
      const me = await api.withToken(res.access_token).me();
      setUser(me);
      await loadDatasets(api.withToken(res.access_token));
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
      setHealth('Account created. You can log in now.');
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadDatasets = async (client = api) => {
    if (!token && client.token === undefined) return;
    try {
      const list = await client.datasets();
      setDatasets(list);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const selectDataset = async (id: string) => {
    setSelectedDataset(id);
    setAnalysisResults(undefined);
    setAnalysisStatus(undefined);
    setAnalysisId('');
    setLogContent('');
    setViewerOpen(false);
    setViewerDismissed(false);
    if (!id) return;
    try {
      const p = await api.previewDataset(id);
      setPreview(p);
      setSelectedColumns([]);
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
      setHealth(`Uploaded dataset ${result.dataset_id.slice(0, 8)}…`);
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

  const handleRunAnalysis = async () => {
    if (!selectedDataset) {
      setError('Pick a dataset first');
      return;
    }
    clearMessages();
    try {
      const { id } = await api.runAnalysis({
        dataset_id: selectedDataset,
        selected_columns: selectedColumns.length ? selectedColumns : undefined,
        model_name: modelName || undefined,
        provider: provider || undefined,
      });
      streamAbortRef.current?.abort();
      setStreamSteps([]);
      setStreamError(null);
      setStreaming(true);
      setAnalysisId(id);
      setAnalysisStatus({ id, status: 'pending' });
      setAnalysisResults(undefined);
      setLogContent('');
      setViewerDismissed(false);
      setViewerOpen(true);
      pollLog(true);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const refreshStatus = async () => {
    if (!analysisId) return;
    try {
      const status = await api.analysisStatus(analysisId);
      setAnalysisStatus(status);
      if (status.decision_steps?.length) {
        setStreamSteps((prev) => {
          const seen = new Set(prev.map((s) => `${s.timestamp || ''}-${s.step}`));
          const merged = [...prev];
          status.decision_steps?.forEach((step) => {
            const key = `${step.timestamp || ''}-${step.step}`;
            if (!seen.has(key)) {
              seen.add(key);
              merged.push(step);
            }
          });
          return merged;
        });
      }
      if (status.intermediate_log && status.intermediate_log.length > logContent.length) {
        setLogContent(status.intermediate_log);
      }
      if (status.status === 'completed') {
        const results = await api.analysisResults(analysisId);
        setAnalysisResults(results);
        setStreaming(false);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const pollLog = async (silent = false) => {
    if (!analysisId) return;
    try {
      const log = await api.analysisLog(analysisId);
      setLogContent(log.log_content);
    } catch (e) {
      const message = (e as Error).message || '';
      if (!silent && !message.toLowerCase().includes('not found')) {
        setError(message);
      }
    }
  };

  const loadLog = async () => {
    if (!analysisId) return;
    setLoadingLog(true);
    try {
      await pollLog();
    } finally {
      setLoadingLog(false);
    }
  };

  const handleExport = async (format: 'pdf' | 'docx' | 'csv') => {
    const id = analysisId || analysisResults?.id;
    if (!id) {
      setError('Run an analysis to export results.');
      return;
    }
    try {
      setExporting(format);
      const blob = await api.exportAnalysis(id, format);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analysis-${id}.${format === 'docx' ? 'docx' : format}`;
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

  useEffect(() => {
    const storedToken = localStorage.getItem('statmate-token');
    if (storedToken) {
      setToken(storedToken);
    }
  }, []);

  useEffect(() => {
    if (token) {
      localStorage.setItem('statmate-token', token);
      api.withToken(token)
        .me()
        .then(setUser)
        .catch(() => setToken(undefined));
      loadDatasets(api.withToken(token));
    } else {
      localStorage.removeItem('statmate-token');
      setUser(undefined);
      setDatasets([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (!analysisId) return undefined;
    const tick = () => {
      refreshStatus();
      pollLog(true);
    };
    tick();
    const interval = setInterval(tick, 2000);
    return () => clearInterval(interval);
  }, [analysisId]);

  useEffect(() => {
    setLastTraceUpdate(null);
  }, [analysisId]);

  useEffect(() => {
    if (!analysisId) return undefined;
    const controller = new AbortController();
    streamAbortRef.current = controller;
    setStreamError(null);
    setStreaming(true);
    setStreamSteps([]);
    setLogContent('');

    const connect = async () => {
      try {
        const res = await api.analysisStream(analysisId, controller.signal);
        if (!res.body) {
          setStreaming(false);
          setStreamError('Streaming not supported by the server response');
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let idx = buffer.indexOf('\n\n');
          while (idx !== -1) {
            const raw = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 2);
            if (!raw) {
              idx = buffer.indexOf('\n\n');
              continue;
            }

            const lines = raw.split('\n');
            const eventLine = lines.find((line) => line.startsWith('event:'));
            const dataLine = lines.find((line) => line.startsWith('data:'));
            const eventName = eventLine ? eventLine.replace('event:', '').trim() : 'message';
            const dataText = dataLine ? dataLine.replace('data:', '').trim() : '';

            let payload: any = {};
            try {
              payload = dataText ? JSON.parse(dataText) : {};
            } catch {
              payload = { raw: dataText };
            }

            if (eventName === 'step') {
              setStreamSteps((prev) => {
                const seen = new Set(prev.map((s) => `${s.timestamp || ''}-${s.step}`));
                const key = `${payload.timestamp || ''}-${payload.step}`;
                if (seen.has(key)) return prev;
                return [...prev, payload as TraceStep];
              });
              setLastTraceUpdate(new Date().toLocaleTimeString());
            } else if (eventName === 'log') {
              setLogContent((prev) => `${prev}${payload.chunk || ''}`);
            } else if (eventName === 'done') {
              setStreaming(false);
              refreshStatus();
            }

            idx = buffer.indexOf('\n\n');
          }
        }
      } catch (e) {
        if (!controller.signal.aborted) {
          setStreamError((e as Error).message);
          setStreaming(false);
        }
      }
    };

    connect();
    return () => controller.abort();
  }, [analysisId, api]);

  const resultTrace: TraceStep[] = useMemo(() => {
    if (analysisResults?.decision_steps?.length) return analysisResults.decision_steps;
    if (analysisResults?.execution_trace?.length) return analysisResults.execution_trace;
    if (analysisResults?.results_detail?.execution_trace?.length) return analysisResults.results_detail.execution_trace;
    if (analysisResults?.results_detail?.messages?.length) {
      return analysisResults.results_detail.messages.map((msg, idx) => ({
        step: `Message ${idx + 1}`,
        detail: msg,
        data: {},
      }));
    }
    return [];
  }, [analysisResults]);

  const logTrace: TraceStep[] = useMemo(() => {
    if (!logContent) return [];
    const lines = logContent.split('\n').filter((line) => line.includes('Trace step:'));
    return lines.map((line, idx) => {
      const [, rest] = line.split('Trace step:');
      const [stepPart, detailPart] = rest ? rest.split('|') : [];
      const step = stepPart?.trim() || `Step ${idx + 1}`;
      const detail = detailPart?.trim() || '';
      return { step, detail, data: {} };
    });
  }, [logContent]);

  const statusTrace = analysisStatus?.decision_steps || analysisStatus?.execution_trace || [];

  const liveTrace: TraceStep[] = useMemo(() => {
    const merged: TraceStep[] = [];
    const seen = new Set<string>();

    const addSteps = (steps: TraceStep[]) => {
      steps.forEach((item, idx) => {
        const key = `${item.timestamp || ''}-${item.step}-${item.detail}`;
        if (seen.has(key)) return;
        seen.add(key);
        merged.push({
          ...item,
          step: item.step || `Step ${merged.length + 1}`,
          detail: item.detail || (item.data ? JSON.stringify(item.data) : ''),
        });
      });
    };

    addSteps(streamSteps);
    addSteps(statusTrace);
    addSteps(logTrace);
    addSteps(resultTrace);

    return merged;
  }, [streamSteps, statusTrace, logTrace, resultTrace]);

  const plots = useMemo(() => {
    if (!analysisResults) return [];
    return analysisResults.plots || analysisResults.results_detail?.plots || [];
  }, [analysisResults]);

  const effectSizes = useMemo<Record<string, number>>(() => {
    if (!analysisResults) return {};
    return (
      analysisResults.effect_sizes ||
      analysisResults.results_detail?.effect_sizes ||
      {}
    );
  }, [analysisResults]);

  const statsRows = useMemo(
    () => {
      if (!analysisResults) return [];
      const probabilities = analysisResults.probabilities || {};
      const keys = new Set([...Object.keys(probabilities), ...Object.keys(effectSizes)]);
      return Array.from(keys).map((name) => ({
        name,
        pValue: probabilities[name],
        effectSize: effectSizes[name],
      }));
    },
    [analysisResults, effectSizes]
  );

  const agentMessages = useMemo(() => analysisResults?.results_detail?.messages || [], [analysisResults]);

  useEffect(() => {
    if (liveTrace.length) {
      setLastTraceUpdate(new Date().toLocaleTimeString());
    }
  }, [liveTrace]);

  const openViewer = () => {
    setViewerDismissed(false);
    setViewerOpen(true);
  };

  const closeViewer = () => {
    setViewerDismissed(true);
    setViewerOpen(false);
  };

  if (!token) {
    return (
      <div className="app-shell">
        <header className="header">
          <div className="logo">
            <div className="logo-mark">Σ</div>
            <div className="title-block">
              <h1>StatmateAI Frontend</h1>
              <p>Modern React client for FastAPI + LangGraph</p>
            </div>
          </div>
          <div className="controls">
            <button className="button" onClick={toggleTheme} aria-label="Toggle theme">
              {theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
            </button>
            <button className="button" onClick={connect}>
              🔌 Check API
            </button>
          </div>
        </header>

        {error && <div className="toast error">{error}</div>}
        {health && <div className="toast">{health}</div>}

        <section className="card-grid">
          <div className="card">
            <h3>API Connection</h3>
            <p className="muted">Point to your running FastAPI instance.</p>
            <div className="input-group">
              <label>API Base URL</label>
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://localhost:8000/api/v1" />
            </div>
            <div className="pill-row">
              <span className="badge">ENV: {import.meta.env.MODE}</span>
              <span className="badge">Theme: {theme}</span>
              <span className="badge">Auth: required</span>
            </div>
          </div>

          <div className="card">
            <h3>{registering ? 'Create Account' : 'Sign In'}</h3>
            <form onSubmit={registering ? handleRegister : handleLogin}>
              <div className="input-group">
                <label>Email</label>
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
              </div>
              <div className="pill-row">
                <button className="button primary" type="submit">
                  {registering ? 'Create & Login' : 'Login'}
                </button>
                <button className="button" type="button" onClick={() => setRegistering((v) => !v)}>
                  {registering ? 'Have an account? Sign in' : 'Need an account? Register'}
                </button>
              </div>
              <p className="muted">Sign in to access datasets and analysis.</p>
            </form>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="header">
        <div className="logo">
          <div className="logo-mark">Σ</div>
          <div className="title-block">
            <h1>StatmateAI Frontend</h1>
            <p>Modern React client for FastAPI + LangGraph</p>
          </div>
        </div>
        <div className="controls">
          <span className="badge">{user ? `Signed in: ${user.email}` : 'Signed in'}</span>
          <button className="button" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
          </button>
          <button className="button" onClick={connect}>
            🔌 Check API
          </button>
          <button className="button" type="button" onClick={() => setToken(undefined)}>
            Log out
          </button>
        </div>
      </header>

      {error && <div className="toast error">{error}</div>}
      {health && <div className="toast">{health}</div>}

      <section className="card-grid">
        <div className="card">
          <h3>API Connection</h3>
          <p className="muted">Point to your running FastAPI instance.</p>
          <div className="input-group">
            <label>API Base URL</label>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://localhost:8000/api/v1" />
          </div>
          <div className="pill-row">
            <span className="badge">ENV: {import.meta.env.MODE}</span>
            <span className="badge">Theme: {theme}</span>
            <span className="badge">{user ? `Signed in: ${user.email}` : 'Signed in'}</span>
          </div>
        </div>

        <div className="card">
          <h3>Account</h3>
          <p className="muted">You are signed in.</p>
          <div className="pill-row">
            <span className="badge">{user?.email}</span>
            <button className="button" type="button" onClick={() => setToken(undefined)}>
              Log out
            </button>
          </div>
        </div>

        <div className="card">
          <h3>Upload Dataset</h3>
          <div className="input-group">
            <label>File</label>
            <input
              type="file"
              accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet,.md,.doc,.docx"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>
          <div className="input-group">
            <label>Description (optional)</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="Study notes or variables" />
          </div>
          <button className="button primary" disabled={!file || uploading || !token} onClick={handleUpload}>
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
        </div>

        <div className="card">
          <h3>Guided tour</h3>
          <p className="muted">Start with a seeded dataset, watch the live decisions, then export to share.</p>
          <ol className="mini-list">
            <li>Download a sample CSV and upload it.</li>
            <li>Select 2–4 columns to keep the visuals crisp.</li>
            <li>Open the viewer to watch streaming agent steps.</li>
            <li>Export the finished run as PDF/Word/CSV.</li>
          </ol>
          <div className="pill-row">
            <a className="button" href="/samples/clinical_trial_sample.csv" download>
              Clinical sample CSV
            </a>
            <a className="button" href="/samples/marketing_uplift.csv" download>
              Marketing uplift CSV
            </a>
          </div>
        </div>
      </section>

      <div className="section-title">Datasets</div>
      <div className="card">
        <div className="pill-row" style={{ marginBottom: 12 }}>
          <span className="badge">{datasets.length} available</span>
          <button className="button" disabled={!token} onClick={() => loadDatasets()}>
            Refresh list
          </button>
        </div>
        <div className="pill-row">
          {datasets.map((ds) => (
            <button
              key={ds.id}
              className="button"
              style={{
                borderColor: selectedDataset === ds.id ? 'var(--primary)' : 'var(--border)',
                color: selectedDataset === ds.id ? 'var(--primary)' : 'var(--text)',
              }}
              onClick={() => selectDataset(ds.id)}
            >
              {selectedDataset === ds.id ? '✓ ' : ''}
              {ds.original_filename}
            </button>
          ))}
          {!datasets.length && <span className="muted">No datasets yet.</span>}
        </div>

        {preview && (
          <div style={{ marginTop: 16 }}>
            <div className="pill-row">
              <span className="badge">Rows: {preview.row_count}</span>
              <span className="badge">Columns: {preview.column_names.length}</span>
            </div>
            <div className="input-group" style={{ marginTop: 12 }}>
              <label title="Tip: keep 2–4 columns for faster plots and clearer box/scatter views">
                Select columns for analysis (optional)
              </label>
              <div className="pill-row">
                {preview.column_names.map((name) => {
                  const active = selectedColumns.includes(name);
                  return (
                    <button
                      key={name}
                      className="button"
                      style={{ borderColor: active ? 'var(--primary)' : 'var(--border)', color: active ? 'var(--primary)' : 'var(--text)' }}
                      type="button"
                      onClick={() =>
                        setSelectedColumns((cols) => (cols.includes(name) ? cols.filter((c) => c !== name) : [...cols, name]))
                      }
                    >
                      {active ? '✓ ' : ''}
                      {name}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="table-like">
              <table>
                <thead>
                  <tr>
                    {preview.column_names.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview_data.slice(0, 5).map((row, idx) => (
                    <tr key={idx}>
                      {preview.column_names.map((c) => (
                        <td key={c}>{String((row as Record<string, unknown>)[c])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      <div className="section-title">Analysis</div>
      <div className="card-grid">
        <div className="card">
          <h3>Model</h3>
          <div className="input-group">
            <label>Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="google">Gemini</option>
              <option value="groq">Groq</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>
          <div className="input-group">
            <label>Model name</label>
            <input value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="gpt-4o" />
          </div>
          <button className="button primary" disabled={!selectedDataset || !token} onClick={handleRunAnalysis}>
            🚀 Run analysis
          </button>
          {!selectedDataset && <p className="muted">Pick a dataset first.</p>}
        </div>

        <div className="card">
          <h3>Status</h3>
          <div className="input-group">
            <label>Analysis ID</label>
            <input value={analysisId} onChange={(e) => setAnalysisId(e.target.value)} placeholder="auto-filled after run" />
          </div>
          <div className="pill-row">
            <button className="button" disabled={!analysisId} onClick={refreshStatus}>
              Refresh
            </button>
            <span className="badge">{analysisStatus ? `Status: ${analysisStatus.status}` : 'Waiting to start'}</span>
            {analysisResults && (
              <button className="button" type="button" onClick={openViewer}>
                Open detailed view
              </button>
            )}
          </div>
          {analysisStatus?.message && <p className="muted">{analysisStatus.message}</p>}
        </div>
      </div>

      {analysisResults && (
        <div className="card" style={{ marginTop: 12 }}>
          <h3>Results</h3>
          <div className="pill-row" style={{ marginBottom: 12 }}>
            {analysisResults.model_name && <span className="tag">Model: {analysisResults.model_name}</span>}
            {analysisResults.provider && <span className="tag">Provider: {analysisResults.provider}</span>}
          </div>
          {(analysisResults.summary || analysisResults.results_detail?.summary) && (
            <p>{analysisResults.summary || analysisResults.results_detail?.summary}</p>
          )}
          <div className="pill-row" style={{ marginTop: 8 }}>
            <button className="button primary" type="button" onClick={openViewer}>
              Open detailed view
            </button>
          </div>
          <div className="pill-row" style={{ marginTop: 6 }}>
            <button className="button" onClick={() => handleExport('pdf')} disabled={exporting !== null}>
              Export PDF
            </button>
            <button className="button" onClick={() => handleExport('docx')} disabled={exporting !== null}>
              Export Word
            </button>
            <button className="button" onClick={() => handleExport('csv')} disabled={exporting !== null}>
              Export CSV
            </button>
          </div>
        </div>
      )}

      {viewerOpen && (
        <div className="viewer-overlay">
          <div className="viewer-backdrop" onClick={closeViewer} />
          <div className="viewer-panel">
            <div className="viewer-header">
              <div>
                <div className="section-title" style={{ margin: 0 }}>Analysis Detail</div>
                <div className="pill-row" style={{ marginTop: 6 }}>
                  {analysisId && <span className="tag">ID: {analysisId.slice(0, 8)}…</span>}
                  <span className="tag">Status: {analysisStatus?.status || analysisResults?.status || 'pending'}</span>
                  {analysisResults?.model_name && <span className="tag">Model: {analysisResults.model_name}</span>}
                  {analysisResults?.provider && <span className="tag">Provider: {analysisResults.provider}</span>}
                </div>
                <div className="pill-row" style={{ marginTop: 8 }}>
                  <button className="button" onClick={() => handleExport('pdf')} disabled={exporting !== null}>
                    {exporting === 'pdf' ? 'Generating PDF…' : 'Export PDF'}
                  </button>
                  <button className="button" onClick={() => handleExport('docx')} disabled={exporting !== null}>
                    {exporting === 'docx' ? 'Generating DOCX…' : 'Export Word'}
                  </button>
                  <button className="button" onClick={() => handleExport('csv')} disabled={exporting !== null}>
                    {exporting === 'csv' ? 'Generating CSV…' : 'Export CSV'}
                  </button>
                </div>
              </div>
              <div className="pill-row" style={{ gap: 8 }}>
                <button className="button" onClick={() => { refreshStatus(); pollLog(); }}>
                  Refresh now
                </button>
                <button className="button" onClick={closeViewer}>Close</button>
              </div>
            </div>

            {analysisStatus?.message && <p className="muted">{analysisStatus.message}</p>}
            {(analysisResults?.summary || analysisResults?.results_detail?.summary) && (
              <p>{analysisResults?.summary || analysisResults?.results_detail?.summary}</p>
            )}

            <div style={{ marginTop: 12 }}>
              <div className="section-title" style={{ marginTop: 0 }}>
                Live agent steps
              </div>
              <div className="pill-row" style={{ marginBottom: 8 }}>
                <span className="badge">Live trace</span>
                {analysisStatus?.status === 'running' && <span className="badge">Status: running</span>}
                {streaming && <span className="badge">Streaming…</span>}
                {streamError && <span className="badge">Stream error: {streamError}</span>}
                {lastTraceUpdate && <span className="badge">Updated: {lastTraceUpdate}</span>}
              </div>
              {liveTrace.length > 0 ? (
                <div className="trace-grid">
                  {liveTrace.map((item, idx) => (
                    <div className="trace-card" key={`${item.step}-${idx}`}>
                      <div className="trace-header">
                        <span className="badge">{item.step || `Step ${idx + 1}`}</span>
                        <div className="pill-row" style={{ gap: 6 }}>
                          {item.timestamp && <span className="tag">{new Date(item.timestamp).toLocaleTimeString()}</span>}
                          {typeof item.p_value === 'number' && (
                            <span className="tag">p = {item.p_value.toFixed(4)}</span>
                          )}
                        </div>
                      </div>
                      {item.detail && <p className="muted">{item.detail}</p>}
                      {item.data && (
                        <div className="trace-data">
                          {Object.entries(item.data).map(([key, value]) => (
                            <div key={key} className="pill mono">
                              {key}: {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">Waiting for the first agent decision…</p>
              )}
            </div>

            <div style={{ marginTop: 12 }}>
              <div className="section-title" style={{ marginTop: 0 }}>
                Live log
              </div>
              <div className="pill-row" style={{ marginBottom: 8 }}>
                <button className="button" onClick={() => pollLog()} disabled={!analysisId}>
                  Fetch latest log
                </button>
                {loadingLog && <span className="badge">Loading…</span>}
                {analysisStatus?.log_available && <span className="badge">Log streaming</span>}
                <span className="badge">Auto refresh every 2s</span>
              </div>
              <pre className="log-viewer">{logContent || 'Collecting log output…'}</pre>
            </div>

            {analysisResults && (
              <>
                {(statsRows.length > 0 || plots.length > 0) && (
                  <div style={{ marginTop: 12 }}>
                    <div className="section-title" style={{ marginTop: 0 }}>
                      Results & Diagnostics
                    </div>
                    <div className="stats-visual-wrap">
                      {statsRows.length > 0 && (
                        <div className="table-like stats-table">
                          <table>
                            <thead>
                              <tr>
                                <th>Test / Metric</th>
                                <th>P-Value</th>
                                <th>Effect size</th>
                                <th>Callouts</th>
                              </tr>
                            </thead>
                            <tbody>
                              {statsRows.map((row) => (
                                <tr key={row.name}>
                                  <td>{row.name}</td>
                                  <td>{typeof row.pValue === 'number' ? row.pValue.toFixed(4) : '—'}</td>
                                  <td>
                                    {typeof row.effectSize === 'number' ? (
                                      <div className="pill-row" style={{ gap: 4 }}>
                                        <span className="badge">d = {row.effectSize.toFixed(3)}</span>
                                        <span className="badge">
                                          {Math.abs(row.effectSize) >= 0.8
                                            ? 'Large'
                                            : Math.abs(row.effectSize) >= 0.5
                                              ? 'Medium'
                                              : Math.abs(row.effectSize) >= 0.2
                                                ? 'Small'
                                                : 'Trivial'}
                                        </span>
                                      </div>
                                    ) : (
                                      '—'
                                    )}
                                  </td>
                                  <td>
                                    {typeof row.pValue === 'number' ? (
                                      <span className="tag">{row.pValue < 0.05 ? 'Significant' : 'Not significant'}</span>
                                    ) : (
                                      <span className="tag">Exploratory</span>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {plots.length > 0 && (
                        <div className="plot-panel">
                          <div className="plot-grid">
                            {plots.map((plot, idx) => (
                              <div className="plot-card" key={`${plot.title}-${idx}`}>
                                <img src={`data:image/png;base64,${plot.image_base64}`} alt={plot.title} />
                                <div className="plot-meta">
                                  <div className="plot-title">{plot.title}</div>
                                  {plot.description && <p className="muted">{plot.description}</p>}
                                  <div className="pill-row" style={{ gap: 6 }}>
                                    {plot.column && <span className="tag">Column: {plot.column}</span>}
                                    {plot.type && <span className="tag">{plot.type}</span>}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {agentMessages.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div className="section-title" style={{ marginTop: 0 }}>
                      Agent messages
                    </div>
                    <div className="trace-messages">
                      {agentMessages.map((msg, idx) => (
                        <div key={idx} className="message-block">
                          <div className="tag">Message {idx + 1}</div>
                          <p>{msg}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="pill-row" style={{ marginTop: 12 }}>
                  <button className="button" onClick={loadLog} disabled={loadingLog}>
                    {loadingLog ? 'Loading log…' : 'Force reload log'}
                  </button>
                  {logContent && <span className="badge">Log loaded</span>}
                </div>

                {analysisResults.results_detail && (
                  <details style={{ marginTop: 12 }}>
                    <summary>Raw result payload</summary>
                    <pre className="card" style={{ overflow: 'auto', background: 'var(--bg-input)' }}>
{JSON.stringify(analysisResults.results_detail, null, 2)}
                    </pre>
                  </details>
                )}
              </>
            )}
          </div>
        </div>
      )}

      <footer className="footer">
        <span>StatmateAI • React + Vite • FastAPI backend</span>
      </footer>
    </div>
  );
}

export default App;
